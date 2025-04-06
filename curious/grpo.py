from typing import List, Tuple, Dict

import torch
from transformers import (
    PreTrainedTokenizer,
    GenerationConfig,
    PreTrainedModel,
)
from curious.reward import GSM8KRewardModel

from transformers.generation.utils import GenerateDecoderOnlyOutput

@torch.no_grad()
def rollout(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    batch_inputs: Dict[str, torch.Tensor],
    reward_model: GSM8KRewardModel,
    generation_config: GenerationConfig,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
    """
    Performs a rollout of the model.

    Args:
        model (AutoModelForCausalLM): The model to rollout.
        tokenizer (PreTrainedTokenizer): The tokenizer to use for the model.
        inputs (dict[str, torch.Tensor]): The inputs to the model.
        oracle_answer (str): The oracle answer to the task.
        generation_config (GenerationConfig): The generation configuration to use for the rollout.

    Returns:
        tuple: A tuple containing the sequence ids, the returns, the action mask, and the completions.
    """
    # get the batch size
    num_samples = batch_inputs["input_ids"].shape[0]
    oracle_answers = batch_inputs.pop("oracle_answer")
    num_return_sequences = generation_config.num_return_sequences
    # num_rollouts = num_samples * num_return_sequences

    # get the parallel rollouts
    pad_token_id = tokenizer.eos_token_id
    sequence_ids = model.generate(**batch_inputs, generation_config=generation_config)
    
    # get the completions
    completions = tokenizer.batch_decode(
        sequence_ids[:, batch_inputs["input_ids"].shape[1] :], 
        skip_special_tokens=True
    )

    # action mask (state that has performed an action = 1)
    action_mask = torch.zeros_like(sequence_ids, dtype=torch.bool)
    action_mask[:, batch_inputs["input_ids"].shape[1] :] = True
    action_mask[sequence_ids == pad_token_id] = False
    action_mask = action_mask[:, 1:]  # (num_samples * group_size, seq_len-1)

    # outputs
    num_words_in_completions = [len(completion.split(' ')) for completion in completions]
    returns = torch.zeros(
        (num_samples, num_return_sequences),
        dtype=torch.float,
        device="cpu",
    )
    solved_rates = torch.zeros(
        (num_samples,), 
        dtype=torch.float, 
        device="cpu"
    )
    info_list = []
    
    # compute the rewards
    for i in range(0, len(completions), num_return_sequences):

        question_idx = i // num_return_sequences

        group_completions = completions[i : i + num_return_sequences]
        orcale_answer_replicates = [oracle_answers[question_idx]] * num_return_sequences
        
        rewards, infos, solved_rate = reward_model(
            group_completions,
            orcale_answer_replicates,
        )
        returns[question_idx, :] = torch.tensor(
            rewards, 
            dtype=torch.float, 
            device="cpu",
        )
        solved_rates[question_idx] = solved_rate
        info_list.extend(infos)

    return sequence_ids, returns, solved_rates, action_mask, completions, info_list, num_words_in_completions

@torch.compile(dynamic=True)
def group_advantages(returns: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Normalizes the advantages of a group of returns.

    Args:
        returns (torch.Tensor): The returns to normalize. (num_samples, num_return_sequences)
        eps (float): The epsilon value to add to the standard deviation to prevent division by zero.

    Returns:
        torch.Tensor: The normalized advantages. (num_samples, num_return_sequences)
    """
    return (returns - returns.mean(dim=1, keepdim=True)) / (returns.std(dim=1, keepdim=True) + eps)

@torch.compile(dynamic=True)
def sequence_log_probs_from_logits(
    logits: torch.Tensor, output_ids: torch.Tensor
) -> torch.Tensor:
    """
    Computes the log probabilities of the output ids from the logits.

    Args:
        logits (torch.Tensor): The logits of the model. (num_samples * num_rollouts, seq_len, vocab_size)
        output_ids (torch.Tensor): The output ids to compute the log probabilities for. (num_samples * num_rollouts, seq_len)

    Returns:
        torch.Tensor: The log probabilities of the output ids. (num_samples * num_rollouts, seq_len)
    """
    # Gather the logits corresponding to the output_ids
    gathered_logits = logits.gather(dim=-1, index=output_ids.unsqueeze(-1)).squeeze(-1)  # (num_samples * num_rollouts, seq_len)

    # Compute log-sum-exp over the vocabulary dimension
    log_sum_exp = logits.logsumexp(dim=-1)  # (num_samples * num_rollouts, seq_len)
    return gathered_logits - log_sum_exp

@torch.compile(dynamic=True)
def sequences_log_probs(
    model: PreTrainedModel,
    sequence_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Computes the log probabilities of the output ids from the logits.

    Args:
        model (PreTrainedModel): The model to compute the log probabilities for.
        sequence_ids (torch.Tensor): The sequence ids to compute the log probabilities for.
            (num_samples * group_size, seq_len)
        attention_mask (torch.Tensor): The attention mask to compute the log probabilities for.
            (num_samples * group_size, seq_len)

    Returns:
        torch.Tensor: The log probabilities of the output ids. (num_samples * group_size, seq_len-1, vocab_size)
    """
    #position_ids = attention_mask.long().cumsum(dim=-1) - 1
    #position_ids.masked_fill_(mask=(attention_mask == 0), value=1)
    output = model(
        input_ids=sequence_ids,
        attention_mask=attention_mask,
        #position_ids=position_ids,
        use_cache=False,
    )
    logits = output["logits"]
    del output 

    # logits: [batch_size * num_rollouts, seq_len, vocab_size]
    log_probs = sequence_log_probs_from_logits(
        logits=logits[:, :-1],
        output_ids=sequence_ids[:, 1:],  # right shift 1 block to get the actual output ids
    )
    del logits
    return log_probs

def sequences_log_probs_with_mask(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    generation_output: GenerateDecoderOnlyOutput,
) -> torch.Tensor:
    """
    Computes the log probabilities of a sequence of tokens, given the output of a generate() call from a HuggingFace model.

    Args:
        model (PreTrainedModel): The model to use for computing the log probabilities.
        tokenizer (PreTrainedTokenizer): The tokenizer to use for computing the log probabilities.
        generation_output (GenerateDecoderOnlyOutput): The output of the generate() call.

    Returns:
        logprobs (torch.Tensor): The log probabilities of the sequence of tokens. (num_seqs, seq_len)
        action_mask (torch.Tensor): A boolean mask indicating where the action was taken. (num_seqs, seq_len)
    """
    seq_ids = generation_output.sequences 
    len_inputs = seq_ids.shape[1] - len(generation_output.logits)

    pad_token_id = tokenizer.eos_token_id 

    logprobs = torch.zeros_like(
        seq_ids, 
        dtype= torch.float, 
        device= seq_ids.device,
    )
    action_mask = torch.zeros_like(
        seq_ids, 
        dtype=torch.bool,
        device=seq_ids.device
    )
    action_mask[:, seq_ids.shape[1] :] = True
    action_mask[seq_ids == pad_token_id] = False
    # action_mask = action_mask[:, 1:]  
    # (num_seqs, seq_len-1)
    generation_logprobs = model.compute_transition_scores(
        generation_output.sequences, 
        generation_output.logits, 
        normalize_logits=True, 
    )
    # (num_seqs, generation_length)
    logprobs[:, len_inputs:] = generation_logprobs
    masked_logprobs = logprobs * action_mask
    return masked_logprobs 