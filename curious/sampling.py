from typing import List, Dict

import torch
from transformers import (
    PreTrainedTokenizer,
    GenerationConfig,
    PreTrainedModel,
)
from accelerate.utils import set_seed
from transformers.generation.utils import GenerateDecoderOnlyOutput

from curious.reward import GSM8KRewardModel

@torch.no_grad()
def sample_responses(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    batch_inputs: Dict[str, torch.Tensor],
    generation_config: GenerationConfig,
    seed: int = 42,
) -> Dict[str, torch.Tensor]:
    
    # set the seed
    set_seed(seed)
    group_size = generation_config.num_return_sequences

    num_samples = batch_inputs["input_ids"].shape[0]
    pad_token_id = tokenizer.eos_token_id

    # get the sequence ids
    sequence_ids = model.generate(
        input_ids=batch_inputs["input_ids"].to(model.device),
        attention_mask=batch_inputs["attention_mask"].to(model.device),
        generation_config=generation_config
    )

    # action mask (state that has performed an action = 1)
    # interpretation: an `action` has been performed to produce the token at the current position
    action_mask = torch.zeros_like(sequence_ids, dtype=torch.bool)
    action_mask[:, batch_inputs["input_ids"].shape[1] :] = True
    action_mask[sequence_ids == pad_token_id] = False
    action_mask = action_mask[:, 1:]  # (num_samples * group_size, seq_len-1)

    return {
        "num_samples": num_samples * group_size,
        "input_ids": batch_inputs["input_ids"],
        "sequence_ids": sequence_ids,
        "action_mask": action_mask,
        "completions": tokenizer.batch_decode(
            sequence_ids[:, batch_inputs["input_ids"].shape[1] :], 
            skip_special_tokens=True
        ),
    }


@torch.no_grad()
def rollout(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    batch_inputs: Dict[str, torch.Tensor],
    reward_model: GSM8KRewardModel,
    generation_config: GenerationConfig,
    group_size: int,
    seed: int = 42,
    normalize_centered_returns: bool = False,
) -> Dict[str, torch.Tensor]:
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
    assert generation_config.num_return_sequences == group_size, \
    f"The number of return sequences must be equal to the group size: {generation_config.num_return_sequences} != {group_size}"

    # get the batch size
    oracle_answers = batch_inputs["oracle_answer"]
    
    # get the sequence ids
    sampled_responses = sample_responses(
        model,
        tokenizer,
        batch_inputs,
        generation_config,
        seed=seed,
    )
    
    # get the rewards
    rewards_out = compute_rewards(
        reward_model,
        completions=sampled_responses["completions"],
        oracle_answers=oracle_answers,
        group_size=generation_config.num_return_sequences,
    )

    advantages = compute_group_advantages(
        returns=rewards_out["returns"],
        normalize=normalize_centered_returns,
    )

    # outputs
    completions = sampled_responses["completions"]
    num_words_in_completions = [len(completion.split(' ')) for completion in completions]

    rewards_out.update(
        {
            "num_samples": batch_inputs["input_ids"].shape[0] * generation_config.num_return_sequences,
            "num_samples_per_group": generation_config.num_return_sequences,
            "sequence_ids": sampled_responses["sequence_ids"],
            "action_mask": sampled_responses["action_mask"],
            "advantages": advantages,
            "num_words_in_completions": torch.IntTensor(num_words_in_completions, device="cpu"),
            "completions": completions,
        }
    )

    return rewards_out


def compute_rewards(
    reward_model: GSM8KRewardModel,
    completions: List[str],
    oracle_answers: List[str],
    group_size: int,
) -> Dict[str, torch.Tensor]:
    """
    Computes the rewards for a list of completions and oracle answers.
    """
    assert int(len(completions) / len(oracle_answers)) == group_size, \
    f"The number of completions must be equal to the number of oracle answers times the group size: {len(completions)} != {len(oracle_answers)} * {group_size}"

    returns = []
    infos = []
    solved_masks = []

    for begin_idx in range(0, len(completions), group_size):

        end_idx = begin_idx + group_size
        group_completions = completions[begin_idx:end_idx]

        group_idx = begin_idx // group_size
        group_oracle_answers = [oracle_answers[group_idx]] * group_size
        
        group_rewards, group_infos, solved = reward_model(
            group_completions,
            group_oracle_answers,
        )

        returns.append(group_rewards)
        infos.append(group_infos)
        solved_masks.append(solved)
    
    return {
        "returns": torch.FloatTensor(returns, device="cpu"), # (num_questions, group_size)
        "solved_masks": torch.FloatTensor(solved_masks, device="cpu"), # (num_questions, group_size)
        "infos": infos, # double list (first level: question, second level: group)
    }

@torch.compile(dynamic=True)
def compute_group_advantages(returns: torch.Tensor, eps: float = 1e-8, normalize: bool = True) -> torch.Tensor:
    """
    Normalizes the advantages of a group of returns.

    Args:
        returns (torch.Tensor): The returns to normalize. (num_samples, num_return_sequences)
        eps (float): The epsilon value to add to the standard deviation to prevent division by zero.

    Returns:
        torch.Tensor: The normalized advantages. (num_samples, num_return_sequences)
    """
    centered_returns = returns - returns.mean(dim=1, keepdim=True)
    if normalize:
        return centered_returns / (centered_returns.std(dim=1, keepdim=True) + eps)
    else:
        return centered_returns


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
    output = model(input_ids=sequence_ids, attention_mask=attention_mask)
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