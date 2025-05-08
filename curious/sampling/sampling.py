from typing import List, Dict, Tuple
import gc

import torch
from transformers import (
    PreTrainedTokenizer,
    GenerationConfig,
    PreTrainedModel,
)
from accelerate.utils import set_seed
from transformers.generation.utils import GenerateDecoderOnlyOutput

from curious.reward.rule.gsm8k import GSM8KRewardModel
import vllm 

def linear_temperature_annealing(
    current_step: int, 
    total_steps: int, 
    start_temp: float, 
    end_temp: float,
) -> float:
    """
    Computes the linearly decayed temperature at a given step for a linear annealing schedule.

    Args:
        current_step (int): The current step number, starting from 0.
        total_steps (int): The total number of steps for which the temperature is annealed.
        start_temp (float): The starting temperature.
        end_temp (float): The final temperature.

    Returns:
        float: The decayed temperature.
    """
    if total_steps <= 0:
        raise ValueError("total_steps must be greater than 0")
    # Ensure the fraction is between 0 and 1
    fraction = min(max(current_step / total_steps, 0.0), 1.0)
    return start_temp * (1.0 - fraction) + end_temp * fraction

def compute_rewards(
    model: PreTrainedModel,
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
        infos.extend(group_infos)
        solved_masks.append(solved)
    
    return {
        "returns": torch.tensor(returns, dtype=torch.bfloat16, device=model.device), # (num_questions, group_size)
        "solved_masks": torch.tensor(solved_masks, dtype=torch.bfloat16, device=model.device), # (num_questions, group_size)
        "infos": infos, # single list (len = num_questions * group_size)
    }

@torch.compile(dynamic=True)
def compute_learnability(solved_masks: torch.Tensor) -> torch.Tensor:
    """
    Computes the learnability of a group of completions.

    Args:
        solved_masks (torch.Tensor): The solved masks of the group of completions. (num_questions, group_size)

    Returns:
        torch.Tensor: The learnability of the group of completions. (num_questions)
    """
    solved_rate = solved_masks.mean(dim=1)
    return solved_rate * (1 - solved_rate)

@torch.no_grad()
def sample_responses_hf(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    batch_inputs: Dict[str, torch.Tensor],
    generation_config: GenerationConfig,
    seed: int = 42,
) -> Dict[str, torch.Tensor]:
    
    model.eval()
    model.gradient_checkpointing_disable()
    # set the seed
    set_seed(seed)
    pad_token_id = tokenizer.pad_token_id

    sequence_ids = model.generate(
        input_ids=batch_inputs["input_ids"].to(model.device),
        attention_mask=batch_inputs["attention_mask"].to(model.device),
        generation_config=generation_config
    )

    action_mask = torch.zeros_like(sequence_ids, dtype=torch.bool)
    action_mask[:, batch_inputs["input_ids"].shape[1] :] = True
    action_mask[sequence_ids == pad_token_id] = False
    action_mask = action_mask[:, 1:]
    return {
            "num_samples": batch_inputs["input_ids"].shape[0] * generation_config.num_return_sequences,
            "input_ids": batch_inputs["input_ids"],
            "sequence_ids": sequence_ids,
            "action_mask": action_mask,
            "completions": tokenizer.batch_decode(
                sequence_ids[:, batch_inputs["input_ids"].shape[1] :], 
                skip_special_tokens=True
            ),
    }

@torch.no_grad()
def sample_response_vllm(
    inputs:List[str],
    vllm: vllm.LLM,
    sampling_params: vllm.SamplingParams,
) -> List[vllm.outputs.RequestOutput]:

    tokenizer = vllm.get_tokenizer()
    if tokenizer.bos_token:
        # lstrip bos_token because vllm will add it.
        inputs = [text.lstrip(tokenizer.bos_token) for text in inputs]

    return vllm.generate(
        prompts=inputs,
        sampling_params=sampling_params,
        use_tqdm=False,
    )

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
    use_rloo_scalar: bool = False,
    use_vllm: bool = False,
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
    
    if use_vllm:
        # get the sequence ids
        sampled_responses = sample_response_vllm(
            inputs=batch_inputs["input_ids"],
            vllm=vllm,
            sampling_params=sampling_params,
        )
    else:
        # get the sequence ids from huggingface
        sampled_responses = sample_responses_hf(
            model,
            tokenizer,
            batch_inputs,
            generation_config,
            seed=seed,
        )
    
    # get the rewards
    rewards_out = compute_rewards(
        model,
        reward_model,
        completions=sampled_responses["completions"],
        oracle_answers=oracle_answers,
        group_size=generation_config.num_return_sequences,
    )

    advantages = compute_group_advantages(
        returns=rewards_out["returns"],
        normalize=normalize_centered_returns,
        use_rloo_scalar=use_rloo_scalar,
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
            "advantages": advantages, # (num_samples, group_size)
            "num_words_in_completions": torch.tensor(num_words_in_completions, dtype=torch.bfloat16, device="cpu"),
            "completions": completions,
        }
    )

    return rewards_out

@torch.compile(dynamic=True)
def compute_group_advantages(
    returns: torch.Tensor, 
    eps: float = 1e-8, 
    normalize: bool = True, 
    use_rloo_scalar: bool = False,
) -> torch.Tensor:
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
        centered_returns = centered_returns / (centered_returns.std(dim=1, keepdim=True) + eps)

    if use_rloo_scalar:
        # unbiased scalar G / (G-1)
        centered_returns = centered_returns * (returns.shape[1] / (returns.shape[1] - 1))

    return centered_returns    

@torch.compile(dynamic=True)
def _sequence_log_probs_from_logits(
    logits: torch.Tensor, 
    output_ids: torch.Tensor,
    return_entropy: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor | None]:
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
    log_probs = gathered_logits - log_sum_exp

    if return_entropy:
        pd = torch.nn.functional.softmax(logits, dim=-1)
        entropy = log_sum_exp - torch.sum(pd * logits, dim=-1)
        return log_probs, entropy
    else:
        return log_probs, None

@torch.compile(dynamic=True)
def _minibatch_sequence_log_probs_from_logits(
    logits: torch.Tensor, 
    output_ids: torch.Tensor,
    return_entropy: bool = False,
    logits_minibatch_size: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor | None]:
    """
    Computes the log probabilities of the output ids from the logits.
    """
    token_logprobs = []
    if return_entropy:
        token_entropy = []

    for i in range(0, logits.shape[0], logits_minibatch_size):
        logits_rows = logits[i:i+logits_minibatch_size]
        index_rows = output_ids[i:i+logits_minibatch_size]
        token_logprob, entropy = _sequence_log_probs_from_logits(
            logits=logits_rows,
            output_ids=index_rows,
            return_entropy=return_entropy,
        )
        token_logprobs.append(token_logprob)
        if return_entropy:
            token_entropy.append(entropy)

        del logits_rows, index_rows, token_logprob, entropy
        gc.collect()
        torch.cuda.empty_cache()

    return torch.cat(token_logprobs, dim=0), torch.cat(token_entropy, dim=0) if return_entropy else None

@torch.compile(dynamic=True)
def sequences_log_probs(
    model: PreTrainedModel,
    sequence_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    return_entropy: bool = True,
    logits_minibatch_size: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor | None]:
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
    torch.cuda.empty_cache()

    token_logprobs = []
    if return_entropy:
        token_entropy = []

    for i in range(0, sequence_ids.shape[0], logits_minibatch_size):
        mini_ids = sequence_ids[i:i+logits_minibatch_size]
        mini_mask = attention_mask[i:i+logits_minibatch_size]
        mini_output = model(input_ids=mini_ids, attention_mask=mini_mask, use_cache=False)
        mini_logits = mini_output["logits"]

        del mini_mask, mini_output

        # logits: [batch_size * num_rollouts, seq_len, vocab_size]
        log_probs, entropy = _sequence_log_probs_from_logits(
            logits=mini_logits[:, :-1],
            output_ids = mini_ids[:, 1:],  # right shift 1 block to get the actual output ids
            return_entropy=return_entropy,
        )
        del mini_logits, mini_ids
        gc.collect()
        torch.cuda.empty_cache()

        token_logprobs.append(log_probs)
        if return_entropy:
            token_entropy.append(entropy)
            
    return torch.cat(token_logprobs, dim=0), torch.cat(token_entropy, dim=0) if return_entropy else None
