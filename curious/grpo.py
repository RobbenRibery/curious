from collections.abc import Callable
import json
from pathlib import Path
import random
import re
from typing import Any, Iterator, Optional, List, Tuple
import wandb
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from transformers import (
    PreTrainedTokenizer,
    GenerationConfig,
    PreTrainedModel, 
)
from curious.reward import RewardModel

@torch.no_grad()
def rollout(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    batch_inputs: dict[str, torch.Tensor],
    oracle_answers: List[str],
    generation_config: GenerationConfig,
) -> Tuple[ 
        torch.Tensor, 
        torch.Tensor, 
        torch.Tensor, 
        torch.Tensor, 
        List[str]
    ]:
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
    # assume model.eval()

    # get the batch size
    num_samples = batch_inputs["input_ids"].shape[0]
    num_return_sequences = generation_config.num_return_sequences
    num_rollouts = num_samples * num_return_sequences

    # get the parallel rollouts
    pad_token_id = tokenizer.eos_token_id
    sequence_ids = model.generate(**batch_inputs, generation_config=generation_config)
    completions = tokenizer.batch_decode(
        sequence_ids[:, batch_inputs["input_ids"].shape[1] :], 
        skip_special_tokens=True
    )

    # action mask (state that has performed an action = 1)
    action_mask = torch.zeros_like(sequence_ids, dtype=torch.bool)
    action_mask[:, batch_inputs["input_ids"].shape[1]:] = True
    action_mask[sequence_ids == pad_token_id] = False
    action_mask = action_mask[:, 1:] # (num_samples * group_size, seq_len-1)
    
    # compute the rewards
    returns = torch.zeros(
        (num_samples, num_return_sequences), 
        dtype=torch.float,
        device=sequence_ids.device
    )
    sovled_rate = torch.zeros(
        (num_samples, ), 
        dtype=torch.float,
        device=sequence_ids.device
    )

    for i in range(0, num_rollouts, num_return_sequences):
        group_completions = completions[i:i+num_return_sequences]
        # compute the reward
        rewards, solved_rate = RewardModel.reward_batch(group_completions, oracle_answers)
        # TODO: map the process reward from the string space into the token space 
        returns[i] = torch.tensor(rewards, dtype=torch.float, device=sequence_ids.device)
        sovled_rate[i] = torch.tensor(solved_rate, dtype=torch.float, device=sequence_ids.device)

    returns = returns.reshape(-1) # (num_samples * num_return_sequences, )
    return sequence_ids, returns, solved_rate, action_mask, completions


def group_advantages(returns: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Normalizes the advantages of a group of returns.

    Args:
        returns (torch.Tensor): The returns to normalize.
        eps (float): The epsilon value to add to the standard deviation to prevent division by zero.

    Returns:
        torch.Tensor: The normalized advantages.
    """
    return (returns - returns.mean()) / (returns.std() + eps)


def sequence_log_probs_from_logits(logits: torch.Tensor, output_ids: torch.Tensor) -> torch.Tensor:
    """
    Computes the log probabilities of the output ids from the logits.

    Args:
        logits (torch.Tensor): The logits of the model. (num_samples * num_rollouts, seq_len, vocab_size)
        output_ids (torch.Tensor): The output ids to compute the log probabilities for. (num_samples * num_rollouts, seq_len)

    Returns:
        torch.Tensor: The log probabilities of the output ids. (num_samples * num_rollouts, seq_len)
    """
    log_prob = F.log_softmax(logits, dim=-1) # (num_samples * num_rollouts, seq_len, vocab_size)
    return log_prob.gather(
        dim=-1, 
        index=output_ids.unsqueeze(-1)
    ).squeeze(-1)


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
    position_ids = attention_mask.long().cumsum(dim=-1) - 1
    position_ids.masked_fill_(mask=(attention_mask == 0), value=1)
    output = model(
        input_ids=sequence_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_cache=False,
    )
    logits = output["logits"]

    # logits: [batch_size * num_rollouts, seq_len, vocab_size]
    log_probs = sequence_log_probs_from_logits(
        logits=logits[:, :-1],
        output_ids=sequence_ids[:, 1:],#right shift 1 block to get the actual output ids
    )
    return log_probs
