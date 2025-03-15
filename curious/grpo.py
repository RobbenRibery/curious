from collections.abc import Callable
import json
from pathlib import Path
import random
import re
from typing import Any, Iterator, Optional, List
import wandb
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedTokenizer,
    GenerationConfig,
    PreTrainedModel, 
    PreTrainedTokenizer, 
)
from curious.utils import tokenize_questions, load_model_tokenizer
from curious.loss import GRPOLoss, approx_kl_divergence
from curious.buffer import ReplayBuffer, Experience, join_experience_batch
from curious.reward import RewardModel
from dataclasses import dataclass
from lightning.pytorch import seed_everything

@torch.no_grad()
def rollout(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    batch_inputs: dict[str, torch.Tensor],
    oracle_answers: List[str],
    generation_config: GenerationConfig,
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
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
    action_mask = action_mask[:, 1:] 
    
    # compute the rewards
    returns = torch.zeros(
        (num_samples, num_return_sequences), 
        dtype=torch.float,
        device=sequence_ids.device
    )
    sovled_rate = torch.zeros(
        (num_samples, 1), 
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
        logits (torch.Tensor): The logits of the model. (num_samples, seq_len, vocab_size)
        output_ids (torch.Tensor): The output ids to compute the log probabilities for. (num_samples, seq_len)

    Returns:
        torch.Tensor: The log probabilities of the output ids.(num_samples, seq_len)
    """
    log_prob = F.log_softmax(logits, dim=-1) # (num_samples, seq_len, vocab_size)
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
        attention_mask (torch.Tensor): The attention mask to compute the log probabilities for.

    Returns:
        torch.Tensor: The log probabilities of the output ids.
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


def main():
    seed = 42
    wandb_project = None  # "tiny_grpo"
    device_index = 0
    model_name = "deepseek/math-instruct-v1.0"
    checkpoint_path = Path("./output")
    checkpoint_interval = 20
    train_batch_size = 16
    lr = 5e-6
    kl_weight = 0.01
    clip_eps = 0.2

    group_size = 12
    rollouts_per_step = 32
    epochs_per_step = 1
    max_norm = 1.0  # gradient clipping

    # rollout params
    max_length = 1024
    top_p = 1.0
    temperature = 1.0

    device = torch.device("cuda", device_index)
    cpu_device = torch.device("cpu")
    
    seed_everything(seed)

    reference_model, _ = load_model(model_name, device_map=device)
    model, tokenizer = load_model(model_name, device_map=device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    reference_model.eval()
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    pad_token_id = tokenizer.eos_token_id

    prompts = read_prompts(
        "data/math_tasks.jsonl",
        predicate=lambda x: len(x["question"]) < 128
        and x["num_terms"] <= 3
        and x["num_digits"] <= 3,
        max_rows=64 * 1024,
    )
    print(f"found {len(prompts)} matching prompts")
    prompt_loader = DataLoader(
        prompts,
        batch_size=rollouts_per_step,
        shuffle=True,
        drop_last=True,
        pin_memory=False,
    )

    replay_buffer = ReplayBuffer()
    objective = GRPOLoss(clip_eps=clip_eps, kl_weight=kl_weight)

    if wandb_project is None:
        wandb.init(mode="disabled")
    else:
        wandb.init(project=wandb_project)

    for k, prompt_batch in enumerate(prompt_loader):
        rollout_returns = []

        replay_buffer.clear()

        questions = prompt_batch["question"]
        answers = prompt_batch["answer"]

        with torch.no_grad():
            for q, a in zip(questions, answers):
                sequence_ids, returns, solved_rate, action_mask, completions = rollout(
                    model,
                    tokenizer,
                    q,
                    a,
                    num_rollouts=group_size,
                    max_length=max_length,
                    temperature=temperature,
                    top_p=top_p,
                )

                print(
                    f"rollout q='{q}', a='{a}', returns={returns.sum().item():.2f}, \
                    solved_rate={solved_rate.item():.2f}, \
                    replay_buffer_size={len(replay_buffer)}, \
                    sequence_ids={sequence_ids.shape}"
                )
                rollout_returns.append(returns.cpu())

                advantages = group_advantages(returns)
                attention_mask = sequence_ids != pad_token_id

                log_probs = sequences_log_probs(
                    model=model,
                    sequence_ids=sequence_ids,
                    attention_mask=attention_mask,
                )
                log_probs_ref = sequences_log_probs(
                    model=reference_model,
                    sequence_ids=sequence_ids,
                    attention_mask=attention_mask,
                )
                kl = approx_kl_divergence(
                    log_probs=log_probs,
                    log_probs_ref=log_probs_ref,
                    action_mask=action_mask,
                )

                experience = Experience(
                    sequences=sequence_ids,
                    action_log_probs=log_probs,
                    log_probs_ref=log_probs_ref,
                    returns=returns,
                    advantages=advantages,
                    attention_mask=attention_mask,
                    action_mask=action_mask,
                    kl=kl,
                )
                replay_buffer.append(experience.to(cpu_device))

        torch.cuda.empty_cache()
        episode_return_sum = torch.stack(rollout_returns).sum()
        print(f"returns of step {k}: {episode_return_sum:.4f}")
        wandb.log({"returns": episode_return_sum})

        experience_sampler = DataLoader(
            replay_buffer,
            batch_size=train_batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=join_experience_batch,
        )

        for step_epoch in range(epochs_per_step):
            model.train()

            for exp in experience_sampler:
                exp: Experience

                exp = exp.to(device)

                optimizer.zero_grad()

                log_probs = sequences_log_probs(
                    model, sequence_ids=exp.sequences, attention_mask=exp.attention_mask
                )

                loss, kl = objective(log_probs=log_probs, experience=exp)

                if not loss.isfinite():
                    print(f"Loss not finite, skipping backward, loss={loss}")
                    print(f"experience.advantages={experience.advantages}")
                    continue

                loss.backward()
                grad_norm = clip_grad_norm_(model.parameters(), max_norm=max_norm)
                print(f"{step_epoch}: kl={kl: .4f}, grad_norm={grad_norm: .4f}")
                wandb.log({"kl": kl, "grad_norm": grad_norm})

                optimizer.step()

        if (
            checkpoint_path is not None
            and checkpoint_interval is not None
            and (k + 1) % checkpoint_interval == 0
        ):
            model.save_pretrained(checkpoint_path / f"step_{k}")

    if checkpoint_path is not None:
        model.save_pretrained(checkpoint_path / f"step_{k}")


if __name__ == "__main__":
    main()