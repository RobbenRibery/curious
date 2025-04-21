from typing import Dict, List

import torch
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    GenerationConfig,
)

from torch.utils.data import DataLoader
from curious.sampling.sampling import (
    sample_responses, 
    compute_rewards, 
    compute_learnability,
    compute_group_advantages,
)
from curious.replay.buffer import Experience, ReplayBuffer
from curious.reward.rule.reward import GSM8KRewardModel

def sfl_sampling(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    reward_model: GSM8KRewardModel,
    data_loader: DataLoader,
    generation_config: GenerationConfig,
    seed: int = 42,
    adv_normalize_centered_returns: bool = False,
    adv_use_rloo_scalar: bool = False,
    sfl_num_samples_to_collect: int = 100,
    sfl_use_top_k: bool = True,
    cpu_device: torch.device = torch.device("cpu"),
) -> ReplayBuffer:
    """
    Sample experiences from the dataset using SFL.
    """
    # get the pad token id
    pad_token_id = tokenizer.eos_token_id

    # initialize the replay buffer
    replay_buffer = ReplayBuffer()
    
    # iterate through the dataset
    for encoded_batch in data_loader:
        # sample responses
        sampling_output = sample_responses(
            model,
            tokenizer,
            encoded_batch,
            generation_config,
            seed,
        )

        # get the sequence ids
        sequence_ids: torch.Tensor = sampling_output["sequence_ids"]
        attention_mask: torch.Tensor = sequence_ids != pad_token_id
        action_mask: torch.Tensor = sampling_output["action_mask"]

        # get the completions
        completions:List[str] = sampling_output["completions"]

        # compute the rewards
        ## compute on cpu device
        ## output on cuda device
        reward_output = compute_rewards(
            model,
            reward_model,
            sampling_output["completions"],
            encoded_batch["oracle_answers"],
            generation_config.num_return_sequences,
        )

        # compute the advantages of the group
        ## compute on cuda device
        ## output on cuda device
        advantages = compute_group_advantages(
            reward_output["returns"],
            normalize=adv_normalize_centered_returns,
            use_rloo_scalar=adv_use_rloo_scalar,
        )

        # compute the learnability of the group
        ## compute on cuda device
        ## output on cuda device
        learnability = compute_learnability(reward_output["solved_masks"])

        # create the experience
        experience: Experience = Experience(
            sequences=sequence_ids,
            attention_mask=attention_mask,
            action_mask=action_mask,
            returns=reward_output["returns"],
            solved_mask=reward_output["solved_masks"],
            advantages=advantages,
            learnability=learnability,
            completion=completions,
        )

        # append the experience to the replay buffer
        replay_buffer.append(experience.to(cpu_device))

    # sample the experiences
    sampled_replay_buffer = replay_buffer.sample_for_learnability(
        num_samples=sfl_num_samples_to_collect,
        use_top_k=sfl_use_top_k,
    )

    # return the sampled replay buffer
    replay_buffer.clear()

    return sampled_replay_buffer


