from typing import List

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
)
from curious.replay.curriculum import Curriculum, CurriculumBuffer
from curious.reward.rule.gsm8k import GSM8KRewardModel
import gc 

def sfl_sampling(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    reward_model: GSM8KRewardModel,
    data_loader: DataLoader,
    generation_config: GenerationConfig,
    seed: int = 42,
    sfl_total_scanning_size: int = 512,
    sfl_num_samples_to_collect: int = 100,
    cpu_device: torch.device = torch.device("cpu"),
) -> List[Curriculum]:
    """
    Sample experiences from the dataset using SFL.
    """
    # learnability scores
    learnability_scores = torch.zeros(
        (sfl_total_scanning_size,),
        dtype=torch.bfloat16, 
        device=model.device,
    )
    curriculum_buffer = CurriculumBuffer()

    # iterate through the dataset
    scanned_samples = 0
    for encoded_batch in data_loader:
        # check if we have scanned enough samples
        if scanned_samples >= sfl_total_scanning_size:
            break
        
        # sample responses
        sampling_output = sample_responses(
            model,
            tokenizer,
            encoded_batch,
            generation_config,
            seed,
        )
        completions:List[str] = sampling_output["completions"]
        group_completions:List[str] = []
        for i in range(0, len(completions), generation_config.num_return_sequences):
            group_completions_string = "--\n--".join(completions[i:i+generation_config.num_return_sequences])
            group_completions.append(group_completions_string)

        # compute the rewards
        ## compute on cpu device
        ## output on cuda device
        solved_masks = compute_rewards(
            model,
            reward_model,
            sampling_output["completions"],
            encoded_batch["oracle_answer"],
            generation_config.num_return_sequences,
        )["solved_masks"]

        # compute the learnability of the group
        ## compute on cuda device
        ## output on cuda device
        learnability = compute_learnability(solved_masks)

        # update the curriculum buffer
        questions = encoded_batch["question"]
        oracle_answers = encoded_batch["oracle_answer"]

        # create the curriculum
        tmp_curriculum = Curriculum(
            question=questions,
            oracle_answer=oracle_answers,
            learnability=learnability,
            completion=group_completions,
        )
        curriculum_buffer.append(tmp_curriculum.to(cpu_device))

        # get the starting and ending question indices
        starting_question_index = scanned_samples
        ending_question_index = scanned_samples + learnability.shape[0]

        # assign the learnability scores
        learnability_scores[starting_question_index:ending_question_index] = learnability

        # increment the scanned samples
        scanned_samples += encoded_batch["input_ids"].shape[0]

    # sample the experiences
    top_k_output = torch.topk(learnability_scores, k=sfl_num_samples_to_collect)
    top_k_indices = top_k_output.indices

    del solved_masks
    del learnability
    del learnability_scores
    del top_k_output
    gc.collect()
    torch.cuda.empty_cache()

    return [
        curriculum_buffer[i] for i in top_k_indices.tolist()
    ]


