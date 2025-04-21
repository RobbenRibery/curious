from typing import Dict, List, Self, Tuple

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
from curious.reward.rule.gsm8k import GSM8KRewardModel
from dataclasses import dataclass, fields

from torch.utils.data import Dataset
from tqdm import tqdm

@dataclass
class Curriculum:
    """
    A dataclass for storing the curriculum.
    """
    question: List[str]
    """The quesitons provided"""

    oracle_answer: List[str]
    """The oracle answers"""
    
    learnability: torch.Tensor
    """The learnability scores"""

    def to(self, device: torch.device) -> Self:
        members = {}
        for field in fields(self):
            v = getattr(self, field.name)
            if isinstance(v, torch.Tensor):
                v = v.to(device=device)
            members[field.name] = v
        return Curriculum(**members)
    

class CurriculumBuffer:
    """
    A buffer for storing the curriculum.
    """ 
    def __init__(self, buffer: List[Curriculum] = None) -> None:
        self.buffer = buffer if buffer is not None else []
        self.keys = [
            "question",
            "oracle_answer",
            "learnability",
        ]

    def append(self, curriculum: Curriculum) -> None:
        """
        Append a curriculum to the buffer.
        """
        num_samples = len(curriculum.question)
        individual_curriculums:List[Curriculum] = [{} for _ in range(num_samples)]

        for key in self.keys:
            print(key)
            if key not in curriculum.__dict__:
                raise ValueError(f"Curriculum must have a {key} attribute")
            
            values = getattr(curriculum, key)
            if isinstance(values, torch.Tensor):
                values: Tuple[torch.Tensor, ...] = torch.unbind(values)
            
            for i, v in enumerate(values):
                individual_curriculums[i][key] = v

        self.buffer.extend([Curriculum(**data) for data in individual_curriculums])

    def clear(self) -> None:
        self.buffer.clear()

    def __len__(self) -> int:
        return len(self.buffer) 
    
    def __getitem__(self, idx: int) -> Curriculum:
        return self.buffer[idx]
    
    def to_dataset(self) -> Dataset:
        pass 
    

def sfl_sampling(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    reward_model: GSM8KRewardModel,
    data_loader: DataLoader,
    generation_config: GenerationConfig,
    seed: int = 42,
    sfl_total_scaning_size: int = 512,
    sfl_num_samples_to_collect: int = 100,
    cpu_device: torch.device = torch.device("cpu"),
) -> ReplayBuffer:
    """
    Sample experiences from the dataset using SFL.
    """
    # learnability scores
    learnability_scores = torch.zeros(
        (sfl_total_scaning_size,),
        dtype=torch.bfloat16, 
        device=model.device,
    )
    curriculum_buffer:CurriculumBuffer = CurriculumBuffer()

    # iterate through the dataset
    scanned_samples = 0
    for encoded_batch in data_loader:
        # check if we have scanned enough samples
        if scanned_samples >= sfl_total_scaning_size:
            break
        
        # sample responses
        sampling_output = sample_responses(
            model,
            tokenizer,
            encoded_batch,
            generation_config,
            seed,
        )
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

        tmp_curriculum = Curriculum(
            question=questions,
            oracle_answer=oracle_answers,
            learnability=learnability,
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
    #top_k_values = top_k_output.values

    return [
        curriculum_buffer[i] for i in top_k_indices.tolist()
    ]


