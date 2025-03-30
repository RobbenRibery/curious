from dataclasses import dataclass
from typing import List
import tyro 
import wandb 

import torch
from torch.utils.data import DataLoader
from transformers import GenerationConfig

from curious.data import GSM8KDataset
from curious.utils import load_model_tokenizer, tokenize_questions
from curious.reward import GSM8KRewardModel, THINK_PATTERN, ANSWER_PATTERN

from uuid import uuid4

@dataclass
class WandbConfig:
    """
    A dataclass for storing the wandb configuration.
    """
    
    project: str = "curious"
    """
    The project to use for the wandb.
    """
    
    name: str = f"curious-{uuid4()}"
    """
    The name to use for the wandb.
    """

@dataclass
class BaseConfig:
    """
    A dataclass for storing the evaluation configuration.
    """
    # Model and dataset
    model_name: str = "Qwen/Qwen2.5-Coder-32B-Instruct"
    """
    The name of the model to use for the evaluation.
    """
    dataset_name: str = "openai/gsm8k"
    """
    The name of the dataset to use for the evaluation.
    """
    batch_size: int = 32
    """
    The batch size to use for the evaluation.
    """
    num_workers: int = 1
    """
    The number of workers to use for the evaluation.
    """
    seed: int = 42
    """
    The seed to use for the evaluation.
    """
    mode: str = "test"
    """
    The mode to use for the evaluation.
    """
    log_dir: str = "logs"
    """
    The directory to use for the evaluation.
    """

@dataclass
class SamplingConfig:
    """
    A dataclass for storing the sampling configuration.
    """
    num_samples: int = 1
    """
    The number of samples to use for the evaluation.
    """
    max_new_tokens: int = 100
    """
    The maximum number of new tokens to use for the evaluation.
    """
    temperature: float = 0.7
    """
    The temperature to use for the evaluation.
    """
    top_p: float = 0.9
    """
    The top p to use for the evaluation.
    """
    top_k: int = 50
    """
    The top k to use for the evaluation.
    """
    do_sample: bool = True
    """
    Whether to sample from the model.
    """

@dataclass
class RewardConfig:
    """
    A dataclass for storing the reward configuration.
    """
    answer_pattern: str = ANSWER_PATTERN
    """
    The pattern to use for the answer.
    """
    think_pattern: str = THINK_PATTERN
    """
    The pattern to use for the think.
    """
    use_format_reward: bool = False
    """
    Whether to use the format reward.
    """

@dataclass
class EvaluationConfig:
    """
    A dataclass for storing the evaluation configuration.
    """
    wandb_config: WandbConfig

    base_config: BaseConfig
    
    sampling_config: SamplingConfig
    
    reward_config: RewardConfig


@torch.no_grad()
def evaluate(config: EvaluationConfig):
    """
    Evaluate the model on the dataset.

    Args:
        config (EvaluationConfig): The evaluation configuration.

    Returns:
        None
    """
    if config.base_config.dataset_name != "openai/gsm8k":
        raise NotImplementedError("Only GSM8K is supported for now")    

    # Set the mode to test if it is train
    if config.base_config.mode == "train":
        config.base_config.mode = "test"
    
    # Load the model 
    model, tokenizer = load_model_tokenizer(config.base_config.model_name)

    # Load the dataset
    dataset = GSM8KDataset(
        dataset_name=config.base_config.dataset_name,
        seed=config.base_config.seed,
        mode=config.base_config.mode,
    )

    # Create a dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=config.base_config.batch_size,
        num_workers=config.base_config.num_workers,
    )

    # Initialize the reward model
    reward_model = GSM8KRewardModel(
        answer_pattern=config.reward_config.answer_pattern,
        think_pattern=config.reward_config.think_pattern,
        use_format_reward=config.reward_config.use_format_reward,
    )
    # sampling config 
    sampling_config = GenerationConfig(
        max_new_tokens=config.sampling_config.max_new_tokens,
        temperature=config.sampling_config.temperature,
        top_p=config.sampling_config.top_p,
        top_k=config.sampling_config.top_k,
        do_sample=config.sampling_config.do_sample,
    )

    for batch in dataloader:
        # Get the questions and answers
        questions = batch["question"]
        oracle_answers = batch["oracle_answer"]

        # Tokenize the questions
        batch_inputs = tokenize_questions(tokenizer, questions)

        # Get the model predictions
        seq_ids = model.generate(
            **batch_inputs,
            generation_config=sampling_config,
        )

        # Decode the generations 
        completions:List[str] = tokenizer.batch_decode(
            seq_ids[:, batch_inputs["input_ids"].shape[1] :], 
            skip_special_tokens=True
        )

        # Compute the rewards and the solved rate 
        candidate_answers, rewards, solved_times = reward_model(
            completions,
            oracle_answers,
        )

        # Log the rewards and the solved rate
        wandb.log({
            "rewards": rewards,
            "solved_times": solved_times,
        })
        

if __name__ == "__main__":

    # Parse the command line arguments
    config = tyro.cli(EvaluationConfig)
    
    # Initialize the wandb
    wandb.init(
        project=config.wandb_config.project,
        name=config.wandb_config.name,
        config=config,
    )
    
    # Evaluate the model
    evaluate(config)