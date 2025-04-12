from dataclasses import dataclass
from typing import List, Callable, Dict, Any
import os
import tyro
import wandb
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer, PreTrainedModel

from curious.data import GSM8KDataset
from curious.utils import LOGGING_TEMPLATE, load_model_tokenizer
from curious.utils.rl.sampling import compute_rewards
from curious.reward import GSM8KRewardModel
from curious.utils.llm.prompt import *
from config import WandbConfig, BaseConfig, RewardConfig

from accelerate.utils import set_seed
import gc 
from textwrap import dedent

@dataclass
class FixedSamplingConfig:
    """
    A dataclass for storing the fixed sampling configuration.
    """
    
    max_new_tokens: int = 1024
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
    
    repetition_penalty: float = 1.1
    """
    The repetition penalty to use for the sampling.
    """
    
    use_cache: bool = True
    """
    Whether to use the cache during the sampling.
    """
    
    system_prompt: str = "qwen_system_prompt"
    """
    The system prompt to use for the sampling.
    """

    model_prompt_length: int = 1024
    """
    The maximum length of the model to use for the evaluation.
    """

@dataclass
class EvaluationConfig:
    """
    A dataclass for storing the evaluation configuration.
    """

    wandb_config: WandbConfig
    """
    The wandb configuration.
    """

    base_config: BaseConfig
    """
    The base configuration.
    """

    sampling_config: FixedSamplingConfig
    """
    The sampling configuration.
    """

    reward_config: RewardConfig
    """
    The reward configuration.
    """


@torch.no_grad()
def evaluate(
    config: EvaluationConfig,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    logger: Callable,
    **kwargs,
) -> Dict[str, Any]:
    """
    Evaluate the model on the GSM8K dataset.
    Args:
        config (EvaluationConfig): The evaluation configuration.
        model (PreTrainedModel): The model to evaluate.
        tokenizer (PreTrainedTokenizer): The tokenizer to evaluate.
        logger (Callable): The logger to use.
        **kwargs: Additional arguments.
    Returns:
        A dictionary containing the rewards, infos and solved rates.
    """
    set_seed(config.base_config.seed) 
    batch_idx = kwargs.get("batch_idx", 0)
    print(f"Batch {batch_idx} #### Evaluating...")

    out_dir = os.path.join(
        config.base_config.eval_log_dir,
        os.path.basename(config.base_config.model_name.replace("-", "_")),
        os.path.basename(config.base_config.dataset_name.replace("-", "_")),
        config.wandb_config.name.replace("-", "_"),
    )
    os.makedirs(out_dir, exist_ok=True)

    # Check if the dataset is GSM8K
    if config.base_config.dataset_name != "openai/gsm8k":
        raise NotImplementedError("Only GSM8K is supported for now")

    # Load the dataset
    dataset = GSM8KDataset(
        tokenizer=tokenizer,
        dataset_name=config.base_config.dataset_name,
        seed=config.base_config.seed,
        mode="test",
        max_prompt_length=config.sampling_config.model_prompt_length,
        system_prompt=eval(config.sampling_config.system_prompt),
    )

    # Create a dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=config.base_config.eval_batch_size,
        num_workers=config.base_config.num_workers,
        shuffle=False,
        drop_last=False,
    )

    # Initialize the reward model
    reward_model = GSM8KRewardModel(
        answer_pattern=config.reward_config.answer_pattern,
        think_pattern=config.reward_config.think_pattern,
        use_format_reward=config.reward_config.use_format_reward,
    )

    rewards: List[float] = []
    infos: List[Dict[str, str]] = []
    solved_masks: List[int] = []
    num_words_in_completions: List[int] = []
    for _, batch_inputs in enumerate(dataloader):
        
        # Get the questions and answers
        questions = batch_inputs["question"]
        oracle_answers = batch_inputs["oracle_answer"]

        # Get the model predictions
        seq_ids = model.generate(
            input_ids = batch_inputs["input_ids"].to(model.device),
            attention_mask = batch_inputs["attention_mask"].to(model.device),
            max_new_tokens=config.sampling_config.max_new_tokens,
            temperature=config.sampling_config.temperature,
            top_p=config.sampling_config.top_p,
            top_k=config.sampling_config.top_k,
            do_sample=config.sampling_config.do_sample,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=config.sampling_config.repetition_penalty,
        )

        # Decode the generations
        completions: List[str] = tokenizer.batch_decode(
            seq_ids[:, batch_inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )

        # get the rewards
        rewards_out: Dict[str, List[Dict[str, str]] | torch.Tensor] = compute_rewards(
            reward_model,
            completions=completions,
            oracle_answers=oracle_answers,
            group_size=1,
        )

        batch_rewards: torch.Tensor = rewards_out["returns"]
        batch_infos: List[Dict[str, str]] = rewards_out["infos"]
        batch_solved_masks: torch.Tensor = rewards_out["solved_masks"]
        batch_num_words_in_completions: List[int] = [len(completion.split(' ')) for completion in completions]

        batch_mean_format_returns: float = np.array([x["format_reward"] for x in batch_infos]).mean()
        batch_mean_outcome_returns: float = np.array([x["outcome_reward"] for x in batch_infos]).mean()

        rewards.extend(batch_rewards.tolist())
        infos.extend(batch_infos)
        solved_masks.extend(batch_solved_masks.tolist())
        num_words_in_completions.extend(batch_num_words_in_completions)
        
        # --------------------- Logging Start ---------------------
        logger(
            {
                "eval/batch_rewards": batch_rewards.mean(),
                "eval/batch_solved_masks": batch_solved_masks.mean(),
                "eval/batch_mean_format_returns": batch_mean_format_returns,
                "eval/batch_mean_outcome_returns": batch_mean_outcome_returns,
                "eval/batch_mean_num_words_in_completions": np.array(batch_num_words_in_completions).mean(),
            }
        )

        ## Save the text on disk if the batch index is a multiple of the eval text log interval
        if batch_idx % config.base_config.eval_text_log_interval == 0:
            # Log the text to logger
            text_to_log = ""
            for question, answer, completion, reward, info in zip(
                questions,
                oracle_answers,
                completions,
                batch_rewards,
                infos,
            ):
                text_to_log += LOGGING_TEMPLATE.format(
                    question=question,
                    answer=answer,
                    completion=completion,
                    reward=reward,
                    info=info,
                )

            file_name = f"log_{batch_idx}.txt" if batch_idx > 0 else "final_log.txt"
            with open(os.path.join(out_dir, file_name), "a") as f:
                f.write(text_to_log)
            f.close()
        # --------------------- Logging End ---------------------
        
        # free up memory
        del (
            batch_inputs,
            seq_ids,
            completions,
            batch_rewards,
            batch_infos,
            batch_solved_masks,
        )
        gc.collect()
        torch.cuda.empty_cache()

    # Log the mean rewards and the mean solved rate
    mean_pass1 = np.array(solved_masks).mean()
    mean_rewards = np.array(rewards).mean()
    mean_num_words_in_completions = np.array(num_words_in_completions).mean()
    logger(
        {
            "eval/mean_rewards": mean_rewards,
            "eval/mean_solved_rate": mean_pass1,
            "eval/mean_num_words_in_completions": mean_num_words_in_completions,
        }
    )
    print(f"Batch {batch_idx} #### Mean Eval pass@1: {mean_pass1}")

    return {
        "rewards": rewards,
        "infos": infos,
        "solved_masks": solved_masks,
        "mean_rewards": mean_rewards,
        "mean_solved_rate": mean_pass1,
        "mean_num_words_in_completions": mean_num_words_in_completions,
    }


if __name__ == "__main__":


    # Parse the command line arguments
    config = tyro.cli(EvaluationConfig)

    # Initialize the wandb
    wandb.init(
        project=config.wandb_config.project,
        name=config.wandb_config.name,
        config=config,
    )
    logger = wandb.log

    # Load the model
    model, tokenizer = load_model_tokenizer(
        config.base_config.model_name, 
        freeze_model=True,
        checkpoint_path=config.base_config.checkpoint_dir,
    )

    # Evaluate the model
    rewards, infos, solved_rates = evaluate(
        config,
        model,
        tokenizer,
        logger
    )