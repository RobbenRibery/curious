from dataclasses import dataclass
from typing import List, Callable, Dict
import os
import tyro
import wandb
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import GenerationConfig, PreTrainedTokenizer, PreTrainedModel
from liger_kernel.transformers import AutoLigerKernelForCausalLM

from curious.data import GSM8KDataset
from curious.utils import load_model_tokenizer, tokenize_questions
from curious.reward import GSM8KRewardModel
from config import WandbConfig, BaseConfig, SamplingConfig, RewardConfig

from lightning import seed_everything
from uuid import uuid4


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

    sampling_config: SamplingConfig
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
):
    """
    Evaluate the model on the GSM8K dataset.
    Args:
        config (EvaluationConfig): The evaluation configuration.
        model (PreTrainedModel): The model to evaluate.
        tokenizer (PreTrainedTokenizer): The tokenizer to evaluate.
        logger (Callable): The logger to use.
        **kwargs: Additional arguments.

    Returns:
        Tuple[List[float], List[Dict[str, str]], List[float]]: A tuple containing the rewards, infos and solved rates.
    """
    seed_everything(config.base_config.seed) 
    text_logger = kwargs.get("text_logger", lambda x: print(x))

    out_dir = os.path.join(
        config.base_config.log_dir,
        os.path.basename(config.base_config.model_name.replace("-", "_")),
        os.path.basename(config.base_config.dataset_name.replace("-", "_")),
        config.wandb_config.name.replace("-", "_"),
    )
    os.makedirs(out_dir, exist_ok=True)

    # Check if the dataset is GSM8K
    if config.base_config.dataset_name != "openai/gsm8k":
        raise NotImplementedError("Only GSM8K is supported for now")

    # Set the mode to test if it is train
    if config.base_config.mode == "train":
        config.base_config.mode = "test"

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
        shuffle=False,
    )

    # Initialize the reward model
    reward_model = GSM8KRewardModel(
        answer_pattern=config.reward_config.answer_pattern,
        think_pattern=config.reward_config.think_pattern,
        use_format_reward=config.reward_config.use_format_reward,
    )

    rewards: List[float] = []
    infos: List[Dict[str, str]] = []
    solved_rates: List[float] = []
    for batch_idx, batch in enumerate(dataloader):
        # Get the questions and answers
        questions = batch["question"]
        oracle_answers = batch["oracle_answer"]

        # Tokenize the questions
        batch_inputs = tokenize_questions(tokenizer, questions)
        batch_inputs = {
            key: value.to(model.device) for key, value in batch_inputs.items()
        }

        # Get the model predictions
        seq_ids = model.generate(
            input_ids = batch_inputs["input_ids"],
            attention_mask = batch_inputs["attention_mask"],
            max_new_tokens=config.sampling_config.max_new_tokens,
            temperature=config.sampling_config.temperature,
            top_p=config.sampling_config.top_p,
            top_k=config.sampling_config.top_k,
            do_sample=config.sampling_config.do_sample,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=config.sampling_config.use_cache, 
            repetition_penalty=config.sampling_config.repetition_penalty,
        )

        # Decode the generations
        completions: List[str] = tokenizer.batch_decode(
            seq_ids[:, batch_inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )

        # Compute the rewards and the solved rate
        batch_rewards, batch_infos, batch_solved_rate = reward_model(
            completions,
            oracle_answers,
        )

        rewards.extend(batch_rewards)
        infos.extend(batch_infos)
        solved_rates.append(batch_solved_rate)

        batch_mean_format_returns = np.array([x["format_reward"] for x in batch_infos]).mean()
        batch_mean_outcome_returns = np.array([x["outcome_reward"] for x in batch_infos]).mean()

        # Log the rewards and the solved rate
        logger(
            {
                "eval/batch_rewards": np.array(batch_rewards).mean(),
                "eval/batch_solved_rate": batch_solved_rate,
                "eval/batch_mean_format_returns": batch_mean_format_returns,
                "eval/batch_mean_outcome_returns": batch_mean_outcome_returns,
            }
        )
        text_logger(
            {
                "question": questions,
                "answer": oracle_answers,
                "completion": completions,
                "reward": batch_rewards,
            }
        )

        for question, answer, completion, reward, info in zip(
            questions,
            oracle_answers,
            completions,
            batch_rewards,
            infos,
        ):
            with open(os.path.join(out_dir, "log.txt"), "a") as f:
                f.write(
                    f"******\nQuestion: {question}\nAnswer: {answer}\n"
                    f"Completion: {completion}\nReward: {reward}\n"
                    f"Info: {info}\n******\n"
                )
            f.close()

        del (
            batch_inputs,
            seq_ids,
            completions,
            batch_rewards,
            batch_infos,
            batch_solved_rate,
        )
        torch.cuda.empty_cache()

    # Log the mean rewards and the mean solved rate
    mean_pass1 = np.array(solved_rates).mean()
    logger(
        {
            "eval/mean_rewards": np.array(rewards).mean(),
            "eval/mean_solved_rate": mean_pass1,
        }
    )
    print(f"#### Mean pass@1: {mean_pass1}")

    # Log the text table
    logger({"eval/text_table": text_table})

    return rewards, infos, solved_rates


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

    # text table logger
    text_table = wandb.Table(columns=["question", "answer", "completion", "reward"])
    text_logger = lambda x: text_table.add_data(
        x["question"],
        x["answer"],
        x["completion"],
        x["reward"],
    )

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
        logger,
        **{
            "text_logger": text_logger,
        },
    )