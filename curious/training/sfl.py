from typing import Tuple, List, Dict, Any, Callable

from transformers import PreTrainedModel, PreTrainedTokenizer, GenerationConfig
from torch.utils.data import DataLoader

from curious.reward.rule.gsm8k import GSM8KRewardModel
from curious.data.gsm8k import GSM8KDataset
from curious.config import TrainingConfig
from curious.training.normal import train as train_normal
from curious.utils.utils import load_model_tokenizer, form_hf_dataset
from curious.sampling.sfl import sfl_sampling
from curious.replay.curriculum import Curriculum
from curious.evaluate import EvaluationConfig, FixedSamplingConfig

import torch
from accelerate.utils import set_seed
import wandb
import numpy as np

from tqdm import tqdm
from rich import print
from dataclasses import dataclass
import tyro
import gc

def train_sfl(
    args:TrainingConfig, 
    logger:Callable
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
       Train the model using SFL.
    """

    # outputs 
    train_outs:List[Dict[str, Any]] = []
    eval_outs:List[Dict[str, Any]] = []

    # device & seeding
    device = torch.device("cuda", args.base_config.device_index)
    cpu_device = torch.device("cpu")
    seed = args.base_config.seed
    set_seed(seed)

    # rng 
    rng = np.random.default_rng(seed)

    # get the run name
    run_name = args.wandb_config.name.replace("-", "_")

    ## dataset
    dataset = GSM8KDataset(
        tokenizer=tokenizer,
        dataset_name=args.base_config.dataset_name,
        seed=args.base_config.seed,
        mode=args.base_config.mode,
        max_prompt_length=args.sampling_config.model_prompt_length,
        system_prompt=eval(args.sampling_config.system_prompt),
    )
    train_max_input_length = dataset.train_max_length

    ## load target policy
    model, tokenizer = load_model_tokenizer(
        args.base_config.model_name, 
        device_map=device
    )
    model.gradient_checkpointing_enable()

    ## Reward model
    reward_model = GSM8KRewardModel(
        answer_pattern=args.reward_config.answer_pattern,
        think_pattern=args.reward_config.think_pattern,
        use_format_reward=args.reward_config.use_format_reward,
        use_overlong_penalty=args.reward_config.use_overlong_penalty,
        l_max=args.reward_config.l_max,
        l_cache=args.reward_config.l_cache,
    )

    # sampling config
    generation_config = GenerationConfig(
        num_return_sequences=args.grpo_config.group_size,
        max_new_tokens=args.sampling_config.max_new_tokens,
        temperature=args.sampling_config.temperature,
        top_p=args.sampling_config.top_p,
        top_k=args.sampling_config.top_k,
    )

    ## evaluation config
    eval_config = EvaluationConfig(
        base_config=args.base_config,
        sampling_config=FixedSamplingConfig(),
        reward_config=args.reward_config,
        wandb_config=args.wandb_config,
    )

    ## total number of steps scheudled 
    step = 0
    while step < args.base_config.total_steps:

        ## shuffle the dataset
        seed = rng.integers(0, 1e03, size=1)[0]
        dataset.train = dataset.train.shuffle(seed=seed)
        rng = np.random.default_rng(seed)

        ## create the data loader
        sfl_sampling_data_loader = DataLoader(
            dataset.train,
            batch_size=args.base_config.train_batch_size,
            shuffle=False,
            num_workers=args.base_config.num_workers,
        )

        ## sfl sampling step 
        sampled_curriculum:List[Curriculum] = sfl_sampling(
            model,
            tokenizer,
            reward_model,
            sfl_sampling_data_loader,
            generation_config,
            seed = seed,
            sfl_total_scaning_size = args.sfl_config.sfl_total_scaning_size,
            sfl_num_samples_to_collect = args.sfl_config.sfl_num_samples_to_collect,
            cpu_device = cpu_device,
        )

        ## create the training dataset & data set and shuffle 
        hf_dataset = form_hf_dataset(
            tokenizer,
            sampled_curriculum,
            seed = seed,
            max_prompt_length = train_max_input_length,
            system_prompt = eval(args.sampling_config.system_prompt),
        )
        ## sfl training step 
        model, train_outs, eval_outs = train_normal(
            args,
            logger,
            trained_model=model,
            tokenizer=tokenizer,
            reward_model=reward_model,
            generation_config=generation_config,
            eval_config=eval_config,
            dataset=hf_dataset,
        )

        ## increment the step
        step += 1

        ## logging
        logger(f"Step {step} completed")
        
    return train_outs, eval_outs

    
    

if __name__ == "__main__":
    
    args = tyro.cli(train_sfl)
    train_sfl(args, print)

