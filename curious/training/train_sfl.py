from typing import List, Callable, Tuple, Dict, Any
from torch.utils.data import DataLoader
from transformers import PreTrainedModel

from curious.config import TrainingConfig
from curious.training.training_setup import TrainingSetup, set_up_training
from curious.training.train_rl import train
from curious.utils.utils import form_hf_dataset
from curious.sampling.sfl import sfl_sampling
from curious.prompt import *
from curious.replay.curriculum import Curriculum

from accelerate.utils import set_seed
import wandb
import numpy as np
import tyro
from rich import print

def train_sfl(
    args:TrainingConfig, 
    training_setup:TrainingSetup,
    logger:Callable
) -> Tuple[
    PreTrainedModel,
    List[Dict[str, Any]],
    List[Dict[str, Any]]
]:
    """
    Train the model using SFL.
    Args:
        args (TrainingConfig): The training configuration.
        training_setup (TrainingSetup): The training setup.
        logger (Callable): The logger to use.
    Returns:
        model (PreTrainedModel): The model to train.
        train_outs (List[Dict[str, Any]]): The training outputs.
        eval_outs (List[Dict[str, Any]]): The evaluation outputs.
    """
    # some sanity checks
    assert args.sfl_config.sfl_enabled, "SFL is not enabled, please set sfl_enabled to True"
    assert args.sfl_config.sfl_num_samples_to_collect <= args.sfl_config.sfl_total_scanning_size, \
        "sfl_num_samples_to_collect must be less than or equal to sfl_total_scanning_size"
    assert args.sfl_config.sfl_num_samples_to_collect % args.base_config.train_batch_size == 0, \
        "sfl_num_samples_to_collect must be divisible by train_batch_size"
    assert args.sfl_config.sfl_total_scanning_size % args.sfl_config.sfl_sampling_batch_size == 0, \
        "sfl_total_scanning_size must be divisible by sfl_sampling_batch_size"
    
    assert not args.grpo_config.anneling_lr, "Annealing learning rate is not supported for SFL"
    assert not args.grpo_config.anneling_temperature, "Annealing temperature is not supported for SFL"

    assert args.grpo_config.kl_weight == 0, "KL weight must be 0 for SFL"
    assert args.grpo_config.ref_model_update_freq == 0, "Reference model update frequency must be 0 for SFL"

    # set the seed for accelerate
    set_seed(args.base_config.seed)
    seed = args.base_config.seed
    rng = np.random.default_rng(seed)

    # unload the training setup 
    model = training_setup["target_policy"]
    dataset = training_setup["dataset"]
    train_max_input_length = dataset.train_max_length

    ## total number of steps scheduled 
    sfl_step, global_batch_idx = 0, 0
    while sfl_step < args.sfl_config.sfl_total_steps:

        ## shuffle the dataset
        seed = rng.integers(0, 1e03, size=1)[0].item()
        dataset.train = dataset.train.shuffle(seed=seed)
        rng = np.random.default_rng(seed)

        ## create the data loader
        sfl_sampling_data_loader = DataLoader(
            dataset.train,
            batch_size=args.sfl_config.sfl_sampling_batch_size,
            shuffle=False,
            num_workers=args.base_config.num_workers,
        )

        ## sfl sampling step 
        sampled_curriculum:List[Curriculum] = sfl_sampling(
            model = model,
            tokenizer = training_setup["tokenizer"],
            reward_model = training_setup["reward_model"],
            data_loader = sfl_sampling_data_loader,
            generation_config = training_setup["generation_config"],
            seed = seed,
            sfl_total_scanning_size = args.sfl_config.sfl_total_scanning_size,
            sfl_num_samples_to_collect = args.sfl_config.sfl_num_samples_to_collect,
            cpu_device = training_setup["cpu_device"],
        )
        assert len(sampled_curriculum) == args.sfl_config.sfl_num_samples_to_collect, \
            "The number of sampled curriculum must be equal to sfl_num_samples_to_collect"
        
        ### logging the learnability scores
        learnability_scores = np.array(
            [
                x.learnability.to(training_setup["cpu_device"]).item() \
                for x in sampled_curriculum
            ]
        )
        logger(
            {
                "sfl/mean_learnability": learnability_scores.mean(),
                "sfl/std_learnability": learnability_scores.std(),
                "sfl/min_learnability": learnability_scores.min(),
                "sfl/max_learnability": learnability_scores.max(),
                "sfl_step": sfl_step,
            }
        )
        print(f"sfl_step: {sfl_step} | mean_learnability: {learnability_scores.mean()}")
        

        ## create the training dataset & data set and shuffle using the seed
        hf_dataset = form_hf_dataset(
            tokenizer = training_setup["tokenizer"],
            data = [c.to_dict() for c in sampled_curriculum],
            seed = seed,
            max_prompt_length = train_max_input_length,
            system_prompt = eval(args.sampling_config.system_prompt),
        )
        train_data_loader = DataLoader(
            hf_dataset,
            batch_size=args.base_config.train_batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=args.base_config.num_workers,
        )
        
        ## sfl training step 
        tmp_train_setup = TrainingSetup(
            run_name = training_setup["run_name"],
            device = training_setup["device"],
            cpu_device = training_setup["cpu_device"],
            train_log_dir = training_setup["train_log_dir"],
            eval_log_dir = training_setup["eval_log_dir"],
            target_policy = model,
            tokenizer = training_setup["tokenizer"],
            pad_token_id = training_setup["pad_token_id"],
            dataset = hf_dataset,
            rollout_data_loader = train_data_loader,
            optimizer = training_setup["optimizer"],
            lr_scheduler = training_setup["lr_scheduler"],
            actor_loss = training_setup["actor_loss"],
            reward_model = training_setup["reward_model"],
            generation_config = training_setup["generation_config"],
            eval_config = training_setup["eval_config"],
            kl_controller = training_setup["kl_controller"],
            reference_model = training_setup["reference_model"],
        )
        model, train_outs, eval_outs = train(
            args=args,
            training_setup=tmp_train_setup,
            logger=logger,
            **{
                "global_batch_idx": global_batch_idx,
            }
        )

        ## increment the step
        global_batch_idx += len(train_data_loader)
        sfl_step += 1


    return model,train_outs, eval_outs


if __name__ == "__main__":
    
    args = tyro.cli(TrainingConfig)
    wandb.init(
        project=args.wandb_config.project,
        name=args.wandb_config.name,
        config=args,
    )
    
    # define the metrics
    wandb.define_metric("sfl_step")
    wandb.define_metric("sfl/mean_learnability", step_metric="sfl_step")
    wandb.define_metric("sfl/std_learnability", step_metric="sfl_step")
    wandb.define_metric("sfl/min_learnability", step_metric="sfl_step")
    wandb.define_metric("sfl/max_learnability", step_metric="sfl_step")

    # define the training metrics
    wandb.define_metric("num_batches_visited")
    wandb.define_metric("train/mean_batch_returns", step_metric="num_batches_visited")
    wandb.define_metric("train/mean_batch_solved_rate", step_metric="num_batches_visited")
    wandb.define_metric("train/max_input_length", step_metric="num_batches_visited")
    wandb.define_metric("train/mean_num_words_in_completions", step_metric="num_batches_visited")
    wandb.define_metric("train/mean_batch_format_returns", step_metric="num_batches_visited")
    wandb.define_metric("train/mean_batch_outcome_returns", step_metric="num_batches_visited")
    
    # set up the training
    training_setup = set_up_training(args)
    model, train_outs, eval_outs = train_sfl(
        args=args,
        training_setup=training_setup,
        logger=wandb.log,
    )

