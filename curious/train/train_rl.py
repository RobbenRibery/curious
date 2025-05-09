import os 
from typing import Callable, List, Dict, Any, Tuple

from transformers import PreTrainedModel

import torch 
from torch.utils.data import DataLoader 
from torch.nn.utils import clip_grad_norm_

from curious.utils.utils import LOGGING_TEMPLATE
from curious.sampling.sampling import (
    rollout, 
    sequences_log_probs, 
    linear_temperature_annealing,
)
from curious.replay.experience import (
    ReplayBuffer, 
    Experience, 
    join_experience_batch
)
from curious.policy_gradient.loss import (
    approx_kl_divergence, 
    masked_mean,
)
from curious.train.training_setup import (
    TrainingSetup, 
    set_up_training,
    TrainState,
)
from curious.prompt import *

from curious.config import TrainingConfig
from curious.evaluate import evaluate

from accelerate.utils import set_seed

import wandb
import numpy as np
from tqdm import tqdm
from rich import print

import tyro 
import gc
    
def train(
    args:TrainingConfig, 
    training_setup: TrainingSetup, 
    logger: Callable,
    **kwargs,
    ) -> Tuple[
        PreTrainedModel, 
        List[Dict[str, Any]], 
        List[Dict[str, Any]]
    ]:
    """
    Train the model.
    Args:
        training_setup (TrainingSetup): The training setup.
        logger (Callable): The logger to use.
        kwargs (Dict[str, Any]): The kwargs to use.
            - global_batch_idx (int): The global batch index.

    Returns:
        Tuple[
            PreTrainedModel, 
            List[Dict[str, Any]], 
            List[Dict[str, Any]]
        ]: The training outputs and the evaluation outputs.
    """
    # get the kwargs
    global_batch_idx = kwargs.get("global_batch_idx", None)
    if args.sfl_config.sfl_enabled:
        assert global_batch_idx is not None, "global_batch_idx must be provided if SFL is enabled"

    # outputs 
    train_outs:List[Dict[str, Any]] = []
    eval_outs:List[Dict[str, Any]] = []

    # set the seed for accelerate
    set_seed(args.base_config.seed)

    # unload the training setup 
    run_name = training_setup["run_name"]
    device = training_setup["device"]
    cpu_device = training_setup["cpu_device"]

    # get the target policy
    model = training_setup["target_policy"]
    tokenizer = training_setup["tokenizer"]
    pad_token_id = training_setup["pad_token_id"]

    reference_model = training_setup["reference_model"]
    kl_controller = training_setup["kl_controller"]

    # Evaluation config
    eval_config = training_setup["eval_config"]
    if args.sfl_config.sfl_enabled and global_batch_idx > 0:
        pass 
    else:
        evaluate(
            config=eval_config,
            model=model,
            tokenizer=tokenizer,
            logger=logger,
            **{
                "batch_idx": 0 if not global_batch_idx else global_batch_idx + 1,
            }
        )  

    ## get the rollout data loader
    rollout_data_loader = training_setup["rollout_data_loader"]
    ## get the optimizer
    optimizer = training_setup["optimizer"]
    ## get the lr scheduler
    lr_scheduler = training_setup["lr_scheduler"]
    ## get the objective
    objective = training_setup["actor_loss"]
    ## get the reward model
    reward_model = training_setup["reward_model"]
    ## get the generation config
    generation_config = training_setup["generation_config"]

    ## Replay buffer
    replay_buffer = ReplayBuffer()
    for batch_idx, batch_inputs in tqdm(enumerate(rollout_data_loader), total=len(rollout_data_loader)):

        batch_idx = batch_idx + 1 if not global_batch_idx else global_batch_idx + 1
        questions = batch_inputs["question"]
        answers = batch_inputs["answer"]

        print(f"Batch indx {batch_idx}")
        print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/(1024**3)))
        print("torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(0)/(1024**3)))
        print("torch.cuda.max_memory_reserved: %fGB"%(torch.cuda.max_memory_reserved(0)/(1024**3)))

        replay_buffer.clear()

        ### ----- Rollout phase START ----- ###
        with torch.no_grad():
            
            model.eval()
            model.gradient_checkpointing_disable()

            if args.grpo_config.anneling_temperature:
                generation_config.temperature = linear_temperature_annealing(
                    current_step=batch_idx,
                    total_steps=len(rollout_data_loader),
                    start_temp=args.sampling_config.temperature,
                    end_temp=args.sampling_config.temperature * 0.7,
                )

            # rollout
            rollout_out = rollout(
                model,
                tokenizer,
                batch_inputs,
                reward_model=reward_model,
                generation_config=generation_config,
                group_size=args.grpo_config.group_size,
                seed=args.base_config.seed,
                normalize_centered_returns=args.grpo_config.normalize_centered_returns,
                use_rloo_scalar=args.grpo_config.use_rloo_scalar,
            )
            # sequence_ids: (num_samples * group_size, seq_len)
            # action_mask: (num_samples * group_size, seq_len)
            # completions: (num_samples * group_size)
            # returns: (num_samples, group_size)
            # solved_masks: (num_samples, group_size)
            # infos: [{format_reward, outcome_reward, ...}, ...]

            info_list: List[Dict[str, float]] = rollout_out["infos"]
            batch_mean_format_returns: float = np.array([x["format_reward"] for x in info_list]).mean()
            batch_mean_outcome_returns: float = np.array([x["outcome_reward"] for x in info_list]).mean()
            batch_mean_length_penalty: float = np.array([x["length_penalty"] for x in info_list]).mean()

            batch_mean_returns: float = rollout_out["returns"].mean().item()
            batch_mean_solved_rate: float = rollout_out["solved_masks"].mean().item()

            batch_mean_num_words_in_completions: float = rollout_out["num_words_in_completions"].mean().item()
            batch_max_num_words_in_completions: float = rollout_out["num_words_in_completions"].max().item()
            batch_min_num_words_in_completions: float = rollout_out["num_words_in_completions"].min().item()

            # compute the log probs
            returns: torch.Tensor = rollout_out["returns"].reshape(-1)
            advantages: torch.Tensor = rollout_out["advantages"].reshape(-1)
            solved_mask: torch.Tensor = rollout_out["solved_masks"].reshape(-1)

            sequence_ids: torch.Tensor = rollout_out["sequence_ids"]
            action_mask: torch.Tensor = rollout_out["action_mask"]
            completions:List[str] = rollout_out["completions"]
        
            attention_mask: torch.Tensor = sequence_ids != pad_token_id # (num_samples * group_size, seq_len)
            log_probs, entropy = sequences_log_probs(
                model=model,
                sequence_ids=sequence_ids,
                attention_mask=attention_mask,
                return_entropy=True,
                logits_minibatch_size=args.grpo_config.logits_minibatch_size,
            ) # (num_samples * group_size, seq_len-1)
            action_entropy = masked_mean(entropy, action_mask, dim=None)

            kl, log_probs_ref = None, None
            if args.grpo_config.kl_weight > 0:
                # compute the log probs of the reference model
                log_probs_ref, _ = sequences_log_probs(
                    model=reference_model,
                    sequence_ids=sequence_ids,
                    attention_mask=attention_mask,
                    return_entropy=False,
                    logits_minibatch_size=args.grpo_config.logits_minibatch_size,
                ) # (num_samples * group_size, seq_len-1)

                # compute the kl divergence
                kl: torch.Tensor = approx_kl_divergence(
                    log_probs=log_probs,
                    log_probs_ref=log_probs_ref,
                    action_mask=action_mask,
                ) # (num_samples * group_size, seq_len-1)
            
            experience: Experience = Experience(
                sequences=sequence_ids,
                action_log_probs=log_probs,
                returns=returns,
                solved_mask=solved_mask,
                advantages=advantages,
                attention_mask=attention_mask,
                action_mask=action_mask,
                log_probs_ref=log_probs_ref,
                kl=kl,
            )
            replay_buffer.append(experience.to(cpu_device))
            
        del (
            kl,
            log_probs_ref,
            log_probs,
            attention_mask,
            sequence_ids,
            action_mask,
            rollout_out,
        )
        gc.collect()
        torch.cuda.empty_cache()
        ### ----- Rollout phase END ----- ###

        ### ----- Logging phase START ----- ###
        # log the stats to wandb 
        logger(
            {
                "train/mean_batch_returns": batch_mean_returns,
                "train/mean_batch_solved_rate": batch_mean_solved_rate,

                "train/mean_num_words_in_completions": batch_mean_num_words_in_completions,
                "train/max_num_words_in_completions": batch_max_num_words_in_completions,
                "train/min_num_words_in_completions": batch_min_num_words_in_completions,

                "train/mean_batch_format_returns": batch_mean_format_returns,
                "train/mean_batch_outcome_returns": batch_mean_outcome_returns,
                "train/mean_batch_length_penalty": batch_mean_length_penalty,
                
                "train/lr": lr_scheduler.get_lr()[0],
                "train/mean_action_entropy": action_entropy.item(),
                
                "num_batches_visited": batch_idx,
            }
        )
        del action_entropy
        gc.collect()
        torch.cuda.empty_cache()
        print(
            "--------------------------------\n"
            f"batch_idx: {batch_idx} |\n "
            f"returns: {batch_mean_returns} |\n "
            f"solved_rate: {batch_mean_solved_rate} |\n "
            f"format_returns: {batch_mean_format_returns} |\n "
            f"outcome_returns: {batch_mean_outcome_returns}\n"
            "--------------------------------"
        )
        train_outs.append(
            {
                "mean_batch_returns": batch_mean_returns,
                "mean_batch_solved_rate": batch_mean_solved_rate,
                "mean_batch_format_returns": batch_mean_format_returns,
                "mean_batch_outcome_returns": batch_mean_outcome_returns,
                "mean_num_words_in_completions": batch_mean_num_words_in_completions,
            }
        )
        if args.base_config.train_text_log_interval > 0:
            if batch_idx % args.base_config.train_text_log_interval == 0:
                file_name = os.path.join(training_setup["train_log_dir"], f"log_{batch_idx}.txt")
                with open(file_name, "a") as f:
                    for i, completion in enumerate(completions):
                        question = questions[i//args.grpo_config.group_size]
                        answer = answers[i//args.grpo_config.group_size]
                        reward = returns[i]
                        info = info_list[i]     

                        text_to_log = LOGGING_TEMPLATE.format(
                            question=question,
                            answer=answer,
                            completion=completion,
                            reward=reward,
                            info=info,
                        )
                        f.write(text_to_log)
                f.close()
            ### ----- Logging phase END ----- ###
        
        ### ----- Training phase START ----- ###
        experience_sampler = DataLoader(
            replay_buffer,
            batch_size=args.grpo_config.mini_batch_size,
            shuffle=True, #if args.grpo_config.use_token_level_loss else False,
            drop_last=False,
            collate_fn=join_experience_batch,
            num_workers=args.base_config.num_workers,
        )

        model.train()
        model.gradient_checkpointing_enable()
        for _ in range(args.grpo_config.epochs_per_step):
            for exp in experience_sampler:
                optimizer.zero_grad()
                # get the experience to cuda 
                exp: Experience
                exp = exp.to(device)
                log_probs, _ = sequences_log_probs(
                    model, 
                    sequence_ids=exp.sequences, 
                    attention_mask=exp.attention_mask,
                    return_entropy=False,
                    logits_minibatch_size=args.grpo_config.logits_minibatch_size,
                )
                loss, mean_kl, mean_actor_loss = objective(log_probs=log_probs, experience=exp)
                
                del log_probs
                del exp 

                if not loss.isfinite():
                    print(f"Loss not finite, skipping backward, loss={loss}")
                    print(f"experience.advantages={experience.advantages}")
                    continue
                
                loss.backward()
                grad_norm = clip_grad_norm_(
                    model.parameters(), 
                    max_norm=args.grpo_config.max_grad_norm,
                )
                optimizer.step()

                ### cpu ops ### 
                mean_kl = mean_kl.cpu().item()
                mean_actor_loss = mean_actor_loss.cpu().item()
                
                if args.grpo_config.kl_weight > 0:
                    kl_controller.update(mean_kl, args.grpo_config.mini_batch_size)
                    objective.kl_weight = kl_controller.value
                
                logger(
                    {
                        "train/grad_norm": grad_norm,
                        "train/loss": loss,
                        "train/actor_loss": mean_actor_loss,
                        "train/mean_kl": mean_kl, 
                        "train/kl_weight": kl_controller.value if args.grpo_config.kl_weight > 0 else 0.0,
                    }
                )

                del (
                    grad_norm,
                    loss,
                    mean_kl,
                    mean_actor_loss,
                )
                gc.collect()
                torch.cuda.empty_cache()
                ### cpu ops ### 
        ### ----- Training phase END ----- ###

        ### ----- Update ref model phase START ----- ###
        if args.grpo_config.ref_model_update_freq > 0:
            if (
                args.grpo_config.kl_weight > 0
                and args.grpo_config.ref_model_update_freq > 0
                and batch_idx % args.grpo_config.ref_model_update_freq == 0
            ):
                reference_model.load_state_dict(
                    model.state_dict()
                )
                reference_model.eval()
        ### ----- Update ref model phase END ----- ###

        ### ----- Interval checkpoint phase START ----- ###
        if args.base_config.checkpoint_interval > 0:
            if (
                args.base_config.checkpoint_dir is not None
                and args.base_config.checkpoint_interval is not None
                and batch_idx % args.base_config.checkpoint_interval == 0
            ):
                state_dict_dir = os.path.join(
                    args.base_config.checkpoint_dir, 
                    run_name,
                )
                os.makedirs(state_dict_dir, exist_ok=True)
                torch.save(
                    model.state_dict(),
                    os.path.join(
                        state_dict_dir,
                        f"step_{batch_idx}.pt"
                    )
                )
        ### ----- Interval checkpoint phase END ----- ###

        ### ----- Interval evaluation phase START ----- ###
        if args.base_config.eval_interval > 0:
            if batch_idx % args.base_config.eval_interval == 0:
                eval_results = evaluate(
                    config=eval_config,
                    model=model,
                    tokenizer=tokenizer,
                    logger=logger,
                    **{
                        "batch_idx": batch_idx,
                    }
                )  
                eval_outs.append(eval_results)
        ### ----- Interval evaluation phase END ----- ###
        
        del batch_inputs
        if args.grpo_config.anneling_lr:
            lr_scheduler.step()

        if args.sfl_config.sfl_enabled:
            global_batch_idx += 1
    
    ### ----- Final checkpoint phase START ----- ###
    if args.base_config.checkpoint_dir is not None:
        # save the final state dict
        state_dict_dir = os.path.join(
            args.base_config.checkpoint_dir, 
            run_name,
        )
        os.makedirs(state_dict_dir, exist_ok=True)
        torch.save(
            model.state_dict(),
            os.path.join(
                state_dict_dir,
                f"step_{batch_idx}_final.pt"
            )
        )
        # evaluate the final model
        eval_results = evaluate(
            config=eval_config,
            model=model,
            tokenizer=tokenizer,
            logger=logger,
            **{
                "batch_idx": batch_idx,
            }
        )  
        eval_outs.append(eval_results)
    ### ----- Final checkpoint phase END ----- ###
    
    return TrainState(
        run_name=run_name,
        device=device,
        cpu_device=cpu_device,
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        reference_model=reference_model,
        kl_controller=kl_controller,
    ), train_outs, eval_outs


if __name__ == "__main__":
    
    args = tyro.cli(TrainingConfig)
    training_setup = set_up_training(args)
    
    wandb.init(
        entity="moed",
        project=args.wandb_config.project,
        name=args.wandb_config.name,
        config=args,
    )
    wandb.define_metric("num_batches_visited")
    wandb.define_metric("train/mean_batch_returns", step_metric="num_batches_visited")
    wandb.define_metric("train/mean_batch_solved_rate", step_metric="num_batches_visited")
    wandb.define_metric("train/max_input_length", step_metric="num_batches_visited")
    wandb.define_metric("train/mean_num_words_in_completions", step_metric="num_batches_visited")
    wandb.define_metric("train/mean_batch_format_returns", step_metric="num_batches_visited")
    wandb.define_metric("train/mean_batch_outcome_returns", step_metric="num_batches_visited")
    
    final_model, train_outs, eval_outs = train(
        args=args,
        training_setup=training_setup,
        logger=wandb.log,
    )