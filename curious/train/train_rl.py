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
from curious.policy_gradient.ad_cispo import (
    ADCispoStats,
    ReferencePolicyFeatureRequest,
    collect_special_token_ids,
    compute_reference_policy_features,
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
    last_batch_idx = global_batch_idx if global_batch_idx is not None else 0
    for local_batch_idx, batch_inputs in tqdm(enumerate(rollout_data_loader), total=len(rollout_data_loader)):

        batch_idx = local_batch_idx + 1 if global_batch_idx is None else global_batch_idx + local_batch_idx + 1
        if args.base_config.max_train_batches > 0 and batch_idx > args.base_config.max_train_batches:
            print(f"Reached max_train_batches={args.base_config.max_train_batches}; stopping training.")
            break

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

            if args.rl_config.anneling_temperature:
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
                group_size=args.rl_config.group_size,
                seed=args.base_config.seed,
                normalize_centered_returns=args.rl_config.normalize_centered_returns,
                use_rloo_scalar=args.rl_config.use_rloo_scalar,
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
        
            entropy_interval = args.base_config.train_entropy_log_interval
            should_log_entropy = entropy_interval > 0 and batch_idx % entropy_interval == 0

            attention_mask: torch.Tensor = sequence_ids != pad_token_id # (num_samples * group_size, seq_len)
            log_probs, entropy = sequences_log_probs(
                model=model,
                sequence_ids=sequence_ids,
                attention_mask=attention_mask,
                return_entropy=should_log_entropy,
                logits_minibatch_size=args.rl_config.logits_minibatch_size,
            ) # (num_samples * group_size, seq_len-1)
            action_entropy = (
                masked_mean(entropy, action_mask, dim=None)
                if should_log_entropy and entropy is not None
                else None
            )

            kl, log_probs_ref, token_clip_high = None, None, None
            token_saliency, token_clip_multiplier = None, None
            ad_cispo_stats: ADCispoStats | None = None
            if args.rl_config.use_ad_cispo:
                ad_cispo_features = compute_reference_policy_features(
                    ReferencePolicyFeatureRequest(
                        model=model,
                        sequence_ids=sequence_ids,
                        attention_mask=attention_mask,
                        action_mask=action_mask,
                        clip_high=objective.epsilon_high,
                        logits_minibatch_size=args.rl_config.logits_minibatch_size,
                        top_layers=args.rl_config.ad_cispo_top_layers,
                        min_multiplier=args.rl_config.ad_cispo_min_multiplier,
                        max_multiplier=args.rl_config.ad_cispo_max_multiplier,
                        eps=args.rl_config.ad_cispo_eps,
                        return_log_probs=False,
                        saliency_method=args.rl_config.ad_cispo_saliency_method,
                        attention_block_size=args.rl_config.ad_cispo_attention_block_size,
                        sink_token_ids=collect_special_token_ids(tokenizer),
                    )
                )
                token_clip_high = ad_cispo_features.token_clip_thresholds.values
                token_saliency = ad_cispo_features.action_saliency.values
                token_clip_multiplier = ad_cispo_features.token_clip_thresholds.multipliers
                ad_cispo_stats = ad_cispo_features.stats

            if args.rl_config.kl_weight > 0:
                # compute the log probs of the reference model
                if reference_model is None:
                    raise RuntimeError("KL regularization requires a reference model.")
                log_probs_ref, _ = sequences_log_probs(
                    model=reference_model,
                    sequence_ids=sequence_ids,
                    attention_mask=attention_mask,
                    return_entropy=False,
                    logits_minibatch_size=args.rl_config.logits_minibatch_size,
                ) # (num_samples * group_size, seq_len-1)

            if args.rl_config.kl_weight > 0:
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
                token_clip_high=token_clip_high,
                token_saliency=token_saliency,
                token_clip_multiplier=token_clip_multiplier,
            )
            replay_buffer.append(experience.to(cpu_device))
            
        del (
            kl,
            log_probs_ref,
            token_clip_high,
            token_saliency,
            token_clip_multiplier,
            log_probs,
            entropy,
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
        log_payload = {
            "train/mean_batch_returns": batch_mean_returns,
            "train/mean_batch_solved_rate": batch_mean_solved_rate,

            "train/mean_num_words_in_completions": batch_mean_num_words_in_completions,
            "train/max_num_words_in_completions": batch_max_num_words_in_completions,
            "train/min_num_words_in_completions": batch_min_num_words_in_completions,

            "train/mean_batch_format_returns": batch_mean_format_returns,
            "train/mean_batch_outcome_returns": batch_mean_outcome_returns,
            "train/mean_batch_length_penalty": batch_mean_length_penalty,

            "train/lr": lr_scheduler.get_lr()[0],
            "num_batches_visited": batch_idx,
        }
        if action_entropy is not None:
            log_payload["train/mean_action_entropy"] = action_entropy.item()

        logger(log_payload)
        if ad_cispo_stats is not None:
            logger(
                {
                    "ad_cispo/clip_mean": ad_cispo_stats.clip_mean,
                    "ad_cispo/clip_min": ad_cispo_stats.clip_min,
                    "ad_cispo/clip_max": ad_cispo_stats.clip_max,
                    "ad_cispo/multiplier_mean": ad_cispo_stats.multiplier_mean,
                    "ad_cispo/saliency_mean": ad_cispo_stats.saliency_mean,
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
                        question = questions[i//args.rl_config.group_size]
                        answer = answers[i//args.rl_config.group_size]
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
            batch_size=args.rl_config.mini_batch_size,
            shuffle=True, #if args.rl_config.use_token_level_loss else False,
            drop_last=False,
            collate_fn=join_experience_batch,
            num_workers=0,
        )

        model.train()
        model.gradient_checkpointing_enable()
        for _ in range(args.rl_config.epochs_per_step):
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
                    logits_minibatch_size=args.rl_config.logits_minibatch_size,
                )
                kl_weight_used = objective.kl_weight if args.rl_config.kl_weight > 0 else 0.0
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
                    max_norm=args.rl_config.max_grad_norm,
                )
                optimizer.step()

                ### cpu ops ### 
                mean_loss = loss.detach().cpu().item()
                mean_kl = mean_kl.cpu().item()
                mean_actor_loss = mean_actor_loss.cpu().item()
                mean_kl_loss = mean_loss - mean_actor_loss
                
                kl_weight_next = kl_weight_used
                if args.rl_config.kl_weight > 0:
                    kl_controller.update(mean_kl, args.rl_config.mini_batch_size)
                    objective.kl_weight = kl_controller.value
                    kl_weight_next = kl_controller.value
                
                logger(
                    {
                        "train/grad_norm": grad_norm,
                        "train/loss": mean_loss,
                        "train/actor_loss": mean_actor_loss,
                        "train/kl_loss": mean_kl_loss,
                        "train/mean_kl": mean_kl, 
                        "train/kl_weight": kl_weight_used,
                        "train/kl_weight_used": kl_weight_used,
                        "train/kl_weight_next": kl_weight_next,
                    }
                )

                del (
                    grad_norm,
                    loss,
                    mean_loss,
                    mean_kl,
                    mean_actor_loss,
                    mean_kl_loss,
                    kl_weight_used,
                    kl_weight_next,
                )
                gc.collect()
                torch.cuda.empty_cache()
                ### cpu ops ### 
        ### ----- Training phase END ----- ###

        ### ----- Update ref model phase START ----- ###
        if args.rl_config.ref_model_update_freq > 0:
            if (
                reference_model is not None
                and args.rl_config.ref_model_update_freq > 0
                and batch_idx % args.rl_config.ref_model_update_freq == 0
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
        if args.rl_config.anneling_lr:
            lr_scheduler.step()

        if args.sfl_config.sfl_enabled:
            global_batch_idx += 1
        last_batch_idx = batch_idx
    
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
                f"step_{last_batch_idx}_final.pt"
            )
        )
        # evaluate the final model
        eval_results = evaluate(
            config=eval_config,
            model=model,
            tokenizer=tokenizer,
            logger=logger,
            **{
                "batch_idx": last_batch_idx,
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
    training_setup, _ = set_up_training(args)
    
    wandb.init(
        entity="moed",
        project=args.wandb_config.project,
        name=args.wandb_config.name,
        config=args,
    )
    wandb.define_metric("num_batches_visited")
    wandb.define_metric("train/*", step_metric="num_batches_visited")
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
