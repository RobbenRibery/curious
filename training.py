import os 
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
from typing import Callable, List, Dict, Any, Tuple

import torch 
from torch.utils.data import DataLoader 
from torch.nn.utils import clip_grad_norm_
import torch.optim as optim

from transformers import GenerationConfig

from curious.data import GSM8KDataset
from curious.utils import LOGGING_TEMPLATE, load_model_tokenizer
from curious.sampling import rollout, sequences_log_probs, sequences_log_probs_entropy
from curious.buffer import ReplayBuffer, Experience, join_experience_batch
from curious.loss import ActorLoss, approx_kl_divergence, masked_mean
from curious.reward import GSM8KRewardModel
from curious.prompt import *

from config import GRPOConfig, WandbConfig, BaseConfig, SamplingConfig, RewardConfig
from evaluate import FixedSamplingConfig, EvaluationConfig, evaluate

from accelerate.utils import set_seed
import wandb
import numpy as np
from tqdm import tqdm
from rich import print

from dataclasses import dataclass
import tyro 
import gc

@dataclass
class TrainingConfig:
    """
    A dataclass for storing the training configuration.
    """

    grpo_config: GRPOConfig
    """
    The GRPO configuration.
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


def train(args:TrainingConfig, logger: Callable) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Train the model.
    Args:
        args (TrainingConfig): The training configuration.
        logger (Callable): The logger to use.

    Returns:
        Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]: The training outputs and the evaluation outputs.
    """
    # outputs 
    train_outs:List[Dict[str, Any]] = []
    eval_outs:List[Dict[str, Any]] = []

    # get the run name
    run_name = args.wandb_config.name.replace("-", "_")

    # check that the data mode is train
    assert args.base_config.mode == "train"
    assert args.grpo_config.mini_batch_size % args.grpo_config.group_size == 0
    assert args.base_config.train_batch_size * args.grpo_config.group_size % args.grpo_config.mini_batch_size == 0

    # device & seeding
    device = torch.device("cuda", args.base_config.device_index)
    cpu_device = torch.device("cpu")
    set_seed(args.base_config.seed)
    
    ## ref policy
    if args.grpo_config.kl_weight > 0:
        reference_model, _ = load_model_tokenizer(
            args.base_config.model_name, 
            device_map=device, 
            freeze_model=True,
        )
        reference_model.eval()

    ## target policy
    model, tokenizer = load_model_tokenizer(
        args.base_config.model_name, 
        device_map=device
    )
    model.gradient_checkpointing_enable()

    ## Tokenizer
    tokenizer.padding_side  = 'left'
    pad_token_id = tokenizer.eos_token_id

    ## Optimizer
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=args.grpo_config.lr,
        weight_decay=args.grpo_config.weight_decay,
    )

    ## Data
    dataset = GSM8KDataset(
        tokenizer=tokenizer,
        dataset_name=args.base_config.dataset_name,
        seed=args.base_config.seed,
        mode=args.base_config.mode,
        max_prompt_length=args.sampling_config.model_prompt_length,
        system_prompt=eval(args.sampling_config.system_prompt),
    )
    rollout_data_loader = DataLoader(
        dataset,
        batch_size=args.base_config.train_batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.base_config.num_workers,
    )

    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=len(rollout_data_loader),
    )
    
    ## Replay buffer
    replay_buffer = ReplayBuffer()

    ## Objective
    objective = ActorLoss(
        epsilon=args.grpo_config.clip_eps,
        epsilon_high=args.grpo_config.clip_eps_high,
        kl_weight=args.grpo_config.kl_weight,
        use_clip_high=args.grpo_config.use_clip_high,
        use_token_level_loss=args.grpo_config.use_token_level_loss,
        use_fixed_response_length=args.grpo_config.use_fixed_response_length,
        use_surrogate_loss=args.grpo_config.use_surrogate_loss,
    )

    ## Reward model
    reward_model = GSM8KRewardModel(
        answer_pattern=args.reward_config.answer_pattern,
        think_pattern=args.reward_config.think_pattern,
        use_format_reward=args.reward_config.use_format_reward,
    )

    ## Sampling config
    generation_config = GenerationConfig(
        num_return_sequences=args.grpo_config.group_size,
        max_new_tokens=args.sampling_config.max_new_tokens,
        temperature=args.sampling_config.temperature,
        top_p=args.sampling_config.top_p,
        top_k=args.sampling_config.top_k,
        do_sample =args.sampling_config.do_sample,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id= tokenizer.eos_token_id,
        use_cache=args.sampling_config.use_cache,
        repetition_penalty=args.sampling_config.repetition_penalty,
    )

    ## Evaluation config
    eval_config = EvaluationConfig(
        base_config=args.base_config,
        sampling_config=FixedSamplingConfig(),
        reward_config=args.reward_config,
        wandb_config=args.wandb_config,
    )
    evaluate(
        config=eval_config,
        model=model,
        tokenizer=tokenizer,
        logger=logger,
        **{"batch_idx": 0}
    )  

    # create the logging directory for training logs
    out_dir = os.path.join(args.base_config.train_log_dir, args.wandb_config.name)
    os.makedirs(out_dir, exist_ok=True)

    for batch_idx, batch_inputs in tqdm(enumerate(rollout_data_loader), total=len(rollout_data_loader)):

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
            batch_mean_returns: float = rollout_out["returns"].mean().item()
            batch_mean_solved_rate: float = rollout_out["solved_masks"].mean().item()
            batch_mean_num_words_in_completions: float = rollout_out["num_words_in_completions"].mean().item()

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
                "train/mean_batch_format_returns": batch_mean_format_returns,
                "train/mean_batch_outcome_returns": batch_mean_outcome_returns,
                "train/lr": lr_scheduler.get_lr()[0],
                "train/mean_action_entropy": action_entropy.item(),
                "num_batches_visited": batch_idx + 1,
            }
        )
        del action_entropy
        gc.collect()
        torch.cuda.empty_cache()
        print(
            f"batch_idx: {batch_idx} | returns: {batch_mean_returns} | solved_rate: {batch_mean_solved_rate} | format_returns: {batch_mean_format_returns} | outcome_returns: {batch_mean_outcome_returns}"
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
        if (batch_idx + 1) % args.base_config.train_text_log_interval == 0:
            file_name = os.path.join(out_dir, f"log_{batch_idx + 1}.txt")

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
            shuffle=True,
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
                )
                loss, mean_kl = objective(log_probs=log_probs, experience=exp)
                loss: torch.Tensor
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
                
                logger(
                    {
                        "train/mean_kl": mean_kl, 
                        "train/grad_norm": grad_norm,
                        "train/loss": loss,
                    }
                )

                del (
                    grad_norm,
                    loss,
                    mean_kl,
                )
                gc.collect()
                torch.cuda.empty_cache()
        ### ----- Training phase END ----- ###

        ### ----- Update ref model phase START ----- ###
        if (
            args.grpo_config.kl_weight > 0
            and args.grpo_config.ref_model_update_freq > 0
            and (batch_idx + 1) % args.grpo_config.ref_model_update_freq == 0
        ):
            reference_model.load_state_dict(
                model.state_dict()
            )
        ### ----- Update ref model phase END ----- ###


        ### ----- Interval checkpoint phase START ----- ###
        if (
            args.base_config.checkpoint_dir is not None
            and args.base_config.checkpoint_interval is not None
            and (batch_idx + 1) % args.base_config.checkpoint_interval == 0
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
                    f"step_{batch_idx + 1}.pt"
                )
            )
        ### ----- Interval checkpoint phase END ----- ###

        ### ----- Interval evaluation phase START ----- ###
        if (batch_idx + 1) % args.base_config.eval_interval == 0:
            eval_results = evaluate(
                config=eval_config,
                model=model,
                tokenizer=tokenizer,
                logger=logger,
                **{
                    "batch_idx": batch_idx + 1,
                }
            )  
            eval_outs.append(eval_results)
        ### ----- Interval evaluation phase END ----- ###
        del batch_inputs
        if args.grpo_config.anneling_lr:
            lr_scheduler.step()
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
                f"step_{batch_idx + 1}_final.pt"
            )
        )
        # evaluate the final model
        eval_results = evaluate(
            config=eval_config,
            model=model,
            tokenizer=tokenizer,
            logger=logger,
            **{
                "batch_idx": batch_idx+1,
            }
        )  
        eval_outs.append(eval_results)
    ### ----- Final checkpoint phase END ----- ###

    return train_outs, eval_outs

if __name__ == "__main__":

    args = tyro.cli(TrainingConfig)
    
    wandb.init(
        entity=args.wandb_config.entity,
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
    
    logger = wandb.log  
    train_outs, eval_outs = train(args, logger)