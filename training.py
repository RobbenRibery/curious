import os 
os.environ["TOKENIZERS_PARALLELIS"] = "true"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
from typing import Callable

import torch 
from torch.utils.data import DataLoader 
from torch.nn.utils import clip_grad_norm_
import torch.optim as optim

from transformers import GenerationConfig

from curious.data import GSM8KDataset
from curious.utils import tokenize_questions, load_model_tokenizer
from curious.sampling import rollout, sequences_log_probs, compute_group_advantages
from curious.buffer import ReplayBuffer, Experience, join_experience_batch
from curious.loss import approx_kl_divergence, ActorLoss
from curious.reward import GSM8KRewardModel
from curious.prompt import *
from config import GRPOConfig, WandbConfig, BaseConfig, SamplingConfig, RewardConfig

from accelerate.utils import set_seed
import wandb
import numpy as np

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


def train(args:TrainingConfig, logger: Callable) -> None:

    run_name = args.wandb_config.name.replace("-", "_")

    # check that the data mode is train
    assert args.base_config.mode == "train"
    assert args.grpo_config.mini_batch_size % args.grpo_config.group_size == 0
    assert args.base_config.batch_size * args.grpo_config.group_size % args.grpo_config.mini_batch_size == 0

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
        batch_size=args.base_config.batch_size,
        shuffle=True,
        drop_last=False,
    )
    
    ## Replay buffer
    replay_buffer = ReplayBuffer()

    ## Objective
    objective = ActorLoss(
        clip_eps=args.grpo_config.clip_eps,
        kl_weight=args.grpo_config.kl_weight,
        epsilon=args.grpo_config.clip_eps,
        epsilon_high=args.grpo_config.clip_eps_high,
        use_clip_high=args.grpo_config.use_clip_high,
        use_token_level_loss=args.grpo_config.use_token_level_loss,
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

    for batch_idx, batch_inputs in enumerate(rollout_data_loader):
        
        print(f"Batch indx {batch_idx}")
        print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/(1024**3)))
        print("torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(0)/(1024**3)))
        print("torch.cuda.max_memory_reserved: %fGB"%(torch.cuda.max_memory_reserved(0)/(1024**3)))

        replay_buffer.clear()

        ### ----- Rollout phase START ----- ###
        with torch.no_grad():
            
            model.eval()
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
            )
            # sequence_ids: (num_samples * group_size, seq_len)
            # action_mask: (num_samples * group_size, seq_len)
            # completions: (num_samples * group_size)
            # returns: (num_samples, group_size)
            # solved_masks: (num_samples, group_size)
            # infos: [[{format_reward: float, outcome_reward: float}, ...], ...]

            info_list = rollout_out["infos"]
            batch_mean_format_returns = np.array([x["format_reward"] for x in info_list]).mean()
            batch_mean_outcome_returns = np.array([x["outcome_reward"] for x in info_list]).mean()
            batch_mean_returns = rollout_out["returns"].mean().item()
            batch_mean_solved_rate = rollout_out["solved_masks"].mean().item()
            batch_mean_num_words_in_completions = rollout_out["num_words_in_completions"].mean().item()

            # compute the log probs
            returns = rollout_out["returns"].reshape(-1)
            advantages = rollout_out["advantages"].reshape(-1)
            solved_mask = rollout_out["solved_masks"].reshape(-1)

            sequence_ids = rollout_out["sequence_ids"]
            action_mask = rollout_out["action_mask"]
        
            attention_mask = sequence_ids != pad_token_id # (num_samples * group_size, seq_len)
            log_probs = sequences_log_probs(
                model=model,
                sequence_ids=sequence_ids,
                attention_mask=attention_mask,
            ) # (num_samples * group_size, seq_len-1)

            kl, log_probs_ref = None, None
            if args.grpo_config.kl_weight > 0:
                # compute the log probs of the reference model
                log_probs_ref = sequences_log_probs(
                    model=reference_model,
                    sequence_ids=sequence_ids,
                    attention_mask=attention_mask,
                ) # (num_samples * group_size, seq_len-1)

                # compute the kl divergence
                kl = approx_kl_divergence(
                    log_probs=log_probs,
                    log_probs_ref=log_probs_ref,
                    action_mask=action_mask,
                ) # (num_samples * group_size, seq_len-1)
            
            experience = Experience(
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
            
        del kl
        del log_probs_ref
        del log_probs 
        del attention_mask
        del sequence_ids
        del action_mask
        del rollout_out
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
                "num_batches_visited": batch_idx + 1,
            }
        )
        print(
            f"batch_idx: {batch_idx} | returns: {batch_mean_returns.item()} | solved_rate: {batch_mean_solved_rate.item()} | format_returns: {batch_mean_format_returns} | outcome_returns: {batch_mean_outcome_returns}"
        )
        out_dir = os.path.join(args.base_config.log_dir, args.wandb_config.name)
        os.makedirs(out_dir, exist_ok=True)

        delimeter = "*"*50
        file_name = os.path.join(out_dir, f"log_{batch_idx}.txt")
        completions = rollout_out["completions"]
        questions = batch_inputs["questions"]
        answers = batch_inputs["answers"]
        with open(file_name, "a") as f:
            for i, completion in enumerate(completions):
                question = questions[i//args.grpo_config.group_size]
                answer = answers[i//args.grpo_config.group_size]
                reward = returns[i]
                info = info_list[i]                
                f.write(
                    f"{delimeter}\n[Question]:\n{question}\n[Canonical Answer]:\n{answer}\n"
                    f"[Completion]:\n{completion}\n[Reward]:\n{reward}\n"
                    f"[Info]:\n{info}\n{delimeter}\n"
                )
        f.close()
        ### ----- Logging phase END ----- ###
        
        ### ----- Training phase START ----- ###
        experience_sampler = DataLoader(
            replay_buffer,
            batch_size=args.grpo_config.mini_batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=join_experience_batch,
        )
        model.train()

        for _ in range(args.grpo_config.epochs_per_step):
            for exp in experience_sampler:

                # get the experience to cuda 
                exp: Experience
                exp = exp.to(device)

                optimizer.zero_grad()

                log_probs = sequences_log_probs(
                    model, 
                    sequence_ids=exp.sequences, 
                    attention_mask=exp.attention_mask
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

                del grad_norm
                del loss 
                del mean_kl
                gc.collect()
                torch.cuda.empty_cache()
                
                logger(
                    {
                        "train/mean_kl": mean_kl, 
                        "train/grad_norm": grad_norm,
                        "train/loss": loss,
                    }
                )
        ### ----- Training phase END ----- ###

        ### ----- Interval checkpoint phase START ----- ###
        if (
            args.base_config.checkpoint_dir is not None
            and args.base_config.checkpoint_interval is not None
            and (batch_idx + 1) % args.base_config.checkpoint_interval == 0
        ):
            model.save_pretrained(
                os.path.join(
                    *[
                        args.base_config.checkpoint_dir, 
                        run_name,
                        f"step_{batch_idx + 1}"
                    ]
                )
            )
        del batch_inputs
        ### ----- Interval checkpoint phase END ----- ###

    ### ----- Final checkpoint phase START ----- ###
    if args.base_config.checkpoint_dir is not None:
        model.save_pretrained(
            os.path.join(
                *[
                    args.base_config.checkpoint_dir, 
                    run_name,
                    f"step_{batch_idx + 1}_final"
                ]
            )
        )
    ### ----- Final checkpoint phase END ----- ###

if __name__ == "__main__":

    args = tyro.cli(TrainingConfig)
    
    wandb.init(
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
    
    train(args, logger)