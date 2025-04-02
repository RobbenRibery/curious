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
from curious.grpo import rollout, sequences_log_probs, group_advantages
from curious.buffer import ReplayBuffer, Experience, join_experience_batch
from curious.loss import GRPOLoss, approx_kl_divergence
from curious.reward import GSM8KRewardModel
from config import GRPOConfig, WandbConfig, BaseConfig, SamplingConfig, RewardConfig

from lightning import seed_everything
import wandb
import numpy as np

from dataclasses import dataclass
from pathlib import Path
import tyro 

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

    # check that the data mode is train
    assert args.base_config.mode == "train"
    #assert args.base_config.batch_size  args.grpo_config.mini_batch_size == 0

    # device & seeding
    device = torch.device("cuda", args.base_config.device_index)
    cpu_device = torch.device("cpu")
    seed_everything(args.base_config.seed)

    # load models
    reference_model, _ = load_model_tokenizer(
        args.base_config.model_name, 
        device_map=device, 
        freeze_model=True
    )
    model, tokenizer = load_model_tokenizer(
        args.base_config.model_name, 
        device_map=device
    )
    tokenizer.padding_side  = 'left'
    optimizer = optim.AdamW(model.parameters(), lr=args.grpo_config.lr)
    reference_model.eval()
    model.gradient_checkpointing_enable()

    # pad token id
    pad_token_id = tokenizer.eos_token_id

    # data
    dataset = GSM8KDataset(
        dataset_name=args.base_config.dataset_name,
        seed=args.base_config.seed,
        mode=args.base_config.mode,
    )
    rollout_data_loader = DataLoader(
        dataset,
        batch_size=args.base_config.batch_size,
        shuffle=True,
        drop_last=True,
        
    )
    
    # replay buffer
    replay_buffer = ReplayBuffer()

    # objective
    objective = GRPOLoss(
        clip_eps=args.grpo_config.clip_eps,
        kl_weight=args.grpo_config.kl_weight,
    )

    # reward model
    reward_model = GSM8KRewardModel(
        answer_pattern=args.reward_config.answer_pattern,
        think_pattern=args.reward_config.think_pattern,
        use_format_reward=args.reward_config.use_format_reward,
    )

    # sampling config
    generation_config = GenerationConfig(
        num_return_sequences=args.grpo_config.group_size,
        max_new_tokens=args.sampling_config.max_new_tokens,
        temperature=args.sampling_config.temperature,
        top_p=args.sampling_config.top_p,
        top_k=args.sampling_config.top_k,
        do_sample =args.sampling_config.do_sample,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id= tokenizer.eos_token_id,
    )

    for batch_idx, batch in enumerate(rollout_data_loader):
        
        print(f"Batch indx {batch_idx}")
        print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
        print("torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(0)/1024/1024/1024))
        print("torch.cuda.max_memory_reserved: %fGB"%(torch.cuda.max_memory_reserved(0)/1024/1024/1024))

        replay_buffer.clear()
        questions = batch["question"]
        answers = batch["answer"]

        batch_inputs = tokenize_questions(
            tokenizer=tokenizer,
            questions=questions,
            trucation_max_length=args.base_config.model_max_length_inuse,
        )
        batch_inputs = {
            k:v.to(device) for k,v in batch_inputs.items()
        }
        batch_inputs["oracle_answer"] = batch["oracle_answer"]
        max_input_length = batch_inputs["input_ids"].shape[1]

        # Rollout phase of GRPO
        with torch.no_grad():
            model.eval()
            # rollout
            sequence_ids, returns, solved_rate, action_mask, completions, info_list, num_words_in_completions = rollout(
                model,
                tokenizer,
                batch_inputs,
                reward_model=reward_model,
                generation_config=generation_config,
            )
            # sequence_ids: (num_samples * group_size, seq_len)
            # action_mask: (num_samples * group_size, seq_len)
            # completions: (num_samples * group_size)
            # returns: (num_samples, group_size)
            # solved_rate: (num_samples, )

            batch_mean_format_returns = np.array([x["format_reward"] for x in info_list]).mean()
            batch_mean_outcome_returns = np.array([x["outcome_reward"] for x in info_list]).mean()
            batch_mean_returns = returns.mean()
            batch_mean_solved_rate = solved_rate.mean()
            print(f"batch_idx: {batch_idx} | returns: {batch_mean_returns.item()} | solved_rate: {batch_mean_solved_rate.item()} | format_returns: {batch_mean_format_returns} | outcome_returns: {batch_mean_outcome_returns}")

            advantages = group_advantages(returns) # (num_samples, group_size)
            returns = returns.reshape(-1)
            advantages = advantages.reshape(-1)
            
            attention_mask = sequence_ids != pad_token_id # (num_samples * group_size, seq_len)

            log_probs = sequences_log_probs(
                model=model,
                sequence_ids=sequence_ids,
                attention_mask=attention_mask,
            ) # (num_samples * group_size, seq_len-1)
            log_probs_ref = sequences_log_probs(
                model=reference_model,
                sequence_ids=sequence_ids,
                attention_mask=attention_mask,
            ) # (num_samples * group_size, seq_len-1)
            kl = approx_kl_divergence(
                log_probs=log_probs,
                log_probs_ref=log_probs_ref,
                action_mask=action_mask,
            ) # (num_samples * group_size, seq_len-1)
            
            experience = Experience(
                sequences=sequence_ids,
                action_log_probs=log_probs,
                log_probs_ref=log_probs_ref,
                returns=returns,
                solved_rate = solved_rate,
                advantages=advantages,
                attention_mask=attention_mask,
                action_mask=action_mask,
                kl=kl,
            )
            replay_buffer.append(experience.to(cpu_device))
            
            del sequence_ids
            del attention_mask
            del action_mask
            del log_probs
            del log_probs_ref
            del advantages

        torch.cuda.empty_cache()

        # log the stats to wandb 
        logger(
            {
                "train/batch_returns": batch_mean_returns,
                "train/batch_solved_rate": batch_mean_solved_rate,
                "train/max_input_length": max_input_length,
                "train/num_words_in_completions": np.array(num_words_in_completions).mean(),
                "train/batch_mean_format_returns": batch_mean_format_returns,
                "train/batch_mean_outcome_returns": batch_mean_outcome_returns,
            }
        )
        out_dir = os.path.join(args.base_config.log_dir, args.wandb_config.name)
        os.makedirs(out_dir, exist_ok=True)

        delimeter = "*"*50
        file_name = os.path.join(out_dir, f"log_{batch_idx}.txt")
        with open(file_name, "a") as f:
            for i, completion in enumerate(completions):
                question = questions[i//args.grpo_config.group_size]
                answer = answers[i//args.grpo_config.group_size]
                reward = returns[i]
                info = info_list[i]                
                f.write(
                    f"{delimeter}\nQuestion:\n{question}\nAnswer:\n{answer}\n"
                    f"Completion:\n{completion}\nReward:\n{reward}\n"
                    f"Info:\n{info}\n{delimeter}\n"
                )
        f.close()
        
        ### Training phase of GRPO
        experience_sampler = DataLoader(
            replay_buffer,
            batch_size=args.grpo_config.mini_batch_size,
            shuffle=True,
            drop_last=False,
            collate_fn=join_experience_batch,
        )
        model.train()

        for _ in range(args.grpo_config.epochs_per_step):
            for exp in experience_sampler:
                exp: Experience
                exp = exp.to(device)

                optimizer.zero_grad()

                log_probs = sequences_log_probs(
                    model, 
                    sequence_ids=exp.sequences, 
                    attention_mask=exp.attention_mask
                )

                loss, mean_kl = objective(log_probs=log_probs, experience=exp)
                del log_probs

                if not loss.isfinite():
                    print(f"Loss not finite, skipping backward, loss={loss}")
                    print(f"experience.advantages={experience.advantages}")
                    continue
                
                loss: torch.Tensor
                loss.backward()

                grad_norm = clip_grad_norm_(
                    model.parameters(), 
                    max_norm=args.grpo_config.max_grad_norm,
                )
                #print(f"{step_epoch}: kl={kl: .4f}, grad_norm={grad_norm: .4f}")
                wandb.log(
                    {
                        "train/mean_kl": mean_kl, 
                        "train/grad_norm": grad_norm,
                        "train/loss": loss,
                    }
                )
                optimizer.step()
                del grad_norm
                del exp 
                del loss 
                del mean_kl
                torch.cuda.empty_cache()

        if (
            args.base_config.checkpoint_path is not None
            and args.base_config.checkpoint_interval is not None
            and (batch_idx + 1) % args.base_config.checkpoint_interval == 0
        ):
            model.save_pretrained(
                os.path.join(
                    args.base_config.checkpoint_path, 
                    f"step_{batch_idx + 1}"
                )
            )
        del batch_inputs

    if args.base_config.checkpoint_path is not None:
        model.save_pretrained(
            os.path.join(
                args.base_config.checkpoint_path, 
                f"step_{batch_idx + 1}"
            )
        )


if __name__ == "__main__":

    args = tyro.cli(TrainingConfig)
    
    wandb.init(
        project=args.wandb_config.project,
        name=args.wandb_config.name,
        config=args,
    )
    logger = wandb.log  
    
    train(args, logger)