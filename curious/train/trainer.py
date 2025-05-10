from typing import Dict, List, Any

from pathlib import Path
from curious.utils.utils import release_memory, LOGGING_TEMPLATE
from curious.replay.experience import Experience, ReplayBuffer, join_experience_batch
from curious.train.training_setup import TrainingSetup, TrainState
from curious.sampling.sampling import rollout, sequences_log_probs
from curious.policy_gradient.loss import masked_mean, approx_kl_divergence

from torch.utils.data import DataLoader
import torch

import numpy as np
from rich import print


class PolicyGradientTrainer:

    def __init__(self, trainining_setup: TrainingSetup) -> None:
        
        self.training_setup = trainining_setup
        self.num_epochs = self.training_setup["num_epochs"]
        self.rollout_data_loader = self.training_setup["rollout_data_loader"]

        self.model = self.training_setup["target_policy"]
        self.tokenizer = self.training_setup["tokenizer"]
        self.reward_model = self.training_setup["reward_model"]

        self.generation_config = self.training_setup["generation_config"]
        self.logger = lambda x: print(x)

    def train(self, train_state: TrainState) -> TrainState:
        for epoch_idx in range(self.num_epochs):
            print(f"Epoch {epoch_idx} of {self.num_epochs}")
            for batch_idx, batch_inputs in enumerate(self.rollout_data_loader):
                replay_buffer = self.collect_trajectories(train_state, batch_inputs, batch_idx)
                #train_state = self.update_policy(train_state, replay_buffer)
                print(len(replay_buffer))
        
                return train_state, replay_buffer
    
    @torch.no_grad()
    def collect_trajectories(self, train_state: TrainState, batch_inputs: Dict[str, torch.Tensor], batch_indx:int) -> ReplayBuffer:
        
        stats = {}
        stats["batch_idx"] = batch_indx
        stats["question"] = batch_inputs["question"]
        stats["answer"] = batch_inputs["answer"]
        stats["rl_scheduler"] = train_state["lr_scheduler"]
        
        replay_buffer = ReplayBuffer() 

        model = train_state["model"]
        model.eval()
        model.gradient_checkpointing_disable()

        rollout_out = rollout(
            model=model,
            tokenizer=self.tokenizer,
            batch_inputs=batch_inputs,
            reward_model=self.reward_model,
            generation_config=self.generation_config,
            group_size=self.generation_config.num_return_sequences,
            seed=train_state["seed"],
            normalize_centered_returns=self.training_setup["rl_config"].normalize_centered_returns,
            use_rloo_scalar=self.training_setup["rl_config"].use_rloo_scalar,
        )
    
        returns: torch.Tensor = rollout_out["returns"].reshape(-1)
        advantages: torch.Tensor = rollout_out["advantages"].reshape(-1)
        solved_mask: torch.Tensor = rollout_out["solved_masks"].reshape(-1)
        action_mask: torch.Tensor = rollout_out["action_mask"]
        
        stats["infos"] = rollout_out["infos"]
        stats["returns"] = returns
        stats["advantages"] = advantages
        stats["solved_masks"] = solved_mask
        stats["num_words_in_completions"] = rollout_out["num_words_in_completions"]

        sequence_ids: torch.Tensor = rollout_out["sequence_ids"]
        attention_mask: torch.Tensor = sequence_ids != self.tokenizer.pad_token_id 
        completions:List[str] = rollout_out["completions"]
        stats["completions"] = completions
        
        # (num_samples * group_size, seq_len-1)
        log_probs, entropy = sequences_log_probs(
            model=model,
            sequence_ids=sequence_ids,
            attention_mask=attention_mask,
            return_entropy=True,
            logits_minibatch_size=self.training_setup["rl_config"].logits_minibatch_size,
        ) 

        mean_action_entropy = masked_mean(entropy, action_mask, dim=None)
        stats["mean_action_entropy"] = mean_action_entropy.item()

        kl, log_probs_ref = None, None
        if self.training_setup["rl_config"].kl_weight > 0:
            # compute the log probs of the reference model
            log_probs_ref, _ = sequences_log_probs(
                model=train_state["reference_model"],
                sequence_ids=sequence_ids,
                attention_mask=attention_mask,
                return_entropy=False,
                logits_minibatch_size=self.training_setup["rl_config"].logits_minibatch_size,
            )
            # compute the kl divergence
            kl: torch.Tensor = approx_kl_divergence(
                log_probs=log_probs,
                log_probs_ref=log_probs_ref,
                action_mask=action_mask,
            )

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
        replay_buffer.append(
            experience.to(
                self.training_setup["cpu_device"]
            )
        )

        self.log_rollout_stats(stats)
        release_memory(
            [
                kl,
                log_probs_ref,
                log_probs,
                attention_mask,
                sequence_ids,
                action_mask,
                rollout_out,
            ]
        )
        return replay_buffer

    def update_policy(self, train_state: TrainState, replay_buffer: ReplayBuffer) -> TrainState:
        
        experience_sampler = DataLoader(
            replay_buffer,
            batch_size=self.training_setup["rl_config"].mini_batch_size,
            shuffle= False,
            drop_last=False,
            collate_fn=join_experience_batch,
            num_workers=self.training_setup["base_config"].num_workers,
        )
    def log_rollout_stats(self, rollout_stats: Dict[str, Any]) -> None:
        
        batch_idx = rollout_stats["batch_idx"]

        print(f"Batch indx {batch_idx}")
        print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/(1024**3)))
        print("torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(0)/(1024**3)))
        print("torch.cuda.max_memory_reserved: %fGB"%(torch.cuda.max_memory_reserved(0)/(1024**3)))

        info_list: List[Dict[str, float]] = rollout_stats["infos"]
        batch_mean_format_returns: float = np.array([x["format_reward"] for x in info_list]).mean().item()
        batch_mean_outcome_returns: float = np.array([x["outcome_reward"] for x in info_list]).mean().item()
        batch_mean_length_penalty: float = np.array([x["length_penalty"] for x in info_list]).mean().item()

        batch_mean_returns: float = rollout_stats["returns"].mean().item()
        batch_mean_solved_rate: float = rollout_stats["solved_masks"].mean().item()

        batch_mean_num_words_in_completions: float = rollout_stats["num_words_in_completions"].mean().item()
        batch_max_num_words_in_completions: float = rollout_stats["num_words_in_completions"].max().item()
        batch_min_num_words_in_completions: float = rollout_stats["num_words_in_completions"].min().item()
        
        self.logger(
            {
                "train/mean_batch_returns": batch_mean_returns,
                "train/mean_batch_solved_rate": batch_mean_solved_rate,

                "train/mean_num_words_in_completions": batch_mean_num_words_in_completions,
                "train/max_num_words_in_completions": batch_max_num_words_in_completions,
                "train/min_num_words_in_completions": batch_min_num_words_in_completions,

                "train/mean_batch_format_returns": batch_mean_format_returns,
                "train/mean_batch_outcome_returns": batch_mean_outcome_returns,
                "train/mean_batch_length_penalty": batch_mean_length_penalty,
                
                "train/lr": rollout_stats["rl_scheduler"].get_lr()[0],
                "train/mean_action_entropy": rollout_stats["mean_action_entropy"],
                
                "num_batches_visited": batch_idx,
            }
        )

        if self.training_setup["base_config"].train_text_log_interval > 0:
            if batch_idx % self.training_setup["base_config"].train_text_log_interval == 0:
                file_name = Path(self.training_setup["train_log_dir"]) / f"log_{batch_idx}.txt"
                with open(file_name, "a") as f:
                    for i, completion in enumerate(rollout_stats["completions"]):
                        question = rollout_stats["question"][i//self.generation_config.num_return_sequences]
                        answer = rollout_stats["answer"][i//self.generation_config.num_return_sequences]
                        reward = rollout_stats["returns"][i]
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
    
