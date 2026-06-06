from typing import Dict, List, Any, Tuple

from pathlib import Path
from curious.evaluate import evaluate
from curious.utils.utils import (
    build_completion_sample_rows,
    format_completion_sample_rows,
    log_completion_sample_table,
    release_memory,
)
from curious.replay.experience import Experience, ReplayBuffer, join_experience_batch
from curious.train.training_setup import TrainingSetup, TrainState
from curious.sampling.sampling import rollout, sequences_log_probs
from curious.policy_gradient.loss import masked_mean, approx_kl_divergence, ActorLoss
from curious.policy_gradient.ad_cispo import (
    ADCispoStats,
    ReferencePolicyFeatureRequest,
    compute_reference_policy_features,
)

from torch.utils.data import DataLoader
import torch

import numpy as np
from rich import print


class PolicyGradientTrainer:

    def __init__(self, trainining_setup: TrainingSetup) -> None:
        
        self.training_setup = trainining_setup
        self.rl_config = self.training_setup["rl_config"]
        self.base_config = self.training_setup["base_config"]
        self.generation_config = self.training_setup["generation_config"]

        self.num_epochs = self.training_setup["num_epochs"]
        self.rollout_data_loader = self.training_setup["rollout_data_loader"]

        self.model = self.training_setup["target_policy"]
        self.tokenizer = self.training_setup["tokenizer"]
        self.reward_model = self.training_setup["reward_model"]

        self.actor_loss: ActorLoss= self.training_setup["actor_loss"]
        self.logger = lambda x: print(x)

    def train(self, train_state: TrainState) -> Tuple[TrainState, ReplayBuffer]:
        evaluate(
            config=self.training_setup["eval_config"],
            model=train_state["model"],
            tokenizer=self.tokenizer,
            logger=self.logger,
            batch_idx=0,
        )

        for epoch_idx in range(self.num_epochs):
            print(f"Epoch {epoch_idx} of {self.num_epochs}")
            for batch_idx, batch_inputs in enumerate(self.rollout_data_loader):
                replay_buffer = self.collect_trajectories(train_state, batch_inputs, batch_idx)
                completed_batches = batch_idx + 1
                train_state = self.update_policy(train_state, replay_buffer, completed_batches)

                eval_interval = self.base_config.eval_interval
                if eval_interval > 0 and completed_batches % eval_interval == 0:
                    evaluate(
                        config=self.training_setup["eval_config"],
                        model=train_state["model"],
                        tokenizer=self.tokenizer,
                        logger=self.logger,
                        batch_idx=completed_batches,
                    )
        
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
            generation_backend=self.training_setup.get("generation_backend"),
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
        
        entropy_interval = self.base_config.train_entropy_log_interval
        should_log_entropy = entropy_interval > 0 and batch_indx % entropy_interval == 0

        # (num_samples * group_size, seq_len-1)
        log_probs, entropy = sequences_log_probs(
            model=model,
            sequence_ids=sequence_ids,
            attention_mask=attention_mask,
            return_entropy=should_log_entropy,
            logits_minibatch_size=self.training_setup["rl_config"].logits_minibatch_size,
        ) 

        if should_log_entropy and entropy is not None:
            mean_action_entropy = masked_mean(entropy, action_mask, dim=None)
            stats["mean_action_entropy"] = mean_action_entropy

        kl, log_probs_ref, token_clip_high = None, None, None
        if self.training_setup["rl_config"].use_ad_cispo:
            reference_features = compute_reference_policy_features(
                ReferencePolicyFeatureRequest(
                    model=train_state["reference_model"],
                    sequence_ids=sequence_ids,
                    attention_mask=attention_mask,
                    action_mask=action_mask,
                    clip_high=self.actor_loss.epsilon_high,
                    logits_minibatch_size=self.training_setup["rl_config"].logits_minibatch_size,
                    top_layers=self.training_setup["rl_config"].ad_cispo_top_layers,
                    min_multiplier=self.training_setup["rl_config"].ad_cispo_min_multiplier,
                    max_multiplier=self.training_setup["rl_config"].ad_cispo_max_multiplier,
                    eps=self.training_setup["rl_config"].ad_cispo_eps,
                )
            )
            token_clip_high = reference_features.token_clip_thresholds.values
            stats["ad_cispo_stats"] = reference_features.stats
            if self.training_setup["rl_config"].kl_weight > 0:
                log_probs_ref = reference_features.log_probs
        elif self.training_setup["rl_config"].kl_weight > 0:
            # compute the log probs of the reference model
            log_probs_ref, _ = sequences_log_probs(
                model=train_state["reference_model"],
                sequence_ids=sequence_ids,
                attention_mask=attention_mask,
                return_entropy=False,
                logits_minibatch_size=self.training_setup["rl_config"].logits_minibatch_size,
            )

        if self.training_setup["rl_config"].kl_weight > 0:
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
            token_clip_high=token_clip_high,
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
                token_clip_high,
                log_probs,
                entropy,
                attention_mask,
                sequence_ids,
                action_mask,
                rollout_out,
            ]
        )
        return replay_buffer

    def update_policy(self, train_state: TrainState, replay_buffer: ReplayBuffer, batch_idx: int) -> TrainState:
        
        model = train_state["model"]
        device = train_state["device"]
        optimizer = train_state["optimizer"]
        lr_scheduler = train_state["lr_scheduler"]
        kl_controller = train_state["kl_controller"]

        experience_sampler = DataLoader(
            replay_buffer,
            batch_size=self.rl_config.mini_batch_size,
            shuffle= False,
            drop_last=False,
            collate_fn=join_experience_batch,
            num_workers=0,
        )

        model.train()
        model.gradient_checkpointing_enable()
        if hasattr(model, "config"):
            model.config.use_cache = False

        did_update = False
        for _ in range(self.rl_config.epochs_per_step):
            for experience in experience_sampler:
                experience: Experience

                optimizer.zero_grad()
                experience = experience.to(device)
                log_probs, _ = sequences_log_probs(
                    model, 
                    sequence_ids=experience.sequences, 
                    attention_mask=experience.attention_mask,
                    return_entropy=False,
                    logits_minibatch_size=self.rl_config.logits_minibatch_size,
                )
                kl_weight_used = self.actor_loss.kl_weight if self.rl_config.kl_weight > 0 else 0.0
                loss, mean_kl, mean_actor_loss = self.actor_loss(
                    log_probs=log_probs, 
                    experience=experience
                )

                if not loss.isfinite():
                    print(f"Loss not finite, skipping backward, loss={loss}")
                    print(f"experience.advantages={experience.advantages}")
                    continue

                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 
                    max_norm=self.rl_config.max_grad_norm,
                )
                optimizer.step()
                did_update = True

                mean_loss = loss.detach().cpu().item()
                mean_kl = mean_kl.detach().cpu().item()
                mean_actor_loss = mean_actor_loss.detach().cpu().item()
                mean_kl_loss = mean_loss - mean_actor_loss

                kl_weight_next = kl_weight_used
                if self.rl_config.kl_weight > 0:
                    kl_controller.update(mean_kl, self.rl_config.mini_batch_size)
                    self.actor_loss.kl_weight = kl_controller.value
                    kl_weight_next = kl_controller.value

                self.logger(
                    {
                        "train/loss": mean_loss,
                        "train/mean_kl": mean_kl,
                        "train/actor_loss": mean_actor_loss,
                        "train/kl_loss": mean_kl_loss,
                        "train/kl_weight": kl_weight_used,
                        "train/kl_weight_used": kl_weight_used,
                        "train/kl_weight_next": kl_weight_next,
                        "train/grad_norm": grad_norm,
                        "num_batches_visited": batch_idx,
                    }
                )

                release_memory(
                    [
                        log_probs,
                        experience,
                        loss,
                        mean_kl,
                        mean_actor_loss,
                        mean_kl_loss,
                    ]
                )

        if self.rl_config.anneling_lr:
            lr_scheduler.step()

        if (
            train_state["reference_model"] is not None
            and self.rl_config.ref_model_update_freq > 0
            and batch_idx % self.rl_config.ref_model_update_freq == 0
        ):
            train_state["reference_model"].load_state_dict(model.state_dict())
            train_state["reference_model"].eval()

        generation_backend = self.training_setup.get("generation_backend")
        sync_interval = self.training_setup["sampling_config"].sglang_weight_sync_interval
        if did_update and generation_backend is not None and batch_idx % sync_interval == 0:
            generation_backend.sync_weights_from_model(model, self.tokenizer, batch_idx)

        train_state["model"] = model
        train_state["optimizer"] = optimizer
        train_state["lr_scheduler"] = lr_scheduler
        train_state["kl_controller"] = kl_controller

        return train_state

    def log_rollout_stats(self, rollout_stats: Dict[str, Any]) -> None:
        
        batch_idx = rollout_stats["batch_idx"]

        print(f"Batch indx {batch_idx}")
        print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/(1024**3)))
        print("torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(0)/(1024**3)))
        print("torch.cuda.max_memory_reserved: %fGB"%(torch.cuda.max_memory_reserved(0)/(1024**3)))

        returns: torch.Tensor = rollout_stats["returns"]

        info_list: List[Dict[str, float]] = rollout_stats["infos"]
        format_returns: np.ndarray = np.array([x["format_reward"] for x in info_list])
        outcome_returns: np.ndarray = np.array([x["outcome_reward"] for x in info_list])
        length_penalty: np.ndarray = np.array([x["length_penalty"] for x in info_list])

        batch_mean_solved_rate: float = rollout_stats["solved_masks"].mean().item()

        num_words_in_completions: torch.Tensor = rollout_stats["num_words_in_completions"]

        log_payload = {
            "train/mean_batch_returns": returns.mean().item(),
            "train/max_batch_returns": returns.max().item(),
            "train/min_batch_returns": returns.min().item(),

            "train/mean_batch_solved_rate": batch_mean_solved_rate,

            "train/mean_batch_format_returns": format_returns.mean().item(),
            "train/max_batch_format_returns": format_returns.max().item(),
            "train/min_batch_format_returns": format_returns.min().item(),

            "train/mean_batch_outcome_returns": outcome_returns.mean().item(),
            "train/max_batch_outcome_returns": outcome_returns.max().item(),
            "train/min_batch_outcome_returns": outcome_returns.min().item(),

            "train/mean_batch_length_penalty": length_penalty.mean().item(),
            "train/max_batch_length_penalty": length_penalty.max().item(),
            "train/min_batch_length_penalty": length_penalty.min().item(),

            "train/mean_num_words_in_completions": num_words_in_completions.mean().item(),
            "train/max_num_words_in_completions": num_words_in_completions.max().item(),
            "train/min_num_words_in_completions": num_words_in_completions.min().item(),

            "train/lr": rollout_stats["rl_scheduler"].get_lr()[0],
            "num_batches_visited": batch_idx,
        }
        mean_action_entropy = rollout_stats.get("mean_action_entropy")
        if mean_action_entropy is not None:
            log_payload["train/mean_action_entropy"] = mean_action_entropy.item()

        self.logger(log_payload)
        ad_cispo_stats: ADCispoStats | None = rollout_stats.get("ad_cispo_stats")
        if ad_cispo_stats is not None:
            self.logger(
                {
                    "ad_cispo/clip_mean": ad_cispo_stats.clip_mean,
                    "ad_cispo/clip_min": ad_cispo_stats.clip_min,
                    "ad_cispo/clip_max": ad_cispo_stats.clip_max,
                    "ad_cispo/multiplier_mean": ad_cispo_stats.multiplier_mean,
                    "ad_cispo/saliency_mean": ad_cispo_stats.saliency_mean,
                    "num_batches_visited": batch_idx,
                }
            )

        text_log_interval = self.training_setup["base_config"].train_text_log_interval
        if text_log_interval > 0 and batch_idx % text_log_interval == 0:
            repeated_questions = [
                rollout_stats["question"][i // self.generation_config.num_return_sequences]
                for i in range(len(rollout_stats["completions"]))
            ]
            repeated_answers = [
                rollout_stats["answer"][i // self.generation_config.num_return_sequences]
                for i in range(len(rollout_stats["completions"]))
            ]
            sample_rows = build_completion_sample_rows(
                phase="train",
                batch_idx=batch_idx,
                questions=repeated_questions,
                answers=repeated_answers,
                completions=rollout_stats["completions"],
                rewards=rollout_stats["returns"].reshape(-1).tolist(),
                infos=info_list,
                max_samples=self.training_setup["base_config"].completion_log_sample_size,
            )
            file_name = Path(self.training_setup["train_log_dir"]) / f"log_{batch_idx}.txt"
            with open(file_name, "a") as f:
                f.write(format_completion_sample_rows(sample_rows))
            log_completion_sample_table(
                logger=self.logger,
                key="train/completion_samples",
                rows=sample_rows,
            )
    
