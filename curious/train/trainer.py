from contextlib import contextmanager
import time
from typing import Dict, List, Any, Tuple, Iterator

from pathlib import Path
from curious.evaluate import evaluate
from curious.utils.utils import (
    CleanupPolicy,
    PerfTrace,
    build_completion_sample_rows,
    cuda_memory_snapshot,
    decide_cleanup,
    execute_cleanup,
    format_completion_sample_rows,
    log_completion_sample_table,
)
from curious.replay.experience import Experience, ReplayBuffer, slice_experience_batch
from curious.train.training_setup import TrainingSetup, TrainState
from curious.sampling.sampling import rollout, sequences_log_probs
from curious.policy_gradient.loss import masked_mean, approx_kl_divergence, ActorLoss
from curious.policy_gradient.ad_cispo import (
    ADCispoStats,
    ReferencePolicyFeatures,
    ReferencePolicyFeatureRequest,
    collect_ad_cispo_sink_token_ids,
    compute_reference_policy_features,
)

import torch

import numpy as np
from rich import print


AD_CISPO_TOP_TOKEN_COLUMNS = [
    "batch_idx",
    "rank",
    "sequence_idx",
    "question_idx",
    "rollout_idx",
    "token_position",
    "token_id",
    "token",
    "token_context",
    "saliency",
    "clip_high",
    "multiplier",
    "return",
    "solved",
    "advantage",
    "question",
    "answer",
    "completion_excerpt",
]

AD_CISPO_DISTRIBUTION_QUANTILES = (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)
AD_CISPO_HIGH_SALIENCY_QUANTILE = 0.90


def _quantile_suffix(quantile: float) -> str:
    return f"p{int(round(quantile * 100)):02d}"


def _finite_flattened(values: torch.Tensor) -> torch.Tensor:
    flattened = values.detach().reshape(-1).to(dtype=torch.float32)
    return flattened[torch.isfinite(flattened)]


def _summarize_distribution(
    prefix: str,
    values: torch.Tensor,
    *,
    quantiles: Tuple[float, ...] = AD_CISPO_DISTRIBUTION_QUANTILES,
) -> Dict[str, float]:
    finite_values = _finite_flattened(values)
    payload: Dict[str, float] = {
        f"{prefix}_count": float(finite_values.numel()),
    }
    if finite_values.numel() == 0:
        return payload

    payload.update(
        {
            f"{prefix}_mean": finite_values.mean().item(),
            f"{prefix}_std": finite_values.std(unbiased=False).item(),
            f"{prefix}_min": finite_values.min().item(),
            f"{prefix}_max": finite_values.max().item(),
        }
    )
    for quantile in quantiles:
        payload[f"{prefix}_{_quantile_suffix(quantile)}"] = torch.quantile(
            finite_values,
            quantile,
        ).item()
    return payload


def _select_high_saliency_tokens(saliency_values: torch.Tensor) -> Tuple[torch.Tensor, float]:
    finite_saliency = _finite_flattened(saliency_values)
    if finite_saliency.numel() == 0:
        return torch.zeros_like(saliency_values, dtype=torch.bool), 0.0
    threshold = torch.quantile(finite_saliency, AD_CISPO_HIGH_SALIENCY_QUANTILE)
    return saliency_values.to(dtype=torch.float32) >= threshold, threshold.item()


def build_ad_cispo_distribution_metrics(reference_features: ReferencePolicyFeatures) -> Dict[str, float]:
    action_saliency = reference_features.action_saliency
    token_clip_thresholds = reference_features.token_clip_thresholds
    action_mask = action_saliency.action_mask.bool()
    active_count = int(action_mask.sum().item())
    payload: Dict[str, float] = {
        "ad_cispo/active_token_count": float(active_count),
    }
    payload.update(reference_features.diagnostics)
    if active_count == 0:
        return payload

    clip_values = token_clip_thresholds.values.detach()[action_mask].to(dtype=torch.float32)
    clip_bounds = 1.0 + clip_values
    saliency_values = action_saliency.values.detach()[action_mask].to(dtype=torch.float32)
    multipliers = token_clip_thresholds.multipliers.detach()[action_mask].to(dtype=torch.float32)

    payload.update(_summarize_distribution("ad_cispo/clip", clip_values))
    payload.update(_summarize_distribution("ad_cispo/clip_bound", clip_bounds))
    payload.update(_summarize_distribution("ad_cispo/saliency", saliency_values))
    payload.update(_summarize_distribution("ad_cispo/multiplier", multipliers))

    high_saliency_mask, threshold = _select_high_saliency_tokens(saliency_values)
    payload["ad_cispo/high_saliency_threshold_p90"] = threshold
    payload["ad_cispo/high_saliency_token_count"] = float(high_saliency_mask.sum().item())
    payload["ad_cispo/high_saliency_token_fraction"] = high_saliency_mask.to(dtype=torch.float32).mean().item()
    if high_saliency_mask.any():
        payload.update(_summarize_distribution("ad_cispo/high_saliency/clip", clip_values[high_saliency_mask]))
        payload.update(_summarize_distribution("ad_cispo/high_saliency/clip_bound", clip_bounds[high_saliency_mask]))
        payload.update(_summarize_distribution("ad_cispo/high_saliency/saliency", saliency_values[high_saliency_mask]))
        payload.update(_summarize_distribution("ad_cispo/high_saliency/multiplier", multipliers[high_saliency_mask]))
    for existing_key in (
        "ad_cispo/clip_mean",
        "ad_cispo/clip_min",
        "ad_cispo/clip_max",
        "ad_cispo/clip_std",
        "ad_cispo/clip_p10",
        "ad_cispo/clip_p50",
        "ad_cispo/clip_p90",
        "ad_cispo/saliency_mean",
        "ad_cispo/saliency_std",
        "ad_cispo/multiplier_mean",
        "ad_cispo/multiplier_min",
        "ad_cispo/multiplier_max",
        "ad_cispo/multiplier_std",
    ):
        payload.pop(existing_key, None)
    return payload


def _decode_token_text(tokenizer: Any, token_id: int) -> str:
    try:
        return tokenizer.decode([token_id], skip_special_tokens=False)
    except Exception:
        return str(token_id)


def _decode_token_context(
    tokenizer: Any,
    sequence_ids: torch.Tensor,
    token_position: int,
    window_size: int = 5,
) -> str:
    begin = max(0, token_position - window_size)
    end = min(sequence_ids.numel(), token_position + window_size + 1)
    token_ids = sequence_ids[begin:end].detach().cpu().tolist()
    try:
        return tokenizer.decode(token_ids, skip_special_tokens=False)
    except Exception:
        return " ".join(str(token_id) for token_id in token_ids)


def build_ad_cispo_top_token_rows(
    *,
    tokenizer: Any,
    reference_features: ReferencePolicyFeatures,
    sequence_ids: torch.Tensor,
    questions: List[str],
    answers: List[str],
    completions: List[str],
    returns: torch.Tensor,
    solved_mask: torch.Tensor,
    advantages: torch.Tensor,
    group_size: int,
    batch_idx: int,
    top_k: int = 10,
) -> List[Dict[str, Any]]:
    action_saliency = reference_features.action_saliency
    token_clip_thresholds = reference_features.token_clip_thresholds
    mask = action_saliency.action_mask.bool()
    active_positions = mask.nonzero(as_tuple=False)
    if active_positions.numel() == 0:
        return []

    active_saliency = action_saliency.values.detach()[mask].to(dtype=torch.float32)
    k = min(top_k, active_saliency.numel())
    top_values, top_indices = torch.topk(active_saliency, k=k)
    positions = active_positions.detach().cpu()[top_indices.detach().cpu()]

    flat_returns = returns.detach().cpu().reshape(-1)
    flat_solved = solved_mask.detach().cpu().reshape(-1)
    flat_advantages = advantages.detach().cpu().reshape(-1)
    clip_values = token_clip_thresholds.values.detach().cpu()
    multipliers = token_clip_thresholds.multipliers.detach().cpu()
    sequence_ids_cpu = sequence_ids.detach().cpu()

    rows: List[Dict[str, Any]] = []
    for rank, ((sequence_idx_tensor, action_idx_tensor), saliency_tensor) in enumerate(
        zip(positions.tolist(), top_values.detach().cpu().tolist()),
        start=1,
    ):
        sequence_idx = int(sequence_idx_tensor)
        action_idx = int(action_idx_tensor)
        token_position = action_idx + 1
        token_id = int(sequence_ids_cpu[sequence_idx, token_position].item())
        question_idx = sequence_idx // group_size
        rollout_idx = sequence_idx % group_size
        rows.append(
            {
                "batch_idx": batch_idx,
                "rank": rank,
                "sequence_idx": sequence_idx,
                "question_idx": question_idx,
                "rollout_idx": rollout_idx,
                "token_position": token_position,
                "token_id": token_id,
                "token": _decode_token_text(tokenizer, token_id),
                "token_context": _decode_token_context(tokenizer, sequence_ids_cpu[sequence_idx], token_position),
                "saliency": float(saliency_tensor),
                "clip_high": float(clip_values[sequence_idx, action_idx].item()),
                "multiplier": float(multipliers[sequence_idx, action_idx].item()),
                "return": float(flat_returns[sequence_idx].item()),
                "solved": float(flat_solved[sequence_idx].item()),
                "advantage": float(flat_advantages[sequence_idx].item()),
                "question": questions[question_idx] if question_idx < len(questions) else "",
                "answer": answers[question_idx] if question_idx < len(answers) else "",
                "completion_excerpt": completions[sequence_idx][:500] if sequence_idx < len(completions) else "",
            }
        )
    return rows


def log_ad_cispo_top_token_table(
    *,
    logger: Any,
    rows: List[Dict[str, Any]],
) -> None:
    if not rows:
        return

    data = [[row[column] for column in AD_CISPO_TOP_TOKEN_COLUMNS] for row in rows]
    try:
        import wandb

        value: Any = wandb.Table(columns=AD_CISPO_TOP_TOKEN_COLUMNS, data=data)
    except Exception:
        value = data

    logger(
        {
            "ad_cispo/top_salient_tokens": value,
            "num_batches_visited": rows[0]["batch_idx"],
        }
    )


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
        self.optimizer_step_count = 0

    def _cleanup_policy(self) -> CleanupPolicy:
        return CleanupPolicy(
            batch_interval=getattr(self.base_config, "memory_cleanup_batch_interval", 1),
            minibatch_empty_cache_interval=getattr(
                self.base_config,
                "memory_cleanup_minibatch_empty_cache_interval",
                0,
            ),
            reserved_ratio_threshold=getattr(
                self.base_config,
                "memory_cleanup_reserved_ratio_threshold",
                0.90,
            ),
            fragmentation_ratio=getattr(
                self.base_config,
                "memory_cleanup_fragmentation_ratio",
                1.35,
            ),
            force_after_eval=getattr(self.base_config, "memory_cleanup_force_after_eval", True),
            force_after_checkpoint=getattr(self.base_config, "memory_cleanup_force_after_checkpoint", True),
        )

    def _should_log_perf(self, batch_idx: int) -> bool:
        interval = getattr(self.base_config, "perf_log_interval", 0)
        return interval > 0 and batch_idx % interval == 0

    @staticmethod
    def _bytes_to_gb(value: int) -> float:
        return value / (1024 ** 3)

    def _log_perf_trace(self, trace: PerfTrace, batch_idx: int) -> None:
        self.logger(
            {
                f"perf/{trace.phase}_seconds": trace.elapsed_seconds,
                f"perf/{trace.phase}_allocated_gb_before": self._bytes_to_gb(trace.memory_before.allocated_bytes),
                f"perf/{trace.phase}_allocated_gb_after": self._bytes_to_gb(trace.memory_after.allocated_bytes),
                f"perf/{trace.phase}_reserved_gb_before": self._bytes_to_gb(trace.memory_before.reserved_bytes),
                f"perf/{trace.phase}_reserved_gb_after": self._bytes_to_gb(trace.memory_after.reserved_bytes),
                "num_batches_visited": batch_idx,
            }
        )

    def _update_diagnostic_tensors(
        self,
        log_probs: torch.Tensor,
        experience: Experience,
    ) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            action_mask = experience.action_mask.bool()
            advantages = experience.advantages
            if advantages.ndim == 1:
                advantages = advantages.unsqueeze(1)
            advantages = advantages.to(dtype=torch.float32)
            returns = experience.returns.to(dtype=torch.float32)
            solved_mask = experience.solved_mask.to(dtype=torch.float32)

            active_log_probs = log_probs.detach()[action_mask].to(dtype=torch.float32)
            old_log_probs = experience.action_log_probs.detach()[action_mask].to(dtype=torch.float32)
            ratio = (active_log_probs - old_log_probs).exp()

            if experience.token_clip_high is None:
                clip_high = torch.full_like(ratio, 1.0 + self.actor_loss.epsilon_high)
                token_clip_high = torch.full_like(ratio, self.actor_loss.epsilon_high)
            else:
                token_clip_high = experience.token_clip_high.detach()[action_mask].to(
                    device=ratio.device,
                    dtype=ratio.dtype,
                )
                clip_high = 1.0 + token_clip_high
            if self.actor_loss.use_cispo_loss:
                clip_low = torch.zeros_like(ratio)
            else:
                clip_low = torch.full_like(ratio, 1.0 - self.actor_loss.epsilon_low)

            action_tokens_per_sequence = action_mask.sum(dim=-1).to(dtype=torch.float32)
            nonzero_advantages = advantages.abs() > 1e-8
            if ratio.numel() == 0:
                ratio = torch.ones(1, dtype=torch.float32, device=log_probs.device)
                clip_high = torch.ones_like(ratio)
                clip_low = torch.zeros_like(ratio)
                token_clip_high = torch.zeros_like(ratio)

            tensors: Dict[str, torch.Tensor] = {
                "advantages": advantages.detach().reshape(-1).cpu(),
                "returns": returns.detach().reshape(-1).cpu(),
                "solved_mask": solved_mask.detach().reshape(-1).cpu(),
                "action_tokens_per_sequence": action_tokens_per_sequence.detach().cpu(),
                "nonzero_advantages": nonzero_advantages.detach().reshape(-1).to(dtype=torch.float32).cpu(),
                "ratio": ratio.detach().cpu(),
                "clip_high": clip_high.detach().cpu(),
                "clip_low": clip_low.detach().cpu(),
                "token_clip_high": token_clip_high.detach().cpu(),
            }
            if experience.token_saliency is not None:
                token_saliency = experience.token_saliency.detach()[action_mask].to(
                    device=ratio.device,
                    dtype=ratio.dtype,
                )
                tensors["saliency"] = token_saliency.detach().cpu()
            if experience.token_clip_multiplier is not None:
                token_clip_multiplier = experience.token_clip_multiplier.detach()[action_mask].to(
                    device=ratio.device,
                    dtype=ratio.dtype,
                )
                tensors["multiplier"] = token_clip_multiplier.detach().cpu()
            return tensors

    @staticmethod
    def _concat_diagnostic_tensors(
        tensor_parts: List[Dict[str, torch.Tensor]],
        key: str,
    ) -> torch.Tensor:
        values = [
            part[key].detach().reshape(-1).to(dtype=torch.float32)
            for part in tensor_parts
            if key in part and part[key] is not None and part[key].numel() > 0
        ]
        if not values:
            return torch.empty(0, dtype=torch.float32)
        return torch.cat(values, dim=0)

    def _summarize_update_diagnostics(
        self,
        tensor_parts: List[Dict[str, torch.Tensor]],
    ) -> Dict[str, float]:
        ratio = self._concat_diagnostic_tensors(tensor_parts, "ratio")
        clip_high = self._concat_diagnostic_tensors(tensor_parts, "clip_high")
        clip_low = self._concat_diagnostic_tensors(tensor_parts, "clip_low")
        token_clip_high = self._concat_diagnostic_tensors(tensor_parts, "token_clip_high")
        advantages = self._concat_diagnostic_tensors(tensor_parts, "advantages")
        returns = self._concat_diagnostic_tensors(tensor_parts, "returns")
        solved_mask = self._concat_diagnostic_tensors(tensor_parts, "solved_mask")
        action_tokens_per_sequence = self._concat_diagnostic_tensors(tensor_parts, "action_tokens_per_sequence")
        nonzero_advantages = self._concat_diagnostic_tensors(tensor_parts, "nonzero_advantages")
        saliency = self._concat_diagnostic_tensors(tensor_parts, "saliency")
        multiplier = self._concat_diagnostic_tensors(tensor_parts, "multiplier")

        if ratio.numel() == 0:
            ratio = torch.ones(1, dtype=torch.float32)
            clip_high = torch.ones_like(ratio)
            clip_low = torch.zeros_like(ratio)
            token_clip_high = torch.zeros_like(ratio)

        clipped_mask = ratio > clip_high
        below_clip_mask = ratio < clip_low
        clip_utilization = ratio / clip_high.clamp_min(1e-12)
        clip_margin = ratio - clip_high

        payload: Dict[str, float] = {
            "train/update_active_token_count": float(ratio.numel()),
            "train/update_policy_ratio_mean": ratio.mean().item(),
            "train/update_policy_ratio_std": ratio.std(unbiased=False).item(),
            "train/update_policy_ratio_min": ratio.min().item(),
            "train/update_policy_ratio_max": ratio.max().item(),
            "train/update_clip_high_mean": clip_high.mean().item(),
            "train/update_clip_high_min": clip_high.min().item(),
            "train/update_clip_high_max": clip_high.max().item(),
            "train/update_token_clip_high_mean": token_clip_high.mean().item(),
            "train/update_ratio_above_clip_fraction": clipped_mask.to(dtype=torch.float32).mean().item(),
            "train/update_ratio_below_clip_fraction": below_clip_mask.to(dtype=torch.float32).mean().item(),
        }
        if advantages.numel() > 0:
            payload.update(
                {
                    "train/update_advantage_mean": advantages.mean().item(),
                    "train/update_advantage_abs_mean": advantages.abs().mean().item(),
                    "train/update_advantage_std": advantages.std(unbiased=False).item(),
                }
            )
        if nonzero_advantages.numel() > 0:
            payload["train/update_advantage_nonzero_fraction"] = nonzero_advantages.mean().item()
        if returns.numel() > 0:
            payload["train/update_return_std"] = returns.std(unbiased=False).item()
        if solved_mask.numel() > 0:
            payload["train/update_solved_rate"] = solved_mask.mean().item()
        if action_tokens_per_sequence.numel() > 0:
            payload["train/update_action_tokens_mean"] = action_tokens_per_sequence.mean().item()

        payload.update(_summarize_distribution("train/update_policy_ratio", ratio))
        payload.update(_summarize_distribution("train/update_clip_high", clip_high))
        payload.update(_summarize_distribution("train/update_token_clip_high", token_clip_high))
        payload.update(_summarize_distribution("train/update_clip_utilization", clip_utilization))
        payload.update(_summarize_distribution("train/update_clip_margin", clip_margin))
        if saliency.numel() > 0:
            payload.update(_summarize_distribution("train/update_saliency", saliency))
        if multiplier.numel() > 0:
            payload.update(_summarize_distribution("train/update_multiplier", multiplier))

        def add_subset_metrics(name: str, subset_mask: torch.Tensor) -> None:
            payload[f"train/update_{name}_token_count"] = float(subset_mask.sum().item())
            payload[f"train/update_{name}_token_fraction"] = subset_mask.to(dtype=torch.float32).mean().item()
            if not subset_mask.any():
                return
            payload.update(_summarize_distribution(f"train/update_{name}/policy_ratio", ratio[subset_mask]))
            payload.update(_summarize_distribution(f"train/update_{name}/clip_high", clip_high[subset_mask]))
            payload.update(_summarize_distribution(f"train/update_{name}/token_clip_high", token_clip_high[subset_mask]))
            payload.update(_summarize_distribution(f"train/update_{name}/clip_utilization", clip_utilization[subset_mask]))
            payload.update(_summarize_distribution(f"train/update_{name}/clip_margin", clip_margin[subset_mask]))
            if saliency.numel() == ratio.numel():
                payload.update(_summarize_distribution(f"train/update_{name}/saliency", saliency[subset_mask]))
            if multiplier.numel() == ratio.numel():
                payload.update(_summarize_distribution(f"train/update_{name}/multiplier", multiplier[subset_mask]))

        add_subset_metrics("clipped", clipped_mask)
        if saliency.numel() == ratio.numel():
            high_saliency_mask, high_saliency_threshold = _select_high_saliency_tokens(saliency)
            payload["train/update_high_saliency_threshold_p90"] = high_saliency_threshold
            payload["train/update_high_saliency_ratio_above_clip_fraction"] = (
                clipped_mask[high_saliency_mask].to(dtype=torch.float32).mean().item()
                if high_saliency_mask.any()
                else 0.0
            )
            add_subset_metrics("high_saliency", high_saliency_mask)
            payload["train/update_clipped_high_saliency_fraction"] = (
                high_saliency_mask[clipped_mask].to(dtype=torch.float32).mean().item()
                if clipped_mask.any()
                else 0.0
            )
            payload["train/update_high_saliency_clipped_fraction"] = (
                clipped_mask[high_saliency_mask].to(dtype=torch.float32).mean().item()
                if high_saliency_mask.any()
                else 0.0
            )
        return payload

    @staticmethod
    def _merge_weighted_diagnostics(
        accumulated: Dict[str, float],
        metrics: Dict[str, float],
        weight: float,
    ) -> None:
        for key, value in metrics.items():
            if key.endswith("_min"):
                accumulated[key] = min(accumulated.get(key, value), value)
            elif key.endswith("_max"):
                accumulated[key] = max(accumulated.get(key, value), value)
            else:
                accumulated[key] = accumulated.get(key, 0.0) + value * weight

    @contextmanager
    def _trace_phase(
        self,
        phase: str,
        batch_idx: int,
        device: torch.device | None,
    ) -> Iterator[None]:
        if not self._should_log_perf(batch_idx):
            yield
            return

        should_sync_cuda = torch.cuda.is_available() and (
            device is None or torch.device(device).type == "cuda"
        )
        if should_sync_cuda:
            torch.cuda.synchronize(device)
        before = cuda_memory_snapshot(device)
        start = time.perf_counter()
        try:
            yield
        finally:
            if should_sync_cuda:
                torch.cuda.synchronize(device)
            after = cuda_memory_snapshot(device)
            self._log_perf_trace(
                PerfTrace(
                    phase=phase,
                    elapsed_seconds=time.perf_counter() - start,
                    memory_before=before,
                    memory_after=after,
                ),
                batch_idx=batch_idx,
            )

    def _cleanup(
        self,
        phase: str,
        batch_idx: int,
        device: torch.device | None,
        minibatch_idx: int = 0,
        force: bool = False,
    ) -> None:
        before = cuda_memory_snapshot(device)
        decision = decide_cleanup(
            before,
            self._cleanup_policy(),
            phase=phase,
            batch_idx=batch_idx,
            minibatch_idx=minibatch_idx,
            force=force,
        )
        start = time.perf_counter()
        execute_cleanup(decision)
        elapsed = time.perf_counter() - start
        if self._should_log_perf(batch_idx):
            after = cuda_memory_snapshot(device)
            self.logger(
                {
                    f"perf/cleanup_{phase}_seconds": elapsed,
                    f"memory/{phase}_run_gc": float(decision.run_gc),
                    f"memory/{phase}_empty_cache": float(decision.empty_cache),
                    f"memory/{phase}_reserved_gb_before": self._bytes_to_gb(before.reserved_bytes),
                    f"memory/{phase}_reserved_gb_after": self._bytes_to_gb(after.reserved_bytes),
                    f"memory/{phase}_allocated_gb_after": self._bytes_to_gb(after.allocated_bytes),
                    "num_batches_visited": batch_idx,
                }
            )

    def _iter_backward_microbatches(self, experience: Experience) -> Iterator[tuple[Experience, float]]:
        micro_batch_size = getattr(self.rl_config, "backward_micro_batch_size", 0)
        batch_size = experience.sequences.size(0)
        if micro_batch_size <= 0 or micro_batch_size >= batch_size:
            yield experience, 1.0
            return

        if self.actor_loss.aggregation_dim is None:
            total_weight = float(experience.action_mask.sum().item())
            weight_for = lambda item: float(item.action_mask.sum().item())
        else:
            total_weight = float(batch_size)
            weight_for = lambda item: float(item.sequences.size(0))

        if total_weight <= 0:
            raise ValueError("Cannot update from a minibatch with no weighted training tokens.")

        for start in range(0, batch_size, micro_batch_size):
            end = min(batch_size, start + micro_batch_size)
            micro_experience = slice_experience_batch(experience, start, end)
            yield micro_experience, weight_for(micro_experience) / total_weight

    def train(self, train_state: TrainState) -> Tuple[TrainState, ReplayBuffer]:
        evaluate(
            config=self.training_setup["eval_config"],
            model=train_state["model"],
            tokenizer=self.tokenizer,
            logger=self.logger,
            batch_idx=0,
        )
        self._cleanup("after_eval", batch_idx=0, device=train_state["device"])

        completed_batches = 0
        stop_training = False
        replay_buffer = ReplayBuffer()
        for epoch_idx in range(self.num_epochs):
            print(f"Epoch {epoch_idx} of {self.num_epochs}")
            for _, batch_inputs in enumerate(self.rollout_data_loader):
                next_batch_idx = completed_batches + 1
                if (
                    self.base_config.max_train_batches > 0
                    and next_batch_idx > self.base_config.max_train_batches
                ):
                    print(f"Reached max_train_batches={self.base_config.max_train_batches}; stopping training.")
                    stop_training = True
                    break

                replay_buffer = self.collect_trajectories(train_state, batch_inputs, next_batch_idx)
                completed_batches = next_batch_idx
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
                    self._cleanup("after_eval", batch_idx=completed_batches, device=train_state["device"])
            if stop_training:
                break
        
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

        with self._trace_phase("rollout_generation", batch_indx, train_state["device"]):
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
        with self._trace_phase("collect_policy_logprobs", batch_indx, train_state["device"]):
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
        token_saliency, token_clip_multiplier = None, None
        ad_cispo_features = None
        with self._trace_phase("ad_cispo_saliency_features", batch_indx, train_state["device"]):
            if self.training_setup["rl_config"].use_ad_cispo:
                ad_cispo_features = compute_reference_policy_features(
                    ReferencePolicyFeatureRequest(
                        model=train_state["model"],
                        sequence_ids=sequence_ids,
                        attention_mask=attention_mask,
                        action_mask=action_mask,
                        clip_high=self.actor_loss.epsilon_high,
                        logits_minibatch_size=self.training_setup["rl_config"].logits_minibatch_size,
                        top_layers=self.training_setup["rl_config"].ad_cispo_top_layers,
                        min_multiplier=self.training_setup["rl_config"].ad_cispo_min_multiplier,
                        max_multiplier=self.training_setup["rl_config"].ad_cispo_max_multiplier,
                        eps=self.training_setup["rl_config"].ad_cispo_eps,
                        return_log_probs=False,
                        saliency_method=self.training_setup["rl_config"].ad_cispo_saliency_method,
                        attention_block_size=self.training_setup["rl_config"].ad_cispo_attention_block_size,
                        sink_token_ids=collect_ad_cispo_sink_token_ids(self.tokenizer),
                        advantages=advantages,
                    )
                )
                token_clip_high = ad_cispo_features.token_clip_thresholds.values
                token_saliency = ad_cispo_features.action_saliency.values
                token_clip_multiplier = ad_cispo_features.token_clip_thresholds.multipliers
                stats["ad_cispo_stats"] = ad_cispo_features.stats
                stats["ad_cispo_distribution_metrics"] = build_ad_cispo_distribution_metrics(ad_cispo_features)
                stats["ad_cispo_distribution_metrics"]["ad_cispo/saliency_source_target_policy"] = 1.0
                stats["ad_cispo_distribution_metrics"]["ad_cispo/saliency_source_reference_policy"] = 0.0
                stats["ad_cispo_top_token_rows"] = build_ad_cispo_top_token_rows(
                    tokenizer=self.tokenizer,
                    reference_features=ad_cispo_features,
                    sequence_ids=sequence_ids,
                    questions=stats["question"],
                    answers=stats["answer"],
                    completions=completions,
                    returns=returns,
                    solved_mask=solved_mask,
                    advantages=advantages,
                    group_size=self.generation_config.num_return_sequences,
                    batch_idx=batch_indx,
                    top_k=10,
                )

            if self.training_setup["rl_config"].kl_weight > 0:
                # compute the log probs of the reference model
                if train_state["reference_model"] is None:
                    raise RuntimeError("KL regularization requires a reference model.")
                log_probs_ref, _ = sequences_log_probs(
                    model=train_state["reference_model"],
                    sequence_ids=sequence_ids,
                    attention_mask=attention_mask,
                    return_entropy=False,
                    logits_minibatch_size=self.training_setup["rl_config"].logits_minibatch_size,
                )

            if self.training_setup["rl_config"].kl_weight > 0:
                # compute the kl divergence
                kl = approx_kl_divergence(
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
            token_saliency=token_saliency,
            token_clip_multiplier=token_clip_multiplier,
        )
        replay_buffer.append_batch(experience)

        self.log_rollout_stats(stats)
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
            ad_cispo_features,
            experience,
            stats,
        )
        self._cleanup("after_rollout_batch", batch_idx=batch_indx, device=train_state["device"])
        return replay_buffer

    def update_policy(self, train_state: TrainState, replay_buffer: ReplayBuffer, batch_idx: int) -> TrainState:
        
        model = train_state["model"]
        device = train_state["device"]
        optimizer = train_state["optimizer"]
        lr_scheduler = train_state["lr_scheduler"]
        kl_controller = train_state["kl_controller"]

        model.train()
        model.gradient_checkpointing_enable()
        if hasattr(model, "config"):
            model.config.use_cache = False

        did_update = False
        for _ in range(self.rl_config.epochs_per_step):
            for minibatch_idx, training_minibatch in enumerate(
                replay_buffer.iter_minibatches(self.rl_config.mini_batch_size),
                start=1,
            ):
                experience: Experience = training_minibatch.experience

                optimizer.zero_grad(set_to_none=True)
                experience = experience.to(device)
                kl_weight_used = self.actor_loss.kl_weight if self.rl_config.kl_weight > 0 else 0.0
                mean_loss = 0.0
                mean_kl = 0.0
                mean_actor_loss = 0.0
                update_diagnostic_tensors: List[Dict[str, torch.Tensor]] = []
                skip_minibatch = False

                for micro_experience, micro_weight in self._iter_backward_microbatches(experience):
                    with self._trace_phase("update_forward_loss", batch_idx, device):
                        log_probs, _ = sequences_log_probs(
                            model,
                            sequence_ids=micro_experience.sequences,
                            attention_mask=micro_experience.attention_mask,
                            return_entropy=False,
                            logits_minibatch_size=self.rl_config.logits_minibatch_size,
                        )
                        loss, micro_mean_kl, micro_mean_actor_loss = self.actor_loss(
                            log_probs=log_probs,
                            experience=micro_experience,
                        )
                        micro_diagnostic_tensors = self._update_diagnostic_tensors(log_probs, micro_experience)

                    if not loss.isfinite():
                        print(f"Loss not finite, skipping backward, loss={loss}")
                        print(f"experience.advantages={micro_experience.advantages}")
                        del log_probs, loss, micro_mean_kl, micro_mean_actor_loss, micro_diagnostic_tensors
                        optimizer.zero_grad(set_to_none=True)
                        skip_minibatch = True
                        break

                    scaled_loss = loss * micro_weight
                    with self._trace_phase("backward", batch_idx, device):
                        scaled_loss.backward()

                    mean_loss += loss.detach().cpu().item() * micro_weight
                    mean_kl += micro_mean_kl.detach().cpu().item() * micro_weight
                    mean_actor_loss += micro_mean_actor_loss.detach().cpu().item() * micro_weight
                    update_diagnostic_tensors.append(micro_diagnostic_tensors)
                    del log_probs, loss, scaled_loss, micro_mean_kl, micro_mean_actor_loss, micro_diagnostic_tensors, micro_experience

                if skip_minibatch:
                    del experience, update_diagnostic_tensors
                    self._cleanup(
                        "minibatch",
                        batch_idx=batch_idx,
                        device=device,
                        minibatch_idx=minibatch_idx,
                    )
                    continue

                with self._trace_phase("clip_grad", batch_idx, device):
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        max_norm=self.rl_config.max_grad_norm,
                    )
                with self._trace_phase("optimizer_step", batch_idx, device):
                    optimizer.step()
                did_update = True
                self.optimizer_step_count += 1

                mean_kl_loss = mean_loss - mean_actor_loss
                grad_norm_value = grad_norm.detach().cpu().item() if hasattr(grad_norm, "detach") else float(grad_norm)

                kl_weight_next = kl_weight_used
                if self.rl_config.kl_weight > 0:
                    kl_controller.update(mean_kl, self.rl_config.mini_batch_size)
                    self.actor_loss.kl_weight = kl_controller.value
                    kl_weight_next = kl_controller.value

                log_payload = {
                        "train/loss": mean_loss,
                        "train/mean_kl": mean_kl,
                        "train/actor_loss": mean_actor_loss,
                        "train/kl_loss": mean_kl_loss,
                        "train/kl_weight": kl_weight_used,
                        "train/kl_weight_used": kl_weight_used,
                        "train/kl_weight_next": kl_weight_next,
                        "train/grad_norm": grad_norm_value,
                        "train/optimizer_steps": self.optimizer_step_count,
                        "train/minibatch_idx": minibatch_idx,
                        "num_batches_visited": batch_idx,
                }
                update_diagnostics = self._summarize_update_diagnostics(update_diagnostic_tensors)
                log_payload.update(update_diagnostics)
                self.logger(log_payload)

                del experience, grad_norm, update_diagnostic_tensors, update_diagnostics
                self._cleanup(
                    "minibatch",
                    batch_idx=batch_idx,
                    device=device,
                    minibatch_idx=minibatch_idx,
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
            with self._trace_phase("sglang_weight_sync", batch_idx, device):
                generation_backend.sync_weights_from_model(model, self.tokenizer, batch_idx)
            self._cleanup("after_sglang_sync", batch_idx=batch_idx, device=device)

        self._cleanup("after_update_batch", batch_idx=batch_idx, device=device)

        train_state["model"] = model
        train_state["optimizer"] = optimizer
        train_state["lr_scheduler"] = lr_scheduler
        train_state["kl_controller"] = kl_controller

        return train_state

    def log_rollout_stats(self, rollout_stats: Dict[str, Any]) -> None:
        
        batch_idx = rollout_stats["batch_idx"]

        print(f"Batch indx {batch_idx}")
        if torch.cuda.is_available() and self._should_log_perf(batch_idx):
            snapshot = cuda_memory_snapshot()
            print(f"torch.cuda.memory_allocated: {self._bytes_to_gb(snapshot.allocated_bytes):f}GB")
            print(f"torch.cuda.memory_reserved: {self._bytes_to_gb(snapshot.reserved_bytes):f}GB")
            print(f"torch.cuda.max_memory_reserved: {self._bytes_to_gb(snapshot.max_reserved_bytes):f}GB")

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
            "train/rollout_questions": len(rollout_stats["question"]),
            "train/rollout_samples": len(rollout_stats["completions"]),
            "train/minibatches_per_rollout": int(np.ceil(len(rollout_stats["completions"]) / self.rl_config.mini_batch_size)),
            "train/optimizer_steps": self.optimizer_step_count,
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
                    "ad_cispo/clip_std": ad_cispo_stats.clip_std,
                    "ad_cispo/clip_p10": ad_cispo_stats.clip_p10,
                    "ad_cispo/clip_p50": ad_cispo_stats.clip_p50,
                    "ad_cispo/clip_p90": ad_cispo_stats.clip_p90,
                    "ad_cispo/clip_below_mean_fraction": ad_cispo_stats.clip_below_mean_fraction,
                    "ad_cispo/clip_above_mean_fraction": ad_cispo_stats.clip_above_mean_fraction,
                    "ad_cispo/clip_at_min_fraction": ad_cispo_stats.clip_at_min_fraction,
                    "ad_cispo/clip_at_max_fraction": ad_cispo_stats.clip_at_max_fraction,
                    "ad_cispo/multiplier_mean": ad_cispo_stats.multiplier_mean,
                    "ad_cispo/multiplier_min": ad_cispo_stats.multiplier_min,
                    "ad_cispo/multiplier_max": ad_cispo_stats.multiplier_max,
                    "ad_cispo/multiplier_std": ad_cispo_stats.multiplier_std,
                    "ad_cispo/saliency_mean": ad_cispo_stats.saliency_mean,
                    "ad_cispo/saliency_std": ad_cispo_stats.saliency_std,
                    "num_batches_visited": batch_idx,
                }
            )
            distribution_metrics = rollout_stats.get("ad_cispo_distribution_metrics")
            if distribution_metrics:
                self.logger(
                    {
                        **distribution_metrics,
                        "num_batches_visited": batch_idx,
                    }
                )
            log_ad_cispo_top_token_table(
                logger=self.logger,
                rows=rollout_stats.get("ad_cispo_top_token_rows", []),
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
    
