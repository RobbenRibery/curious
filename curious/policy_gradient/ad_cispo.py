from dataclasses import dataclass
from typing import Optional

import torch
from transformers import PreTrainedModel

from curious.sampling.sampling import _sequence_log_probs_from_logits


@dataclass(frozen=True)
class SequenceTokenBatch:
    sequence_ids: torch.Tensor
    attention_mask: torch.Tensor


@dataclass(frozen=True)
class RawTokenSaliency:
    values: torch.Tensor


@dataclass(frozen=True)
class ActionTokenSaliency:
    values: torch.Tensor
    action_mask: torch.Tensor


@dataclass(frozen=True)
class TokenClipThresholds:
    values: torch.Tensor
    multipliers: torch.Tensor


@dataclass(frozen=True)
class ADCispoStats:
    clip_mean: float
    clip_min: float
    clip_max: float
    multiplier_mean: float
    saliency_mean: float


@dataclass(frozen=True)
class ADCispoThresholdConfig:
    clip_high: float
    min_multiplier: float = 0.0
    max_multiplier: Optional[float] = None
    eps: float = 1e-8


@dataclass(frozen=True)
class ReferencePolicyFeatureRequest:
    model: PreTrainedModel
    sequence_ids: torch.Tensor
    attention_mask: torch.Tensor
    action_mask: torch.Tensor
    clip_high: float
    logits_minibatch_size: int
    top_layers: int
    min_multiplier: float
    max_multiplier: Optional[float]
    eps: float


@dataclass(frozen=True)
class ReferencePolicyFeatures:
    log_probs: torch.Tensor
    raw_saliency: RawTokenSaliency
    action_saliency: ActionTokenSaliency
    token_clip_thresholds: TokenClipThresholds
    stats: ADCispoStats


def align_saliency_to_actions(raw_saliency: RawTokenSaliency, action_mask: torch.Tensor) -> ActionTokenSaliency:
    values = raw_saliency.values[:, 1:]
    if values.shape != action_mask.shape:
        raise ValueError(
            "AD-CISPO action saliency must align with action_mask: "
            f"got saliency {tuple(values.shape)} and mask {tuple(action_mask.shape)}"
        )
    return ActionTokenSaliency(values=values, action_mask=action_mask)


def normalize_action_saliency(
    action_saliency: ActionTokenSaliency,
    threshold_config: ADCispoThresholdConfig,
) -> TokenClipThresholds:
    values = action_saliency.values.float()
    mask = action_saliency.action_mask.bool()
    if values.shape != mask.shape:
        raise ValueError(
            "AD-CISPO saliency and mask shapes must match: "
            f"got saliency {tuple(values.shape)} and mask {tuple(mask.shape)}"
        )
    if threshold_config.clip_high <= 0:
        raise ValueError("AD-CISPO clip_high must be positive")
    if threshold_config.min_multiplier < 0:
        raise ValueError("AD-CISPO min_multiplier must be non-negative")
    if threshold_config.min_multiplier > 1:
        raise ValueError("AD-CISPO min_multiplier cannot exceed 1 when mean multiplier must stay 1")
    if threshold_config.max_multiplier is not None and threshold_config.max_multiplier <= 0:
        raise ValueError("AD-CISPO max_multiplier must be positive when set")
    if threshold_config.max_multiplier is not None and threshold_config.max_multiplier < 1:
        raise ValueError("AD-CISPO max_multiplier cannot be below 1 when mean multiplier must stay 1")
    if (
        threshold_config.max_multiplier is not None
        and threshold_config.max_multiplier < threshold_config.min_multiplier
    ):
        raise ValueError("AD-CISPO max_multiplier must be >= min_multiplier")

    finite_positive_values = torch.where(
        torch.isfinite(values) & (values > 0),
        values,
        torch.zeros_like(values),
    )
    token_counts = mask.sum(dim=-1, keepdim=True).clamp_min(1)
    saliency_sums = (finite_positive_values * mask).sum(dim=-1, keepdim=True)
    saliency_means = saliency_sums / token_counts

    fallback_rows = saliency_sums <= threshold_config.eps
    multipliers = finite_positive_values / saliency_means.clamp_min(threshold_config.eps)
    multipliers = torch.where(mask, multipliers, torch.zeros_like(multipliers))
    multipliers = torch.where(
        fallback_rows.expand_as(multipliers),
        torch.ones_like(multipliers),
        multipliers,
    )
    multipliers = torch.where(mask, multipliers, torch.zeros_like(multipliers))

    multipliers = _project_masked_multipliers(
        multipliers=multipliers,
        mask=mask,
        min_multiplier=threshold_config.min_multiplier,
        max_multiplier=threshold_config.max_multiplier,
        eps=threshold_config.eps,
    )
    clip_values = multipliers * threshold_config.clip_high
    scalar_clip = torch.full_like(clip_values, threshold_config.clip_high)
    clip_values = torch.where(mask, clip_values, scalar_clip)
    multipliers = torch.where(mask, multipliers, torch.ones_like(multipliers))

    return TokenClipThresholds(
        values=clip_values.to(dtype=action_saliency.values.dtype),
        multipliers=multipliers.to(dtype=action_saliency.values.dtype),
    )


def _project_masked_multipliers(
    multipliers: torch.Tensor,
    mask: torch.Tensor,
    min_multiplier: float,
    max_multiplier: Optional[float],
    eps: float,
) -> torch.Tensor:
    projected = torch.zeros_like(multipliers)
    upper = float("inf") if max_multiplier is None else max_multiplier
    for row_idx in range(multipliers.shape[0]):
        row_mask = mask[row_idx]
        if not row_mask.any():
            continue
        row_values = multipliers[row_idx, row_mask].clone()
        row_values = torch.clamp(row_values, min=min_multiplier, max=upper)
        row_values = _renormalize_row_to_unit_mean(
            values=row_values,
            min_multiplier=min_multiplier,
            max_multiplier=max_multiplier,
            eps=eps,
        )
        projected[row_idx, row_mask] = row_values
    return projected


def _renormalize_row_to_unit_mean(
    values: torch.Tensor,
    min_multiplier: float,
    max_multiplier: Optional[float],
    eps: float,
) -> torch.Tensor:
    if values.numel() == 0:
        return values

    upper = float("inf") if max_multiplier is None else max_multiplier
    target_sum = torch.tensor(float(values.numel()), device=values.device, dtype=values.dtype)
    lower_bound = torch.full_like(values, min_multiplier)
    upper_bound = torch.full_like(values, upper)
    fixed = torch.zeros_like(values, dtype=torch.bool)
    projected = values.clone()

    for _ in range(values.numel() + 1):
        active = ~fixed
        if not active.any():
            break
        fixed_sum = projected[fixed].sum()
        active_sum = projected[active].sum()
        if active_sum <= eps:
            projected[active] = (target_sum - fixed_sum) / active.sum().clamp_min(1)
        else:
            projected[active] = projected[active] * ((target_sum - fixed_sum) / active_sum)

        low_violations = active & (projected < lower_bound)
        high_violations = active & (projected > upper_bound)
        new_fixed = low_violations | high_violations
        if not new_fixed.any():
            break
        projected[low_violations] = lower_bound[low_violations]
        projected[high_violations] = upper_bound[high_violations]
        fixed = fixed | new_fixed

    return torch.clamp(projected, min=min_multiplier, max=upper)


def summarize_ad_cispo(
    action_saliency: ActionTokenSaliency,
    token_clip_thresholds: TokenClipThresholds,
) -> ADCispoStats:
    mask = action_saliency.action_mask.bool()
    if not mask.any():
        return ADCispoStats(
            clip_mean=0.0,
            clip_min=0.0,
            clip_max=0.0,
            multiplier_mean=0.0,
            saliency_mean=0.0,
        )
    masked_clips = token_clip_thresholds.values.float()[mask]
    masked_multipliers = token_clip_thresholds.multipliers.float()[mask]
    masked_saliency = action_saliency.values.float()[mask]
    return ADCispoStats(
        clip_mean=masked_clips.mean().detach().cpu().item(),
        clip_min=masked_clips.min().detach().cpu().item(),
        clip_max=masked_clips.max().detach().cpu().item(),
        multiplier_mean=masked_multipliers.mean().detach().cpu().item(),
        saliency_mean=masked_saliency.mean().detach().cpu().item(),
    )


def compute_token_clip_thresholds(
    raw_saliency: RawTokenSaliency,
    action_mask: torch.Tensor,
    threshold_config: ADCispoThresholdConfig,
) -> tuple[ActionTokenSaliency, TokenClipThresholds, ADCispoStats]:
    action_saliency = align_saliency_to_actions(raw_saliency, action_mask)
    token_clip_thresholds = normalize_action_saliency(action_saliency, threshold_config)
    stats = summarize_ad_cispo(action_saliency, token_clip_thresholds)
    return action_saliency, token_clip_thresholds, stats


def compute_reference_policy_features(
    request: ReferencePolicyFeatureRequest,
) -> ReferencePolicyFeatures:
    token_logprobs = []
    raw_saliency_values = []

    for begin_idx in range(0, request.sequence_ids.shape[0], request.logits_minibatch_size):
        mini_ids = request.sequence_ids[begin_idx:begin_idx + request.logits_minibatch_size]
        mini_mask = request.attention_mask[begin_idx:begin_idx + request.logits_minibatch_size]
        mini_logits, mini_saliency = extract_kv_norm_saliency(
            model=request.model,
            token_batch=SequenceTokenBatch(sequence_ids=mini_ids, attention_mask=mini_mask),
            top_layers=request.top_layers,
        )
        log_probs, _ = _sequence_log_probs_from_logits(
            logits=mini_logits[:, :-1],
            output_ids=mini_ids[:, 1:],
            return_entropy=False,
        )
        token_logprobs.append(log_probs)
        raw_saliency_values.append(mini_saliency.values)

    log_probs = torch.cat(token_logprobs, dim=0)
    raw_saliency = RawTokenSaliency(values=torch.cat(raw_saliency_values, dim=0))
    action_saliency, token_clip_thresholds, stats = compute_token_clip_thresholds(
        raw_saliency=raw_saliency,
        action_mask=request.action_mask,
        threshold_config=ADCispoThresholdConfig(
            clip_high=request.clip_high,
            min_multiplier=request.min_multiplier,
            max_multiplier=request.max_multiplier,
            eps=request.eps,
        ),
    )
    return ReferencePolicyFeatures(
        log_probs=log_probs,
        raw_saliency=raw_saliency,
        action_saliency=action_saliency,
        token_clip_thresholds=token_clip_thresholds,
        stats=stats,
    )


def extract_kv_norm_saliency(
    model: PreTrainedModel,
    token_batch: SequenceTokenBatch,
    top_layers: int,
) -> tuple[torch.Tensor, RawTokenSaliency]:
    if top_layers <= 0:
        raise ValueError("AD-CISPO top_layers must be positive")

    layers = _find_decoder_layers(model)
    selected_layers = layers[-min(top_layers, len(layers)):]
    captured_norms: list[torch.Tensor] = []
    handles = []

    def capture_key_norm(_module, _inputs, output):
        key_states = output[0] if isinstance(output, tuple) else output
        captured_norms.append(key_states.detach().float().norm(dim=-1))

    for layer in selected_layers:
        key_projection = _find_key_projection(layer)
        handles.append(key_projection.register_forward_hook(capture_key_norm))

    try:
        model_output = model(
            input_ids=token_batch.sequence_ids,
            attention_mask=token_batch.attention_mask,
            use_cache=False,
        )
    finally:
        for handle in handles:
            handle.remove()

    if not captured_norms:
        raise RuntimeError("AD-CISPO did not capture any key-projection activations")

    raw_saliency = torch.stack(captured_norms, dim=0).mean(dim=0)
    return model_output["logits"], RawTokenSaliency(values=raw_saliency)


def _find_decoder_layers(model: PreTrainedModel) -> list[torch.nn.Module]:
    candidate_paths = (
        ("model", "layers"),
        ("transformer", "h"),
        ("gpt_neox", "layers"),
        ("decoder", "layers"),
    )
    for path in candidate_paths:
        value = model
        for attr in path:
            value = getattr(value, attr, None)
            if value is None:
                break
        if value is not None and len(value) > 0:
            return list(value)
    raise ValueError("AD-CISPO could not find decoder layers on the reference model")


def _find_key_projection(layer: torch.nn.Module) -> torch.nn.Module:
    candidate_paths = (
        ("self_attn", "k_proj"),
        ("self_attention", "k_proj"),
        ("attention", "k_proj"),
        ("attn", "k_proj"),
        ("self_attn", "key"),
        ("attention", "key"),
    )
    for path in candidate_paths:
        value = layer
        for attr in path:
            value = getattr(value, attr, None)
            if value is None:
                break
        if isinstance(value, torch.nn.Module):
            return value
    raise ValueError("AD-CISPO could not find a key projection on a decoder layer")
