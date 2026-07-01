from dataclasses import dataclass, field
import importlib
import math
from typing import Any, Optional

import torch
from transformers import PreTrainedModel

from curious.sampling.sampling import _sequence_log_probs_from_logits


FLOAT_TENSOR_DTYPE = torch.bfloat16
_ROTARY_HELPER_CACHE: dict[str, Any] = {}
SEPARATOR_SINK_TOKEN_TEXTS = frozenset(
    {
        "",
        " ",
        "\t",
        "\n",
        "\r",
        ",",
        ".",
        ";",
        ":",
        "!",
        "?",
        "(",
        ")",
        "[",
        "]",
        "{",
        "}",
        '"',
        "'",
        "`",
        " ,",
        " .",
        " ;",
        " :",
        " !",
        " ?",
        "\n,",
        "\n.",
    }
)


def to_ad_cispo_float_dtype(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to(dtype=FLOAT_TENSOR_DTYPE) if tensor.is_floating_point() else tensor


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
class SaliencyMasks:
    attention_mask: torch.Tensor
    action_mask: torch.Tensor
    adaptive_clip_mask: torch.Tensor
    full_target_mask: torch.Tensor


@dataclass(frozen=True)
class LayerAttentionInputs:
    query: torch.Tensor
    key: torch.Tensor
    additive_attention_mask: Optional[torch.Tensor]
    scaling: float
    sliding_window: Optional[int] = None


@dataclass(frozen=True)
class ADCispoStats:
    clip_mean: float
    clip_min: float
    clip_max: float
    clip_std: float = 0.0
    clip_p10: float = 0.0
    clip_p50: float = 0.0
    clip_p90: float = 0.0
    clip_below_mean_fraction: float = 0.0
    clip_above_mean_fraction: float = 0.0
    clip_at_min_fraction: float = 0.0
    clip_at_max_fraction: float = 0.0
    multiplier_mean: float = 0.0
    multiplier_min: float = 0.0
    multiplier_max: float = 0.0
    multiplier_std: float = 0.0
    saliency_mean: float = 0.0
    saliency_std: float = 0.0


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
    return_log_probs: bool = True
    saliency_method: str = "future_attention_in_degree"
    attention_block_size: int = 256
    saliency_minibatch_size: int = 0
    sink_token_ids: frozenset[int] = frozenset()
    advantages: Optional[torch.Tensor] = None


@dataclass(frozen=True)
class ReferencePolicyFeatures:
    log_probs: Optional[torch.Tensor]
    raw_saliency: RawTokenSaliency
    action_saliency: ActionTokenSaliency
    token_clip_thresholds: TokenClipThresholds
    stats: ADCispoStats
    diagnostics: dict[str, float] = field(default_factory=dict)


def align_saliency_to_actions(raw_saliency: RawTokenSaliency, action_mask: torch.Tensor) -> ActionTokenSaliency:
    values = raw_saliency.values[:, 1:]
    if values.shape != action_mask.shape:
        raise ValueError(
            "AD-CISPO action saliency must align with action_mask: "
            f"got saliency {tuple(values.shape)} and mask {tuple(action_mask.shape)}"
        )
    return ActionTokenSaliency(values=values, action_mask=action_mask)


def collect_special_token_ids(tokenizer: Any) -> frozenset[int]:
    token_ids = set()
    for attr in ("pad_token_id", "bos_token_id", "eos_token_id", "unk_token_id"):
        token_id = getattr(tokenizer, attr, None)
        if token_id is not None:
            token_ids.add(int(token_id))
    for token_id in getattr(tokenizer, "additional_special_tokens_ids", []) or []:
        if token_id is not None:
            token_ids.add(int(token_id))
    return frozenset(token_ids)


def collect_ad_cispo_sink_token_ids(tokenizer: Any) -> frozenset[int]:
    cached_token_ids = getattr(tokenizer, "_ad_cispo_sink_token_ids_cache", None)
    if cached_token_ids is not None:
        return frozenset(cached_token_ids)

    token_ids = set(collect_special_token_ids(tokenizer))
    encode = getattr(tokenizer, "encode", None)
    if callable(encode):
        for text in SEPARATOR_SINK_TOKEN_TEXTS:
            try:
                encoded = encode(text, add_special_tokens=False)
            except TypeError:
                encoded = encode(text)
            except Exception:
                continue
            for token_id in encoded or []:
                if token_id is not None:
                    token_ids.add(int(token_id))
    sink_token_ids = frozenset(token_ids)
    try:
        setattr(tokenizer, "_ad_cispo_sink_token_ids_cache", sink_token_ids)
    except Exception:
        pass
    return sink_token_ids


def build_adaptive_clip_mask(
    sequence_ids: torch.Tensor,
    action_mask: torch.Tensor,
    sink_token_ids: frozenset[int],
) -> torch.Tensor:
    if sequence_ids.ndim != 2 or action_mask.ndim != 2:
        raise ValueError("AD-CISPO sequence_ids and action_mask must be rank-2 tensors")
    if sequence_ids.shape[0] != action_mask.shape[0] or sequence_ids.shape[1] - 1 != action_mask.shape[1]:
        raise ValueError(
            "AD-CISPO action_mask must align with sequence_ids[:, 1:]: "
            f"got sequence_ids {tuple(sequence_ids.shape)} and action_mask {tuple(action_mask.shape)}"
        )

    adaptive_mask = action_mask.bool().clone()
    if sink_token_ids:
        generated_token_ids = sequence_ids[:, 1:]
        sink_tokens = torch.zeros_like(adaptive_mask)
        for token_id in sink_token_ids:
            sink_tokens = sink_tokens | (generated_token_ids == token_id)
        adaptive_mask = adaptive_mask & ~sink_tokens
    return adaptive_mask


def expand_action_mask_to_full_targets(action_mask: torch.Tensor) -> torch.Tensor:
    prefix = torch.zeros(
        (action_mask.shape[0], 1),
        dtype=torch.bool,
        device=action_mask.device,
    )
    return torch.cat([prefix, action_mask.bool()], dim=1)


def build_saliency_masks(
    sequence_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    action_mask: torch.Tensor,
    sink_token_ids: frozenset[int],
) -> SaliencyMasks:
    adaptive_clip_mask = build_adaptive_clip_mask(
        sequence_ids=sequence_ids,
        action_mask=action_mask,
        sink_token_ids=sink_token_ids,
    )
    return SaliencyMasks(
        attention_mask=attention_mask.bool(),
        action_mask=action_mask.bool(),
        adaptive_clip_mask=adaptive_clip_mask,
        full_target_mask=expand_action_mask_to_full_targets(adaptive_clip_mask),
    )


def compute_blockwise_future_indegree(
    attention_inputs: LayerAttentionInputs,
    masks: SaliencyMasks,
    block_size: int,
) -> torch.Tensor:
    query = to_ad_cispo_float_dtype(attention_inputs.query)
    key = to_ad_cispo_float_dtype(attention_inputs.key)
    if query.ndim != 4 or key.ndim != 4:
        raise ValueError("AD-CISPO attention query/key tensors must have shape (batch, heads, seq_len, head_dim)")
    if block_size <= 0:
        raise ValueError("AD-CISPO attention_block_size must be positive")
    if query.shape[0] != key.shape[0] or query.shape[2] != key.shape[2] or query.shape[3] != key.shape[3]:
        raise ValueError(f"AD-CISPO query/key shapes are incompatible: {tuple(query.shape)} vs {tuple(key.shape)}")

    batch_size, num_heads, seq_len, _ = query.shape
    key = _repeat_key_heads(key=key, num_query_heads=num_heads)
    saliency = torch.zeros((batch_size, seq_len), device=query.device, dtype=FLOAT_TENSOR_DTYPE)
    full_target_mask = masks.full_target_mask.to(device=query.device)
    full_query_mask = full_target_mask
    key_positions = torch.arange(seq_len, device=query.device)

    for begin_idx in range(0, seq_len, block_size):
        end_idx = min(seq_len, begin_idx + block_size)
        query_block = query[:, :, begin_idx:end_idx, :]
        query_positions = torch.arange(begin_idx, end_idx, device=query.device)
        key_limit = end_idx
        if attention_inputs.sliding_window is not None:
            key_start = max(0, begin_idx - attention_inputs.sliding_window + 1)
        else:
            key_start = 0

        key_slice = key[:, :, key_start:key_limit, :]
        scores = torch.matmul(query_block, key_slice.transpose(2, 3)) * attention_inputs.scaling
        scores = _apply_attention_masks(
            scores=scores,
            additive_attention_mask=attention_inputs.additive_attention_mask,
            query_begin=begin_idx,
            query_end=end_idx,
            key_begin=key_start,
            key_end=key_limit,
            query_positions=query_positions,
            key_positions=key_positions[key_start:key_limit],
        )
        weights = torch.softmax(scores, dim=-1).to(dtype=FLOAT_TENSOR_DTYPE)
        weights = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)

        query_mask = full_query_mask[:, begin_idx:end_idx]
        target_mask = full_target_mask[:, key_start:key_limit]
        future_mask = build_future_attention_masks(
            query_positions=query_positions,
            key_positions=key_positions[key_start:key_limit],
        )
        block_mask = (
            query_mask[:, None, :, None]
            & target_mask[:, None, None, :]
            & future_mask[None, None, :, :]
        )
        weights = torch.where(block_mask, weights, torch.zeros_like(weights))
        saliency[:, key_start:key_limit] += weights.sum(dim=(1, 2)) / weights.new_tensor(num_heads)

    return saliency


def build_future_attention_masks(
    query_positions: torch.Tensor,
    key_positions: torch.Tensor,
) -> torch.Tensor:
    return key_positions.unsqueeze(0) < query_positions.unsqueeze(1)


def compute_layer_future_indegree(
    attention_inputs: LayerAttentionInputs,
    masks: SaliencyMasks,
    block_size: int,
) -> torch.Tensor:
    return compute_blockwise_future_indegree(
        attention_inputs=attention_inputs,
        masks=masks,
        block_size=block_size,
    )


def merge_layer_saliencies(layer_saliencies: list[torch.Tensor]) -> RawTokenSaliency:
    if not layer_saliencies:
        raise RuntimeError("AD-CISPO did not capture any exact attention saliency tensors")
    return RawTokenSaliency(values=torch.stack(layer_saliencies, dim=0).mean(dim=0))


def _repeat_key_heads(key: torch.Tensor, num_query_heads: int) -> torch.Tensor:
    num_key_heads = key.shape[1]
    if num_key_heads == num_query_heads:
        return key
    if num_query_heads % num_key_heads != 0:
        raise ValueError(f"Cannot repeat {num_key_heads} key heads to {num_query_heads} query heads")
    repeat_factor = num_query_heads // num_key_heads
    return (
        key[:, :, None, :, :]
        .expand(key.shape[0], num_key_heads, repeat_factor, key.shape[2], key.shape[3])
        .reshape(key.shape[0], num_query_heads, key.shape[2], key.shape[3])
    )


def _apply_attention_masks(
    scores: torch.Tensor,
    additive_attention_mask: Optional[torch.Tensor],
    query_begin: int,
    query_end: int,
    key_begin: int,
    key_end: int,
    query_positions: torch.Tensor,
    key_positions: torch.Tensor,
) -> torch.Tensor:
    causal_mask = key_positions.unsqueeze(0) > query_positions.unsqueeze(1)
    scores = scores.masked_fill(causal_mask[None, None, :, :], torch.finfo(scores.dtype).min)
    if additive_attention_mask is None:
        return scores
    if additive_attention_mask.ndim == 4:
        mask_slice = additive_attention_mask[:, :, query_begin:query_end, key_begin:key_end]
        return scores + mask_slice.to(device=scores.device, dtype=scores.dtype)
    if additive_attention_mask.ndim == 2:
        key_padding_mask = additive_attention_mask[:, key_begin:key_end].bool()
        return scores.masked_fill(~key_padding_mask[:, None, None, :], torch.finfo(scores.dtype).min)
    raise ValueError(f"Unsupported AD-CISPO attention mask shape: {tuple(additive_attention_mask.shape)}")


def normalize_action_saliency(
    action_saliency: ActionTokenSaliency,
    threshold_config: ADCispoThresholdConfig,
) -> TokenClipThresholds:
    values = to_ad_cispo_float_dtype(action_saliency.values)
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
    token_counts = mask.sum(dim=-1, keepdim=True).clamp_min(1).to(dtype=values.dtype)
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

    if threshold_config.min_multiplier != 0.0 or threshold_config.max_multiplier is not None:
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
        values=clip_values.to(dtype=FLOAT_TENSOR_DTYPE),
        multipliers=multipliers.to(dtype=FLOAT_TENSOR_DTYPE),
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
            projected[active] = (target_sum - fixed_sum) / active.sum().clamp_min(1).to(dtype=values.dtype)
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
    masked_clips = to_ad_cispo_float_dtype(token_clip_thresholds.values)[mask]
    masked_multipliers = to_ad_cispo_float_dtype(token_clip_thresholds.multipliers)[mask]
    masked_saliency = to_ad_cispo_float_dtype(action_saliency.values)[mask]
    clip_values = masked_clips.to(dtype=torch.float32)
    multiplier_values = masked_multipliers.to(dtype=torch.float32)
    saliency_values = masked_saliency.to(dtype=torch.float32)
    clip_mean = clip_values.mean()
    clip_min = clip_values.min()
    clip_max = clip_values.max()
    clip_atol = torch.finfo(clip_values.dtype).eps * 8
    return ADCispoStats(
        clip_mean=clip_mean.detach().cpu().item(),
        clip_min=clip_min.detach().cpu().item(),
        clip_max=clip_max.detach().cpu().item(),
        clip_std=clip_values.std(unbiased=False).detach().cpu().item(),
        clip_p10=torch.quantile(clip_values, 0.10).detach().cpu().item(),
        clip_p50=torch.quantile(clip_values, 0.50).detach().cpu().item(),
        clip_p90=torch.quantile(clip_values, 0.90).detach().cpu().item(),
        clip_below_mean_fraction=(clip_values < clip_mean).to(dtype=torch.float32).mean().detach().cpu().item(),
        clip_above_mean_fraction=(clip_values > clip_mean).to(dtype=torch.float32).mean().detach().cpu().item(),
        clip_at_min_fraction=torch.isclose(clip_values, clip_min, atol=clip_atol, rtol=0.0).to(dtype=torch.float32).mean().detach().cpu().item(),
        clip_at_max_fraction=torch.isclose(clip_values, clip_max, atol=clip_atol, rtol=0.0).to(dtype=torch.float32).mean().detach().cpu().item(),
        multiplier_mean=multiplier_values.mean().detach().cpu().item(),
        multiplier_min=multiplier_values.min().detach().cpu().item(),
        multiplier_max=multiplier_values.max().detach().cpu().item(),
        multiplier_std=multiplier_values.std(unbiased=False).detach().cpu().item(),
        saliency_mean=saliency_values.mean().detach().cpu().item(),
        saliency_std=saliency_values.std(unbiased=False).detach().cpu().item(),
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


def derive_adaptive_clip_bounds(
    raw_saliency: RawTokenSaliency,
    action_mask: torch.Tensor,
    threshold_config: ADCispoThresholdConfig,
) -> tuple[ActionTokenSaliency, TokenClipThresholds, ADCispoStats]:
    return compute_token_clip_thresholds(
        raw_saliency=raw_saliency,
        action_mask=action_mask,
        threshold_config=threshold_config,
    )


def compute_reference_policy_features(
    request: ReferencePolicyFeatureRequest,
) -> ReferencePolicyFeatures:
    token_logprobs = []
    raw_saliency_values = []
    feature_minibatch_size = (
        request.saliency_minibatch_size
        if request.saliency_minibatch_size > 0
        else request.logits_minibatch_size
    )
    if request.logits_minibatch_size <= 0:
        raise ValueError("AD-CISPO logits_minibatch_size must be positive")
    if feature_minibatch_size <= 0:
        raise ValueError("AD-CISPO saliency_minibatch_size must be positive when enabled")
    if request.return_log_probs and feature_minibatch_size > request.logits_minibatch_size:
        raise ValueError("AD-CISPO saliency_minibatch_size must be <= logits_minibatch_size when returning log probs")

    for begin_idx in range(0, request.sequence_ids.shape[0], feature_minibatch_size):
        end_idx = begin_idx + feature_minibatch_size
        mini_ids = request.sequence_ids[begin_idx:end_idx]
        mini_mask = request.attention_mask[begin_idx:end_idx]
        mini_action_mask = request.action_mask[begin_idx:end_idx]
        mini_masks = build_saliency_masks(
            sequence_ids=mini_ids,
            attention_mask=mini_mask,
            action_mask=mini_action_mask,
            sink_token_ids=request.sink_token_ids,
        )
        token_batch = SequenceTokenBatch(sequence_ids=mini_ids, attention_mask=mini_mask)
        if request.saliency_method == "future_attention_in_degree":
            mini_logits, mini_saliency = extract_future_attention_indegree_saliency(
                model=request.model,
                token_batch=token_batch,
                masks=mini_masks,
                top_layers=request.top_layers,
                block_size=request.attention_block_size,
                return_logits=request.return_log_probs,
            )
        elif request.saliency_method == "kv_norm":
            mini_logits, mini_saliency = extract_kv_norm_saliency(
                model=request.model,
                token_batch=token_batch,
                top_layers=request.top_layers,
                return_logits=request.return_log_probs,
            )
        elif request.saliency_method in {"causal_tangent", "causal_tangent_smoothed"}:
            if request.advantages is None:
                raise ValueError(f"AD-CISPO {request.saliency_method} saliency requires advantages")
            mini_advantages = request.advantages[begin_idx:end_idx]
            mini_logits, mini_saliency = extract_causal_tangent_saliency(
                model=request.model,
                token_batch=token_batch,
                masks=mini_masks,
                advantages=mini_advantages,
                top_layers=request.top_layers,
                return_logits=request.return_log_probs,
            )
            if request.saliency_method == "causal_tangent_smoothed":
                mini_saliency = smooth_raw_saliency_radius_one(
                    raw_saliency=mini_saliency,
                    full_target_mask=mini_masks.full_target_mask,
                )
        else:
            raise ValueError(f"Unsupported AD-CISPO saliency method: {request.saliency_method}")
        if request.return_log_probs:
            if mini_logits is None:
                raise RuntimeError("AD-CISPO reference logits are required when return_log_probs=True")
            log_probs, _ = _sequence_log_probs_from_logits(
                logits=mini_logits[:, :-1],
                output_ids=mini_ids[:, 1:],
                return_entropy=False,
            )
            token_logprobs.append(log_probs)
        raw_saliency_values.append(mini_saliency.values)

    log_probs = torch.cat(token_logprobs, dim=0) if request.return_log_probs else None
    raw_saliency = RawTokenSaliency(values=torch.cat(raw_saliency_values, dim=0))
    adaptive_clip_mask = build_adaptive_clip_mask(
        sequence_ids=request.sequence_ids,
        action_mask=request.action_mask,
        sink_token_ids=request.sink_token_ids,
    )
    action_saliency, token_clip_thresholds, stats = derive_adaptive_clip_bounds(
        raw_saliency=raw_saliency,
        action_mask=adaptive_clip_mask,
        threshold_config=ADCispoThresholdConfig(
            clip_high=request.clip_high,
            min_multiplier=request.min_multiplier,
            max_multiplier=request.max_multiplier,
            eps=request.eps,
        ),
    )
    diagnostics = summarize_sink_guard(
        raw_saliency=raw_saliency,
        action_mask=request.action_mask,
        adaptive_clip_mask=adaptive_clip_mask,
    )
    diagnostics[f"ad_cispo/saliency_method_{request.saliency_method}"] = 1.0
    diagnostics["ad_cispo/logits_minibatch_size"] = float(request.logits_minibatch_size)
    diagnostics["ad_cispo/saliency_minibatch_size"] = float(feature_minibatch_size)
    return ReferencePolicyFeatures(
        log_probs=log_probs,
        raw_saliency=raw_saliency,
        action_saliency=action_saliency,
        token_clip_thresholds=token_clip_thresholds,
        stats=stats,
        diagnostics=diagnostics,
    )


def smooth_raw_saliency_radius_one(
    raw_saliency: RawTokenSaliency,
    full_target_mask: torch.Tensor,
    center_weight: float = 0.70,
    neighbor_weight: float = 0.15,
    eps: float = 1e-8,
) -> RawTokenSaliency:
    values = raw_saliency.values.to(dtype=torch.float32)
    mask = full_target_mask.to(device=values.device).bool()
    if values.shape != mask.shape:
        raise ValueError(
            "AD-CISPO smoothed saliency mask must align with raw saliency: "
            f"got saliency {tuple(values.shape)} and mask {tuple(mask.shape)}"
        )
    if center_weight < 0 or neighbor_weight < 0:
        raise ValueError("AD-CISPO smoothing weights must be non-negative")
    if center_weight <= 0 and neighbor_weight <= 0:
        raise ValueError("AD-CISPO smoothing weights cannot all be zero")

    masked_values = torch.where(mask, values, torch.zeros_like(values))
    left_values = torch.zeros_like(masked_values)
    right_values = torch.zeros_like(masked_values)
    left_values[:, 1:] = masked_values[:, :-1]
    right_values[:, :-1] = masked_values[:, 1:]

    left_mask = torch.zeros_like(mask)
    right_mask = torch.zeros_like(mask)
    left_mask[:, 1:] = mask[:, :-1]
    right_mask[:, :-1] = mask[:, 1:]

    weighted_sum = (
        center_weight * masked_values
        + neighbor_weight * left_values
        + neighbor_weight * right_values
    )
    available_weight = (
        center_weight * mask.to(dtype=torch.float32)
        + neighbor_weight * left_mask.to(dtype=torch.float32)
        + neighbor_weight * right_mask.to(dtype=torch.float32)
    )
    smoothed = weighted_sum / available_weight.clamp_min(eps)
    smoothed = torch.where(mask, smoothed, torch.zeros_like(smoothed))
    return RawTokenSaliency(values=smoothed.to(dtype=FLOAT_TENSOR_DTYPE))


def summarize_sink_guard(
    raw_saliency: RawTokenSaliency,
    action_mask: torch.Tensor,
    adaptive_clip_mask: torch.Tensor,
) -> dict[str, float]:
    original_mask = action_mask.bool()
    adaptive_mask = adaptive_clip_mask.bool()
    removed_mask = original_mask & ~adaptive_mask
    original_count = int(original_mask.sum().item())
    removed_count = int(removed_mask.sum().item())
    active_count = int(adaptive_mask.sum().item())

    action_values = raw_saliency.values[:, 1:]
    finite_positive_values = torch.where(
        torch.isfinite(action_values) & (action_values > 0),
        action_values,
        torch.zeros_like(action_values),
    ).to(dtype=torch.float32)
    total_saliency = (finite_positive_values * original_mask.to(dtype=torch.float32)).sum()
    removed_saliency = (finite_positive_values * removed_mask.to(dtype=torch.float32)).sum()
    total_saliency_value = total_saliency.detach().cpu().item()
    if total_saliency_value > 0:
        removed_saliency_fraction = (removed_saliency / total_saliency).detach().cpu().item()
    else:
        removed_saliency_fraction = 0.0

    return {
        "ad_cispo/sink_guard_original_token_count": float(original_count),
        "ad_cispo/sink_guard_active_token_count": float(active_count),
        "ad_cispo/sink_guard_removed_token_count": float(removed_count),
        "ad_cispo/sink_guard_removed_token_fraction": (
            float(removed_count) / float(original_count) if original_count else 0.0
        ),
        "ad_cispo/sink_guard_removed_saliency_fraction": float(removed_saliency_fraction),
    }


def extract_future_attention_indegree_saliency(
    model: PreTrainedModel,
    token_batch: SequenceTokenBatch,
    masks: SaliencyMasks,
    top_layers: int,
    block_size: int,
    return_logits: bool = True,
) -> tuple[Optional[torch.Tensor], RawTokenSaliency]:
    if top_layers <= 0:
        raise ValueError("AD-CISPO top_layers must be positive")

    selected_layers = select_reference_attention_layers(model, top_layers)
    captured_saliencies: list[torch.Tensor] = []
    handles = []

    def capture_exact_saliency(module, args, kwargs):
        hidden_states = _get_forward_arg(args=args, kwargs=kwargs, name="hidden_states", index=0)
        position_embeddings = _get_forward_arg(args=args, kwargs=kwargs, name="position_embeddings", index=1)
        additive_attention_mask = _get_forward_arg(args=args, kwargs=kwargs, name="attention_mask", index=2)
        query, key = _project_post_rope_query_key(
            attention_module=module,
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
        )
        captured_saliencies.append(
            compute_layer_future_indegree(
                attention_inputs=LayerAttentionInputs(
                    query=query,
                    key=key,
                    additive_attention_mask=additive_attention_mask,
                    scaling=float(getattr(module, "scaling", query.shape[-1] ** -0.5)),
                    sliding_window=getattr(module, "sliding_window", None),
                ),
                masks=masks,
                block_size=block_size,
            ).detach()
        )

    for layer in selected_layers:
        attention_module = _find_attention_module(layer)
        handles.append(attention_module.register_forward_pre_hook(capture_exact_saliency, with_kwargs=True))

    try:
        logits = _forward_reference_for_saliency(model, token_batch, return_logits=return_logits)
    finally:
        for handle in handles:
            handle.remove()

    raw_saliency = aggregate_saliency_layers(captured_saliencies)
    return logits, raw_saliency


def extract_causal_tangent_saliency(
    model: PreTrainedModel,
    token_batch: SequenceTokenBatch,
    masks: SaliencyMasks,
    advantages: torch.Tensor,
    top_layers: int,
    return_logits: bool = True,
) -> tuple[Optional[torch.Tensor], RawTokenSaliency]:
    if top_layers <= 0:
        raise ValueError("AD-CISPO top_layers must be positive")
    if advantages.ndim != 1 or advantages.shape[0] != token_batch.sequence_ids.shape[0]:
        raise ValueError(
            "AD-CISPO causal_tangent advantages must be a rank-1 tensor aligned with sequence_ids: "
            f"got advantages {tuple(advantages.shape)} and sequence_ids {tuple(token_batch.sequence_ids.shape)}"
        )

    selected_layers = select_reference_attention_layers(model, top_layers)
    captured_hidden_states: list[torch.Tensor] = []
    handles = []

    def capture_layer_input(_module, args, kwargs):
        hidden_states = _get_forward_arg(args=args, kwargs=kwargs, name="hidden_states", index=0)
        if not hidden_states.requires_grad:
            raise RuntimeError("AD-CISPO causal_tangent hidden states must require gradients")
        captured_hidden_states.append(hidden_states)

    for layer in selected_layers:
        handles.append(layer.register_forward_pre_hook(capture_layer_input, with_kwargs=True))

    try:
        with torch.enable_grad():
            model_output = model(
                input_ids=token_batch.sequence_ids,
                attention_mask=token_batch.attention_mask,
                use_cache=False,
            )
            logits = model_output["logits"]
            action_log_probs = _sequence_log_probs_for_saliency_objective(
                logits=logits[:, :-1],
                output_ids=token_batch.sequence_ids[:, 1:],
            )
            objective_mask = masks.adaptive_clip_mask.to(device=action_log_probs.device)
            objective = _causal_tangent_objective(
                action_log_probs=action_log_probs,
                action_mask=objective_mask,
                advantages=advantages.to(device=action_log_probs.device),
            )
            if not captured_hidden_states:
                raise RuntimeError("AD-CISPO causal_tangent did not capture any hidden states")
            hidden_grads = torch.autograd.grad(
                objective,
                captured_hidden_states,
                retain_graph=False,
                create_graph=False,
                allow_unused=True,
            )
            raw_saliency = _score_causal_tangent_layers(
                hidden_states=captured_hidden_states,
                hidden_grads=hidden_grads,
                full_target_mask=masks.full_target_mask,
            )
            logits_for_return = logits.detach() if return_logits else None
    finally:
        for handle in handles:
            handle.remove()

    return logits_for_return, raw_saliency


def _sequence_log_probs_for_saliency_objective(logits: torch.Tensor, output_ids: torch.Tensor) -> torch.Tensor:
    logits = logits.to(dtype=torch.float32)
    gathered_logits = logits.gather(dim=-1, index=output_ids.unsqueeze(-1)).squeeze(-1)
    return gathered_logits - logits.logsumexp(dim=-1)


def _causal_tangent_objective(
    action_log_probs: torch.Tensor,
    action_mask: torch.Tensor,
    advantages: torch.Tensor,
) -> torch.Tensor:
    mask = action_mask.to(device=action_log_probs.device, dtype=action_log_probs.dtype)
    token_counts = mask.sum(dim=-1).clamp_min(1.0)
    per_sequence_log_prob = (action_log_probs * mask).sum(dim=-1) / token_counts
    weights = torch.nan_to_num(advantages.detach().abs().to(dtype=action_log_probs.dtype), nan=0.0, posinf=0.0, neginf=0.0)
    return (weights * per_sequence_log_prob).sum()


def _score_causal_tangent_layers(
    hidden_states: list[torch.Tensor],
    hidden_grads: tuple[Optional[torch.Tensor], ...],
    full_target_mask: torch.Tensor,
) -> RawTokenSaliency:
    if len(hidden_states) != len(hidden_grads):
        raise RuntimeError("AD-CISPO causal_tangent hidden-state and gradient captures are misaligned")

    layer_scores: list[torch.Tensor] = []
    for hidden_state, hidden_grad in zip(hidden_states, hidden_grads):
        if hidden_grad is None:
            continue
        hidden = hidden_state.detach().to(dtype=torch.float32)
        grad = hidden_grad.detach().to(dtype=torch.float32)
        mask = full_target_mask.to(device=hidden.device, dtype=torch.float32)
        token_counts = mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
        mean_hidden = (hidden * mask.unsqueeze(-1)).sum(dim=1, keepdim=True) / token_counts.unsqueeze(-1)
        centered_hidden = hidden - mean_hidden
        score = (grad * centered_hidden).sum(dim=-1).abs()
        score = score / math.sqrt(float(hidden.shape[-1]))
        score = torch.where(mask.bool(), score, torch.zeros_like(score))
        layer_scores.append(score.to(dtype=FLOAT_TENSOR_DTYPE))

    if not layer_scores:
        raise RuntimeError("AD-CISPO causal_tangent did not produce any usable hidden-state gradients")
    return RawTokenSaliency(values=torch.stack(layer_scores, dim=0).mean(dim=0))


def extract_kv_norm_saliency(
    model: PreTrainedModel,
    token_batch: SequenceTokenBatch,
    top_layers: int,
    return_logits: bool = True,
) -> tuple[Optional[torch.Tensor], RawTokenSaliency]:
    if top_layers <= 0:
        raise ValueError("AD-CISPO top_layers must be positive")

    selected_layers = select_reference_attention_layers(model, top_layers)
    captured_norms: list[torch.Tensor] = []
    handles = []

    def capture_key_norm(_module, _inputs, output):
        key_states = output[0] if isinstance(output, tuple) else output
        captured_norms.append(to_ad_cispo_float_dtype(key_states.detach()).norm(dim=-1))

    for layer in selected_layers:
        key_projection = _find_key_projection(layer)
        handles.append(key_projection.register_forward_hook(capture_key_norm))

    try:
        logits = _forward_reference_for_saliency(model, token_batch, return_logits=return_logits)
    finally:
        for handle in handles:
            handle.remove()

    if not captured_norms:
        raise RuntimeError("AD-CISPO did not capture any key-projection activations")

    raw_saliency = torch.stack(captured_norms, dim=0).mean(dim=0)
    return logits, RawTokenSaliency(values=raw_saliency)


def select_reference_attention_layers(model: PreTrainedModel, top_layers: int) -> list[torch.nn.Module]:
    layers = _find_decoder_layers(model)
    return layers[-min(top_layers, len(layers)):]


def aggregate_saliency_layers(layer_scores: list[torch.Tensor]) -> RawTokenSaliency:
    return merge_layer_saliencies(layer_scores)


def _forward_reference_for_saliency(
    model: PreTrainedModel,
    token_batch: SequenceTokenBatch,
    return_logits: bool,
) -> Optional[torch.Tensor]:
    if return_logits:
        model_output = model(
            input_ids=token_batch.sequence_ids,
            attention_mask=token_batch.attention_mask,
            use_cache=False,
        )
        return model_output["logits"]

    backbone = getattr(model, "model", None)
    if isinstance(backbone, torch.nn.Module):
        try:
            backbone(
                input_ids=token_batch.sequence_ids,
                attention_mask=token_batch.attention_mask,
                use_cache=False,
            )
            return None
        except (NotImplementedError, TypeError):
            pass

    model(
        input_ids=token_batch.sequence_ids,
        attention_mask=token_batch.attention_mask,
        use_cache=False,
    )
    return None


def _get_forward_arg(args: tuple[Any, ...], kwargs: dict[str, Any], name: str, index: int) -> Any:
    if name in kwargs:
        return kwargs[name]
    if len(args) > index:
        return args[index]
    raise ValueError(f"AD-CISPO could not read attention forward argument: {name}")


def _project_post_rope_query_key(
    attention_module: torch.nn.Module,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    if not hasattr(attention_module, "q_proj") or not hasattr(attention_module, "k_proj"):
        raise ValueError("AD-CISPO exact saliency requires an attention module with q_proj and k_proj")
    if not hasattr(attention_module, "head_dim"):
        raise ValueError("AD-CISPO exact saliency requires attention_module.head_dim")

    input_shape = hidden_states.shape[:-1]
    head_dim = int(attention_module.head_dim)
    hidden_shape = (*input_shape, -1, head_dim)
    query_states = attention_module.q_proj(hidden_states).view(hidden_shape)
    key_states = attention_module.k_proj(hidden_states).view(hidden_shape)
    if hasattr(attention_module, "q_norm"):
        query_states = attention_module.q_norm(query_states)
    if hasattr(attention_module, "k_norm"):
        key_states = attention_module.k_norm(key_states)
    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    apply_rotary_pos_emb = _load_apply_rotary_pos_emb(attention_module)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, *position_embeddings)
    return to_ad_cispo_float_dtype(query_states), to_ad_cispo_float_dtype(key_states)


def _load_apply_rotary_pos_emb(attention_module: torch.nn.Module):
    module_name = attention_module.__class__.__module__
    apply_rotary_pos_emb = _ROTARY_HELPER_CACHE.get(module_name)
    if apply_rotary_pos_emb is None:
        module = importlib.import_module(module_name)
        apply_rotary_pos_emb = getattr(module, "apply_rotary_pos_emb", None)
        if apply_rotary_pos_emb is not None:
            _ROTARY_HELPER_CACHE[module_name] = apply_rotary_pos_emb
    if apply_rotary_pos_emb is None:
        raise ValueError("AD-CISPO exact saliency requires a model module with apply_rotary_pos_emb")
    return apply_rotary_pos_emb


def _find_decoder_layers(model: PreTrainedModel) -> list[torch.nn.Module]:
    cached_layers = getattr(model, "_ad_cispo_decoder_layers_cache", None)
    if cached_layers is not None:
        return list(cached_layers)

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
            layers = list(value)
            setattr(model, "_ad_cispo_decoder_layers_cache", layers)
            return layers
    raise ValueError("AD-CISPO could not find decoder layers on the reference model")


def _find_attention_module(layer: torch.nn.Module) -> torch.nn.Module:
    candidate_paths = (
        ("self_attn",),
        ("self_attention",),
        ("attention",),
        ("attn",),
    )
    for path in candidate_paths:
        value = layer
        for attr in path:
            value = getattr(value, attr, None)
            if value is None:
                break
        if isinstance(value, torch.nn.Module):
            return value
    raise ValueError("AD-CISPO could not find an attention module on a decoder layer")


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
