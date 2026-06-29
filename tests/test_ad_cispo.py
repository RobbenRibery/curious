import math

import torch
import torch.nn as nn

from curious.policy_gradient.ad_cispo import (
    ADCispoStats,
    ADCispoThresholdConfig,
    LayerAttentionInputs,
    RawTokenSaliency,
    ActionTokenSaliency,
    ReferencePolicyFeatureRequest,
    ReferencePolicyFeatures,
    SaliencyMasks,
    SequenceTokenBatch,
    TokenClipThresholds,
    align_saliency_to_actions,
    build_adaptive_clip_mask,
    build_future_attention_masks,
    build_saliency_masks,
    compute_blockwise_future_indegree,
    compute_reference_policy_features,
    compute_token_clip_thresholds,
    extract_kv_norm_saliency,
)
from curious.policy_gradient.loss import ActorLoss
from curious.replay.experience import (
    Experience,
    iter_experience_minibatches,
    join_experience_batch,
    split_experience_batch,
)
from curious.sampling.sampling import _sequence_log_probs_from_logits, compute_group_advantages, decode_action_tokens
from curious.train.trainer import build_ad_cispo_top_token_rows
from curious.utils.utils import (
    CleanupPolicy,
    CudaMemorySnapshot,
    decide_cleanup,
    move_paddings_to_right,
)


BF16_ATOL = 5e-3


def bf16_tensor(data):
    return torch.tensor(data, dtype=torch.bfloat16)


def test_token_clip_thresholds_keep_masked_mean_equal_to_scalar_clip():
    raw_saliency = RawTokenSaliency(
        values=torch.tensor(
            [
                [0.0, 1.0, 3.0, 5.0],
                [0.0, float("nan"), float("inf"), 0.0],
            ]
        )
    )
    action_mask = torch.tensor(
        [
            [False, True, True],
            [True, True, False],
        ]
    )

    _, thresholds, stats = compute_token_clip_thresholds(
        raw_saliency=raw_saliency,
        action_mask=action_mask,
        threshold_config=ADCispoThresholdConfig(
            clip_high=0.28,
            min_multiplier=0.05,
            max_multiplier=None,
        ),
    )

    masked_thresholds = thresholds.values[action_mask]
    assert thresholds.values.dtype == torch.bfloat16
    assert thresholds.multipliers.dtype == torch.bfloat16
    assert torch.allclose(masked_thresholds.mean(), bf16_tensor(0.28), atol=BF16_ATOL)
    assert torch.allclose(thresholds.values[~action_mask], torch.full_like(thresholds.values[~action_mask], 0.28))
    assert math.isclose(stats.clip_mean, 0.28, rel_tol=0, abs_tol=BF16_ATOL)
    assert math.isclose(stats.multiplier_mean, 1.0, rel_tol=0, abs_tol=BF16_ATOL)


def test_token_clip_thresholds_re_normalize_after_min_max_clamp():
    raw_saliency = RawTokenSaliency(values=torch.tensor([[0.0, 1.0, 100.0, 1.0]]))
    action_mask = torch.tensor([[True, True, True]])

    _, thresholds, _ = compute_token_clip_thresholds(
        raw_saliency=raw_saliency,
        action_mask=action_mask,
        threshold_config=ADCispoThresholdConfig(
            clip_high=0.4,
            min_multiplier=0.5,
            max_multiplier=2.0,
        ),
    )

    assert thresholds.values.dtype == torch.bfloat16
    assert torch.allclose(thresholds.values[action_mask].mean(), bf16_tensor(0.4), atol=BF16_ATOL)
    assert thresholds.multipliers[action_mask].min() >= 0.5
    assert thresholds.multipliers[action_mask].max() <= 2.0


def test_token_clip_thresholds_unbounded_path_preserves_expected_multipliers():
    raw_saliency = RawTokenSaliency(values=torch.tensor([[0.0, 1.0, 3.0, 5.0]]))
    action_mask = torch.tensor([[True, True, True]])

    _, thresholds, _ = compute_token_clip_thresholds(
        raw_saliency=raw_saliency,
        action_mask=action_mask,
        threshold_config=ADCispoThresholdConfig(
            clip_high=0.3,
            min_multiplier=0.0,
            max_multiplier=None,
        ),
    )

    expected_multipliers = bf16_tensor([[1.0 / 3.0, 1.0, 5.0 / 3.0]])
    assert thresholds.multipliers.dtype == torch.bfloat16
    assert torch.allclose(thresholds.multipliers, expected_multipliers, atol=BF16_ATOL)
    assert torch.allclose(thresholds.values[action_mask].mean(), bf16_tensor(0.3), atol=BF16_ATOL)


def test_action_saliency_aligns_to_next_token_log_prob_positions():
    raw_saliency = RawTokenSaliency(values=torch.tensor([[10.0, 20.0, 30.0, 40.0]]))
    action_mask = torch.tensor([[False, True, True]])

    action_saliency = align_saliency_to_actions(raw_saliency, action_mask)

    assert torch.equal(action_saliency.values, torch.tensor([[20.0, 30.0, 40.0]]))


def test_adaptive_clip_mask_excludes_prompt_and_special_tokens():
    sequence_ids = torch.tensor([[11, 12, 21, 99, 22, 0]])
    action_mask = torch.tensor([[False, True, True, True, False]])

    adaptive_mask = build_adaptive_clip_mask(
        sequence_ids=sequence_ids,
        action_mask=action_mask,
        sink_token_ids=frozenset({0, 99}),
    )

    assert torch.equal(adaptive_mask, torch.tensor([[False, True, False, True, False]]))


def test_blockwise_future_indegree_uses_strict_future_generated_tokens():
    query = torch.ones(1, 1, 4, 1)
    key = torch.ones(1, 1, 4, 1)
    sequence_ids = torch.tensor([[11, 21, 22, 23]])
    action_mask = torch.tensor([[True, True, True]])
    masks = build_saliency_masks(
        sequence_ids=sequence_ids,
        attention_mask=torch.ones_like(sequence_ids, dtype=torch.bool),
        action_mask=action_mask,
        sink_token_ids=frozenset(),
    )

    saliency = compute_blockwise_future_indegree(
        attention_inputs=LayerAttentionInputs(
            query=query,
            key=key,
            additive_attention_mask=None,
            scaling=1.0,
        ),
        masks=masks,
        block_size=2,
    )

    expected = bf16_tensor([[0.0, 1.0 / 3.0 + 1.0 / 4.0, 1.0 / 4.0, 0.0]])
    assert saliency.dtype == torch.bfloat16
    assert torch.allclose(saliency, expected, atol=BF16_ATOL)


def test_future_attention_mask_is_strictly_backward_to_targets():
    query_positions = torch.tensor([1, 2, 3])
    key_positions = torch.tensor([0, 1, 2, 3])

    mask = build_future_attention_masks(query_positions=query_positions, key_positions=key_positions)

    assert torch.equal(
        mask,
        torch.tensor(
            [
                [True, False, False, False],
                [True, True, False, False],
                [True, True, True, False],
            ]
        ),
    )


def test_blockwise_future_indegree_masks_special_query_and_target_tokens():
    query = torch.ones(1, 1, 4, 1)
    key = torch.ones(1, 1, 4, 1)
    sequence_ids = torch.tensor([[11, 21, 99, 23]])
    action_mask = torch.tensor([[True, True, True]])
    masks = build_saliency_masks(
        sequence_ids=sequence_ids,
        attention_mask=torch.ones_like(sequence_ids, dtype=torch.bool),
        action_mask=action_mask,
        sink_token_ids=frozenset({99}),
    )

    saliency = compute_blockwise_future_indegree(
        attention_inputs=LayerAttentionInputs(
            query=query,
            key=key,
            additive_attention_mask=None,
            scaling=1.0,
        ),
        masks=masks,
        block_size=3,
    )

    assert saliency.dtype == torch.bfloat16
    assert torch.allclose(saliency, bf16_tensor([[0.0, 0.25, 0.0, 0.0]]), atol=BF16_ATOL)


def test_experience_split_join_preserves_token_clip_high():
    experience = Experience(
        sequences=torch.arange(8).reshape(2, 4),
        attention_mask=torch.ones(2, 4, dtype=torch.bool),
        action_mask=torch.ones(2, 3, dtype=torch.bool),
        returns=torch.tensor([1.0, 2.0]),
        solved_mask=torch.tensor([1.0, 0.0]),
        advantages=torch.tensor([0.5, -0.5]),
        action_log_probs=torch.zeros(2, 3),
        log_probs_ref=torch.zeros(2, 3),
        token_clip_high=torch.arange(6, dtype=torch.float32).reshape(2, 3),
    )

    joined = join_experience_batch(split_experience_batch(experience))

    assert torch.equal(joined.token_clip_high, experience.token_clip_high)


def test_iter_experience_minibatches_slices_batched_experience_without_restacking():
    experience = Experience(
        sequences=torch.arange(20).reshape(5, 4),
        attention_mask=torch.ones(5, 4, dtype=torch.bool),
        action_mask=torch.ones(5, 3, dtype=torch.bool),
        returns=torch.arange(5, dtype=torch.bfloat16),
        solved_mask=torch.ones(5, dtype=torch.bfloat16),
        advantages=torch.arange(5, dtype=torch.bfloat16),
        action_log_probs=torch.arange(15, dtype=torch.bfloat16).reshape(5, 3),
        log_probs_ref=torch.full((5, 3), -1.0, dtype=torch.bfloat16),
        token_clip_high=torch.full((5, 3), 0.28, dtype=torch.bfloat16),
    )

    minibatches = list(iter_experience_minibatches(experience, mini_batch_size=2))

    assert [(batch.start, batch.end) for batch in minibatches] == [(0, 2), (2, 4), (4, 5)]
    assert torch.equal(minibatches[1].experience.sequences, experience.sequences[2:4])
    assert torch.equal(minibatches[2].experience.token_clip_high, experience.token_clip_high[4:5])


def test_cleanup_policy_keeps_minibatch_empty_cache_off_by_default():
    snapshot = CudaMemorySnapshot(
        allocated_bytes=8,
        reserved_bytes=10,
        max_reserved_bytes=10,
        total_bytes=100,
    )

    decision = decide_cleanup(
        snapshot,
        CleanupPolicy(),
        phase="minibatch",
        batch_idx=1,
        minibatch_idx=1,
    )

    assert decision.drop_refs
    assert not decision.run_gc
    assert not decision.empty_cache


def test_cleanup_policy_triggers_batch_and_forced_eval_cleanup():
    snapshot = CudaMemorySnapshot(
        allocated_bytes=50,
        reserved_bytes=95,
        max_reserved_bytes=95,
        total_bytes=100,
    )

    batch_decision = decide_cleanup(snapshot, CleanupPolicy(), phase="after_update_batch", batch_idx=1)
    eval_decision = decide_cleanup(snapshot, CleanupPolicy(), phase="after_eval", batch_idx=1)

    assert batch_decision.run_gc
    assert batch_decision.empty_cache
    assert eval_decision.run_gc
    assert eval_decision.empty_cache


def test_actor_loss_matches_scalar_clip_for_constant_token_thresholds():
    log_probs = torch.log(torch.tensor([[1.25, 1.1, 0.7]]))
    base_experience = Experience(
        sequences=torch.arange(4).reshape(1, 4),
        attention_mask=torch.ones(1, 4, dtype=torch.bool),
        action_mask=torch.ones(1, 3, dtype=torch.bool),
        returns=torch.tensor([1.0]),
        solved_mask=torch.tensor([1.0]),
        advantages=torch.tensor([1.0]),
        action_log_probs=torch.zeros(1, 3),
    )
    token_experience = Experience(
        **{
            **base_experience.__dict__,
            "token_clip_high": torch.full((1, 3), 0.3),
        }
    )
    loss_fn = ActorLoss(epsilon=0.2, epsilon_high=0.3, kl_weight=0, use_clip_high=True)

    scalar_loss, _, _ = loss_fn(log_probs=log_probs, experience=base_experience)
    token_loss, _, _ = loss_fn(log_probs=log_probs, experience=token_experience)

    assert torch.allclose(scalar_loss, token_loss, atol=1e-6)


def test_actor_loss_uses_varied_token_thresholds_on_upper_clip_path():
    log_probs = torch.log(torch.tensor([[1.5]]))
    base_experience = Experience(
        sequences=torch.arange(2).reshape(1, 2),
        attention_mask=torch.ones(1, 2, dtype=torch.bool),
        action_mask=torch.ones(1, 1, dtype=torch.bool),
        returns=torch.tensor([1.0]),
        solved_mask=torch.tensor([1.0]),
        advantages=torch.tensor([1.0]),
        action_log_probs=torch.zeros(1, 1),
    )
    token_experience = Experience(
        **{
            **base_experience.__dict__,
            "token_clip_high": torch.tensor([[0.05]]),
        }
    )
    loss_fn = ActorLoss(epsilon=0.2, epsilon_high=0.3, kl_weight=0, use_clip_high=True)

    scalar_loss, _, _ = loss_fn(log_probs=log_probs, experience=base_experience)
    token_loss, _, _ = loss_fn(log_probs=log_probs, experience=token_experience)

    assert token_loss > scalar_loss


def test_cispo_loss_keeps_gradient_when_ratio_exceeds_upper_clip():
    log_probs = torch.log(torch.tensor([[1.5]])).requires_grad_()
    experience = Experience(
        sequences=torch.arange(2).reshape(1, 2),
        attention_mask=torch.ones(1, 2, dtype=torch.bool),
        action_mask=torch.ones(1, 1, dtype=torch.bool),
        returns=torch.tensor([1.0]),
        solved_mask=torch.tensor([1.0]),
        advantages=torch.tensor([1.0]),
        action_log_probs=torch.zeros(1, 1),
    )
    loss_fn = ActorLoss(epsilon=0.2, kl_weight=0, use_cispo_loss=True)

    loss, _, _ = loss_fn(log_probs=log_probs, experience=experience)
    loss.backward()

    assert torch.allclose(log_probs.grad, torch.tensor([[-1.2]]), atol=BF16_ATOL)


def test_cispo_loss_does_not_apply_lower_is_weight_clip():
    log_probs = torch.log(torch.tensor([[0.1]])).requires_grad_()
    experience = Experience(
        sequences=torch.arange(2).reshape(1, 2),
        attention_mask=torch.ones(1, 2, dtype=torch.bool),
        action_mask=torch.ones(1, 1, dtype=torch.bool),
        returns=torch.tensor([1.0]),
        solved_mask=torch.tensor([1.0]),
        advantages=torch.tensor([1.0]),
        action_log_probs=torch.zeros(1, 1),
    )
    loss_fn = ActorLoss(epsilon=0.2, kl_weight=0, use_cispo_loss=True)

    loss, _, _ = loss_fn(log_probs=log_probs, experience=experience)
    loss.backward()

    assert torch.allclose(log_probs.grad, torch.tensor([[-0.1]]), atol=BF16_ATOL)


def test_cispo_loss_forces_zero_kl_and_token_level_aggregation():
    loss_fn = ActorLoss(kl_weight=0.25, use_cispo_loss=True)

    assert loss_fn.kl_weight == 0.0
    assert loss_fn.use_token_level_loss
    assert loss_fn.aggregation_dim is None


def test_grpo_surrogate_suppresses_gradient_when_ratio_exceeds_upper_clip():
    log_probs = torch.log(torch.tensor([[1.5]])).requires_grad_()
    experience = Experience(
        sequences=torch.arange(2).reshape(1, 2),
        attention_mask=torch.ones(1, 2, dtype=torch.bool),
        action_mask=torch.ones(1, 1, dtype=torch.bool),
        returns=torch.tensor([1.0]),
        solved_mask=torch.tensor([1.0]),
        advantages=torch.tensor([1.0]),
        action_log_probs=torch.zeros(1, 1),
    )
    loss_fn = ActorLoss(epsilon=0.2, kl_weight=0)

    loss, _, _ = loss_fn(log_probs=log_probs, experience=experience)
    loss.backward()

    assert torch.allclose(log_probs.grad, torch.zeros_like(log_probs), atol=1e-6)


def test_actor_loss_returns_raw_mean_kl_not_weighted_kl_loss():
    log_probs = torch.log(torch.tensor([[0.4, 0.6]]))
    log_probs_ref = torch.log(torch.tensor([[0.5, 0.5]]))
    experience = Experience(
        sequences=torch.arange(3).reshape(1, 3),
        attention_mask=torch.ones(1, 3, dtype=torch.bool),
        action_mask=torch.ones(1, 2, dtype=torch.bool),
        returns=torch.tensor([1.0]),
        solved_mask=torch.tensor([1.0]),
        advantages=torch.tensor([1.0]),
        action_log_probs=log_probs.detach(),
        log_probs_ref=log_probs_ref,
    )
    loss_fn = ActorLoss(kl_weight=0.25)

    _, mean_kl, _ = loss_fn(log_probs=log_probs, experience=experience)
    expected_log_ratio = log_probs_ref.to(dtype=torch.bfloat16) - log_probs.to(dtype=torch.bfloat16)
    expected_kl = (expected_log_ratio.exp() - expected_log_ratio - 1).mean()

    assert mean_kl.dtype == torch.bfloat16
    assert torch.allclose(mean_kl, expected_kl, atol=BF16_ATOL)


def test_sequence_log_probs_from_logits_returns_bfloat16_tensors():
    logits = torch.randn(2, 3, 5)
    output_ids = torch.tensor([[1, 2, 3], [2, 3, 4]])

    log_probs, entropy = _sequence_log_probs_from_logits(
        logits=logits,
        output_ids=output_ids,
        return_entropy=True,
    )

    assert log_probs.dtype == torch.bfloat16
    assert entropy.dtype == torch.bfloat16


def test_actor_loss_backprops_weighted_kl_when_advantage_is_zero():
    log_probs = torch.tensor([[-0.9, -0.4]], requires_grad=True)
    log_probs_ref = torch.tensor([[-0.7, -0.5]])
    experience = Experience(
        sequences=torch.arange(3).reshape(1, 3),
        attention_mask=torch.ones(1, 3, dtype=torch.bool),
        action_mask=torch.ones(1, 2, dtype=torch.bool),
        returns=torch.tensor([0.0]),
        solved_mask=torch.tensor([0.0]),
        advantages=torch.tensor([0.0]),
        action_log_probs=log_probs.detach(),
        log_probs_ref=log_probs_ref,
    )
    loss_fn = ActorLoss(kl_weight=0.25)

    loss, mean_kl, mean_actor_loss = loss_fn(log_probs=log_probs, experience=experience)
    expected_weighted_kl = 0.25 * mean_kl

    assert torch.allclose(mean_actor_loss, torch.zeros_like(mean_actor_loss), atol=1e-6)
    assert torch.allclose(loss, expected_weighted_kl, atol=1e-6)

    loss.backward()

    assert log_probs.grad is not None
    assert torch.count_nonzero(log_probs.grad).item() > 0


def test_group_advantages_use_population_std_for_grpo_normalization():
    returns = torch.tensor([[1.0, 2.0, 3.0]])

    advantages = compute_group_advantages(returns=returns, normalize=True)

    assert torch.allclose(advantages, torch.tensor([[-1.2247, 0.0, 1.2247]]), atol=1e-4)


class FakeTokenizer:
    def decode(self, token_ids: torch.Tensor, skip_special_tokens: bool = True) -> str:
        del skip_special_tokens
        return " ".join(str(x.item()) for x in token_ids)


class FakeBatchTokenizer:
    def __init__(self) -> None:
        self.batch_decode_calls = 0

    def batch_decode(self, token_batches: list[list[int]], skip_special_tokens: bool = True) -> list[str]:
        del skip_special_tokens
        self.batch_decode_calls += 1
        return [" ".join(str(token_id) for token_id in token_ids) for token_ids in token_batches]


def test_decode_action_tokens_uses_action_mask_not_fixed_prompt_width():
    sequence_ids = torch.tensor([[0, 0, 11, 12, 21, 22, 99]])
    action_mask = torch.tensor([[False, False, False, True, True, False]])

    completions = decode_action_tokens(FakeTokenizer(), sequence_ids, action_mask)

    assert completions == ["21 22"]


def test_decode_action_tokens_uses_batch_decode_when_available():
    tokenizer = FakeBatchTokenizer()
    sequence_ids = torch.tensor([[0, 11, 21, 22], [0, 12, 31, 32]])
    action_mask = torch.tensor([[False, True, True], [False, True, False]])

    completions = decode_action_tokens(tokenizer, sequence_ids, action_mask)

    assert tokenizer.batch_decode_calls == 1
    assert completions == ["21 22", "31"]


def test_build_ad_cispo_top_token_rows_uses_action_aligned_positions():
    class ListDecodeTokenizer:
        def decode(self, token_ids: list[int], skip_special_tokens: bool = False) -> str:
            del skip_special_tokens
            return " ".join(str(token_id) for token_id in token_ids)

    sequence_ids = torch.tensor([[10, 11, 12, 13], [20, 21, 22, 23]])
    action_mask = torch.tensor([[True, True, False], [True, False, True]])
    reference_features = ReferencePolicyFeatures(
        log_probs=None,
        raw_saliency=RawTokenSaliency(values=torch.zeros_like(sequence_ids, dtype=torch.float32)),
        action_saliency=ActionTokenSaliency(
            values=torch.tensor([[0.1, 0.9, 0.0], [0.8, 0.2, 0.7]]),
            action_mask=action_mask,
        ),
        token_clip_thresholds=TokenClipThresholds(
            values=torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]),
            multipliers=torch.tensor([[0.5, 1.0, 1.5], [2.0, 2.5, 3.0]]),
        ),
        stats=ADCispoStats(clip_mean=0.0, clip_min=0.0, clip_max=0.0),
    )

    rows = build_ad_cispo_top_token_rows(
        tokenizer=ListDecodeTokenizer(),
        reference_features=reference_features,
        sequence_ids=sequence_ids,
        questions=["q0", "q1"],
        answers=["a0", "a1"],
        completions=["c0", "c1"],
        returns=torch.tensor([1.0, -1.0]),
        solved_mask=torch.tensor([1.0, 0.0]),
        advantages=torch.tensor([0.5, -0.5]),
        group_size=1,
        batch_idx=7,
        top_k=3,
    )

    assert [row["token_id"] for row in rows] == [12, 21, 23]
    assert [row["token_position"] for row in rows] == [2, 1, 3]
    assert [row["sequence_idx"] for row in rows] == [0, 1, 1]
    assert all(
        math.isclose(row["saliency"], expected, rel_tol=0, abs_tol=1e-6)
        for row, expected in zip(rows, [0.9, 0.8, 0.7])
    )
    assert math.isclose(rows[0]["clip_high"], 0.2, rel_tol=0, abs_tol=1e-6)
    assert math.isclose(rows[1]["multiplier"], 2.0, rel_tol=0, abs_tol=1e-6)
    assert rows[2]["question"] == "q1"


def test_move_paddings_to_right_vectorized_preserves_action_tokens():
    input_ids = torch.tensor([[0, 0, 11, 12], [0, 21, 22, 23]])
    attention_mask = torch.tensor([[0, 0, 1, 1], [0, 1, 1, 1]])
    sequence_ids = torch.tensor(
        [
            [0, 0, 11, 12, 31, 32],
            [0, 0, 11, 12, 41, 42],
            [0, 21, 22, 23, 51, 52],
            [0, 21, 22, 23, 61, 62],
        ]
    )

    new_sequence_ids, action_mask = move_paddings_to_right(
        input_ids=input_ids,
        attention_mask=attention_mask,
        sequence_ids=sequence_ids,
        pad_token_id=0,
    )

    assert torch.equal(
        new_sequence_ids,
        torch.tensor(
            [
                [11, 12, 31, 32, 0],
                [11, 12, 41, 42, 0],
                [21, 22, 23, 51, 52],
                [21, 22, 23, 61, 62],
            ]
        ),
    )
    assert torch.equal(
        new_sequence_ids[:, 1:][action_mask],
        torch.tensor([31, 32, 41, 42, 51, 52, 61, 62]),
    )


class FakeKeyProjection(nn.Module):
    def __init__(self, scale: float) -> None:
        super().__init__()
        self.scale = scale

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states * self.scale


class FakeSelfAttention(nn.Module):
    def __init__(self, scale: float) -> None:
        super().__init__()
        self.k_proj = FakeKeyProjection(scale)


class FakeDecoderLayer(nn.Module):
    def __init__(self, scale: float) -> None:
        super().__init__()
        self.self_attn = FakeSelfAttention(scale)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.self_attn.k_proj(hidden_states)


class FakeInnerModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([FakeDecoderLayer(1.0), FakeDecoderLayer(2.0)])


class FakeReferenceModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = FakeInnerModel()

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, use_cache: bool = False):
        del attention_mask, use_cache
        hidden_states = input_ids.float().unsqueeze(-1).repeat(1, 1, 2)
        for layer in self.model.layers:
            layer(hidden_states)
        return {"logits": torch.zeros(input_ids.shape[0], input_ids.shape[1], 10)}


class FakeCallableInnerModel(FakeInnerModel):
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, use_cache: bool = False):
        del attention_mask, use_cache
        hidden_states = input_ids.float().unsqueeze(-1).repeat(1, 1, 2)
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return {"last_hidden_state": hidden_states}


class FakeBackboneReferenceModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = FakeCallableInnerModel()
        self.full_forward_calls = 0

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, use_cache: bool = False):
        del attention_mask, use_cache
        self.full_forward_calls += 1
        return {"logits": torch.zeros(input_ids.shape[0], input_ids.shape[1], 10)}


def test_kv_norm_saliency_hook_uses_top_decoder_layers():
    sequence_ids = torch.tensor([[1, 2, 3]])
    logits, raw_saliency = extract_kv_norm_saliency(
        model=FakeReferenceModel(),
        token_batch=SequenceTokenBatch(
            sequence_ids=sequence_ids,
            attention_mask=torch.ones_like(sequence_ids),
        ),
        top_layers=1,
    )

    expected = (sequence_ids.to(dtype=torch.bfloat16) * 2.0).unsqueeze(-1).repeat(1, 1, 2).norm(dim=-1)
    assert logits.shape == (1, 3, 10)
    assert raw_saliency.values.dtype == torch.bfloat16
    assert torch.allclose(raw_saliency.values, expected, atol=BF16_ATOL)


def test_reference_policy_features_return_log_probs_and_token_thresholds():
    sequence_ids = torch.tensor([[1, 2, 3]])
    features = compute_reference_policy_features(
        ReferencePolicyFeatureRequest(
            model=FakeReferenceModel(),
            sequence_ids=sequence_ids,
            attention_mask=torch.ones_like(sequence_ids),
            action_mask=torch.tensor([[True, True]]),
            clip_high=0.28,
            logits_minibatch_size=1,
            top_layers=1,
            min_multiplier=0.0,
            max_multiplier=None,
            eps=1e-8,
            saliency_method="kv_norm",
        )
    )

    assert features.log_probs.shape == (1, 2)
    assert features.token_clip_thresholds.values.shape == (1, 2)


def test_reference_policy_features_can_skip_log_probs_when_kl_is_disabled():
    sequence_ids = torch.tensor([[1, 2, 3]])
    features = compute_reference_policy_features(
        ReferencePolicyFeatureRequest(
            model=FakeReferenceModel(),
            sequence_ids=sequence_ids,
            attention_mask=torch.ones_like(sequence_ids),
            action_mask=torch.tensor([[True, True]]),
            clip_high=0.28,
            logits_minibatch_size=1,
            top_layers=1,
            min_multiplier=0.0,
            max_multiplier=None,
            eps=1e-8,
            return_log_probs=False,
            saliency_method="kv_norm",
        )
    )

    assert features.log_probs is None
    assert features.token_clip_thresholds.values.shape == (1, 2)
    assert features.token_clip_thresholds.values.dtype == torch.bfloat16
    assert torch.allclose(features.token_clip_thresholds.values.mean(), bf16_tensor(0.28), atol=BF16_ATOL)


def test_reference_policy_features_uses_backbone_when_log_probs_are_skipped():
    sequence_ids = torch.tensor([[1, 2, 3]])
    model = FakeBackboneReferenceModel()

    features = compute_reference_policy_features(
        ReferencePolicyFeatureRequest(
            model=model,
            sequence_ids=sequence_ids,
            attention_mask=torch.ones_like(sequence_ids),
            action_mask=torch.tensor([[True, True]]),
            clip_high=0.28,
            logits_minibatch_size=1,
            top_layers=1,
            min_multiplier=0.0,
            max_multiplier=None,
            eps=1e-8,
            return_log_probs=False,
            saliency_method="kv_norm",
        )
    )

    assert model.full_forward_calls == 0
    assert features.log_probs is None
    assert features.token_clip_thresholds.values.shape == (1, 2)
