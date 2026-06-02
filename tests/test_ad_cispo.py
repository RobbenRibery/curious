import math

import torch
import torch.nn as nn

from curious.policy_gradient.ad_cispo import (
    ADCispoThresholdConfig,
    RawTokenSaliency,
    ReferencePolicyFeatureRequest,
    SequenceTokenBatch,
    align_saliency_to_actions,
    compute_reference_policy_features,
    compute_token_clip_thresholds,
    extract_kv_norm_saliency,
)
from curious.policy_gradient.loss import ActorLoss
from curious.replay.experience import Experience, join_experience_batch, split_experience_batch
from curious.sampling.sampling import compute_group_advantages, decode_action_tokens


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
    assert torch.allclose(masked_thresholds.mean(), torch.tensor(0.28), atol=1e-6)
    assert torch.allclose(thresholds.values[~action_mask], torch.full_like(thresholds.values[~action_mask], 0.28))
    assert math.isclose(stats.clip_mean, 0.28, rel_tol=0, abs_tol=1e-6)
    assert math.isclose(stats.multiplier_mean, 1.0, rel_tol=0, abs_tol=1e-6)


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

    assert torch.allclose(thresholds.values[action_mask].mean(), torch.tensor(0.4), atol=1e-6)
    assert thresholds.multipliers[action_mask].min() >= 0.5
    assert thresholds.multipliers[action_mask].max() <= 2.0


def test_action_saliency_aligns_to_next_token_log_prob_positions():
    raw_saliency = RawTokenSaliency(values=torch.tensor([[10.0, 20.0, 30.0, 40.0]]))
    action_mask = torch.tensor([[False, True, True]])

    action_saliency = align_saliency_to_actions(raw_saliency, action_mask)

    assert torch.equal(action_saliency.values, torch.tensor([[20.0, 30.0, 40.0]]))


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
    expected_kl = ((log_probs_ref - log_probs).exp() - (log_probs_ref - log_probs) - 1).mean()

    assert torch.allclose(mean_kl, expected_kl, atol=1e-6)


def test_group_advantages_use_population_std_for_grpo_normalization():
    returns = torch.tensor([[1.0, 2.0, 3.0]])

    advantages = compute_group_advantages(returns=returns, normalize=True)

    assert torch.allclose(advantages, torch.tensor([[-1.2247, 0.0, 1.2247]]), atol=1e-4)


class FakeTokenizer:
    def decode(self, token_ids: torch.Tensor, skip_special_tokens: bool = True) -> str:
        del skip_special_tokens
        return " ".join(str(x.item()) for x in token_ids)


def test_decode_action_tokens_uses_action_mask_not_fixed_prompt_width():
    sequence_ids = torch.tensor([[0, 0, 11, 12, 21, 22, 99]])
    action_mask = torch.tensor([[False, False, False, True, True, False]])

    completions = decode_action_tokens(FakeTokenizer(), sequence_ids, action_mask)

    assert completions == ["21 22"]


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

    expected = (sequence_ids.float() * 2.0).unsqueeze(-1).repeat(1, 1, 2).norm(dim=-1)
    assert logits.shape == (1, 3, 10)
    assert torch.allclose(raw_saliency.values, expected)


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
        )
    )

    assert features.log_probs.shape == (1, 2)
    assert features.token_clip_thresholds.values.shape == (1, 2)
    assert torch.allclose(features.token_clip_thresholds.values.mean(), torch.tensor(0.28), atol=1e-6)
