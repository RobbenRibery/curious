from curious.config import BaseConfig, RLConfig
import pytest

from curious.train.training_setup import needs_reference_policy, normalize_rl_config_for_objective


def test_train_model_compile_defaults_off_for_h100_training_path():
    assert not BaseConfig().compile_train_model


def test_zero_kl_grpo_without_ad_cispo_does_not_need_reference_policy():
    rl_config = RLConfig(kl_weight=0.0, use_ad_cispo=False)

    assert not needs_reference_policy(rl_config)


def test_kl_regularization_needs_reference_policy():
    rl_config = RLConfig(kl_weight=0.01, use_ad_cispo=False)

    assert needs_reference_policy(rl_config)


def test_ad_cispo_without_kl_uses_target_policy_saliency_without_reference_policy():
    rl_config = RLConfig(kl_weight=0.0, use_ad_cispo=True)

    assert not needs_reference_policy(rl_config)


def test_cispo_normalization_forces_zero_kl_and_token_level_loss():
    rl_config = RLConfig(kl_weight=0.01, use_cispo_loss=True, use_token_level_loss=False)

    normalize_rl_config_for_objective(rl_config)

    assert rl_config.kl_weight == 0.0
    assert rl_config.use_token_level_loss


def test_ad_cispo_requires_cispo_loss_path():
    rl_config = RLConfig(kl_weight=0.0, use_ad_cispo=True, use_cispo_loss=False)

    with pytest.raises(ValueError, match="AD-CISPO requires use_cispo_loss=True"):
        normalize_rl_config_for_objective(rl_config)
