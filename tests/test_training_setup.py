from curious.config import RLConfig
from curious.train.training_setup import needs_reference_policy


def test_zero_kl_grpo_without_ad_cispo_does_not_need_reference_policy():
    rl_config = RLConfig(kl_weight=0.0, use_ad_cispo=False)

    assert not needs_reference_policy(rl_config)


def test_kl_regularization_needs_reference_policy():
    rl_config = RLConfig(kl_weight=0.01, use_ad_cispo=False)

    assert needs_reference_policy(rl_config)


def test_ad_cispo_needs_reference_policy_even_without_kl_regularization():
    rl_config = RLConfig(kl_weight=0.0, use_ad_cispo=True)

    assert needs_reference_policy(rl_config)
