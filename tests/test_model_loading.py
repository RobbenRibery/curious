import inspect

import pytest

from curious.utils import utils


def test_liger_model_loading_defaults_to_flash_attention_3():
    default = inspect.signature(utils.load_model_tokenizer).parameters["liger_attn_implementation"].default

    assert default == "flash_attention_3"


def test_flash_attention_3_preflight_requires_cuda(monkeypatch):
    monkeypatch.setattr(utils.torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="requires CUDA"):
        utils.validate_flash_attention_3_runtime()


def test_flash_attention_3_preflight_requires_hopper_or_newer(monkeypatch):
    monkeypatch.setattr(utils.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(utils.torch.cuda, "get_device_capability", lambda: (8, 0))

    with pytest.raises(RuntimeError, match="compute capability >= 9.0"):
        utils.validate_flash_attention_3_runtime()
