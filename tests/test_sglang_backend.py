from pathlib import Path

import torch
from transformers import GenerationConfig

from curious.sampling.sglang_backend import (
    SGLangGenerationBackend,
    _extract_output_token_ids,
    build_sampled_response_tensors,
)


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 99

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [int(token) for token in text.split()]

    def save_pretrained(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        (path / "tokenizer.json").write_text("{}")


def test_build_sampled_response_tensors_aligns_actions_to_generated_tokens():
    output = build_sampled_response_tensors(
        tokenizer=FakeTokenizer(),
        prompt_ids=[[11, 12, 13], [21, 22]],
        output_token_ids=[[31, 32], [41]],
        completions=["31 32", "41"],
        device=torch.device("cpu"),
    )

    assert torch.equal(
        output["sequence_ids"],
        torch.tensor(
            [
                [11, 12, 13, 31, 32],
                [21, 22, 41, 0, 0],
            ]
        ),
    )
    assert torch.equal(
        output["action_mask"],
        torch.tensor(
            [
                [False, False, True, True],
                [False, True, False, False],
            ]
        ),
    )
    assert torch.equal(output["sequence_ids"][:, 1:][output["action_mask"]], torch.tensor([31, 32, 41]))


def test_extract_output_token_ids_accepts_sglang_response_shapes():
    assert _extract_output_token_ids({"output_ids": [1, 2]}) == [1, 2]
    assert _extract_output_token_ids({"meta_info": {"output_token_ids": [3, 4]}}) == [3, 4]
    assert _extract_output_token_ids({"text": "5 6"}) is None


class FakeSGLangBackend(SGLangGenerationBackend):
    def __init__(self) -> None:
        self.tokenizer = FakeTokenizer()
        self.requests = []

    def _request(self, method: str, path: str, payload: dict | None = None):
        self.requests.append((method, path, payload))
        return [
            {"text": "31", "meta_info": {"output_token_ids": [31]}},
            {"text": "32", "output_ids": [32]},
            {"text": "41"},
            {"text": "42", "meta_info": {"token_ids": [42]}},
        ]


def test_sglang_sample_responses_replicates_trimmed_prompts_and_parses_outputs():
    backend = FakeSGLangBackend()
    generation_config = GenerationConfig(
        num_return_sequences=2,
        max_new_tokens=8,
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        repetition_penalty=1.0,
        do_sample=True,
    )

    output = backend.sample_responses(
        batch_inputs={
            "input_ids": torch.tensor([[0, 11, 12], [21, 22, 23]]),
            "attention_mask": torch.tensor([[0, 1, 1], [1, 1, 1]]),
        },
        generation_config=generation_config,
    )

    _, _, payload = backend.requests[0]
    assert payload["input_ids"] == [[11, 12], [11, 12], [21, 22, 23], [21, 22, 23]]
    assert payload["sampling_params"]["stop_token_ids"] == [99]
    assert "random_seed" not in payload["sampling_params"]
    assert output["completions"] == ["31", "32", "41", "42"]
    assert torch.equal(output["sequence_ids"][:, 1:][output["action_mask"]], torch.tensor([31, 32, 41, 42]))


class FakeModel:
    def save_pretrained(self, path: Path, safe_serialization: bool = True) -> None:
        assert safe_serialization is True
        path.mkdir(parents=True, exist_ok=True)
        (path / "model.safetensors").write_text("weights")


class FakeSyncBackend(SGLangGenerationBackend):
    def __init__(self, weight_sync_dir: Path) -> None:
        self.weight_sync_dir = weight_sync_dir
        self.weight_sync_dir.mkdir(parents=True, exist_ok=True)
        self._last_synced_path = None
        self.requests = []

    def _request(self, method: str, path: str, payload: dict | None = None):
        self.requests.append((method, path, payload))
        return {"success": True}


def test_sglang_weight_sync_uses_disk_schema_and_keeps_only_latest_dir(tmp_path):
    backend = FakeSyncBackend(tmp_path)

    backend.sync_weights_from_model(FakeModel(), FakeTokenizer(), step=1)
    first_path = tmp_path / "step_00000001"
    assert backend.requests[-1] == ("POST", "/update_weights_from_disk", {"model_path": str(first_path)})
    assert first_path.exists()

    backend.sync_weights_from_model(FakeModel(), FakeTokenizer(), step=2)
    second_path = tmp_path / "step_00000002"
    assert backend.requests[-1] == ("POST", "/update_weights_from_disk", {"model_path": str(second_path)})
    assert not first_path.exists()
    assert second_path.exists()
