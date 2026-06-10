import atexit
from dataclasses import dataclass
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import torch
from transformers import GenerationConfig, PreTrainedModel, PreTrainedTokenizer


@dataclass(frozen=True)
class WeightSyncPlan:
    target_path: Path
    tmp_path: Path
    static_path: Path


class SGLangGenerationBackend:
    def __init__(
        self,
        model_path: str,
        tokenizer: PreTrainedTokenizer,
        host: str,
        port: int,
        attention_backend: str,
        dtype: str,
        mem_fraction_static: float,
        log_level: str,
        startup_timeout: int,
        weight_sync_dir: str,
        max_running_requests: int | None = None,
        chunked_prefill_size: int | None = None,
        request_batch_size: int | None = None,
    ) -> None:
        self.model_path = model_path
        self.tokenizer = tokenizer
        self.host = host
        self.port = port
        self.base_url = f"http://{host}:{port}"
        self.weight_sync_dir = Path(weight_sync_dir)
        self.weight_sync_dir.mkdir(parents=True, exist_ok=True)
        self._last_synced_path: Path | None = None
        self._static_sync_path = self.weight_sync_dir / "_static"
        self.request_batch_size = request_batch_size

        command = [
            sys.executable,
            "-m",
            "sglang.launch_server",
            "--model-path",
            model_path,
            "--host",
            host,
            "--port",
            str(port),
            "--attention-backend",
            attention_backend,
            "--dtype",
            dtype,
            "--mem-fraction-static",
            str(mem_fraction_static),
            "--log-level",
            log_level,
        ]
        if max_running_requests is not None:
            command.extend(["--max-running-requests", str(max_running_requests)])
        if chunked_prefill_size is not None:
            command.extend(["--chunked-prefill-size", str(chunked_prefill_size)])

        print("Starting SGLang rollout server:", " ".join(command), flush=True)
        self.process = subprocess.Popen(command)
        atexit.register(self.close)
        self._wait_until_ready(startup_timeout)

    def plan_weight_sync(self, step: int) -> WeightSyncPlan:
        static_path = getattr(self, "_static_sync_path", self.weight_sync_dir / "_static")
        return WeightSyncPlan(
            target_path=self.weight_sync_dir / f"step_{step:08d}",
            tmp_path=self.weight_sync_dir / f".step_{step:08d}.tmp",
            static_path=static_path,
        )

    def _ensure_static_sync_artifacts(self, tokenizer: PreTrainedTokenizer, static_path: Path) -> None:
        if static_path.exists() and any(static_path.iterdir()):
            return
        tmp_static_path = static_path.with_name(f".{static_path.name}.tmp")
        if tmp_static_path.exists():
            shutil.rmtree(tmp_static_path)
        tmp_static_path.mkdir(parents=True)
        tokenizer.save_pretrained(tmp_static_path)
        if static_path.exists():
            shutil.rmtree(static_path)
        tmp_static_path.rename(static_path)

    def _link_static_sync_artifacts(self, static_path: Path, target_path: Path) -> None:
        for source in static_path.iterdir():
            destination = target_path / source.name
            if destination.exists():
                continue
            if source.is_dir():
                shutil.copytree(source, destination)
                continue
            try:
                os.link(source, destination)
            except OSError:
                shutil.copy2(source, destination)

    def _wait_until_ready(self, timeout_seconds: int) -> None:
        deadline = time.time() + timeout_seconds
        last_error = ""
        while time.time() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(f"SGLang server exited early with code {self.process.returncode}: {last_error}")
            try:
                self._request("GET", "/health")
                return
            except Exception as exc:
                last_error = str(exc)
                time.sleep(2)
        raise TimeoutError(f"SGLang server did not become healthy within {timeout_seconds}s: {last_error}")

    def _request(self, method: str, path: str, payload: dict[str, Any] | None = None) -> Any:
        data = None
        headers = {}
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=data,
            headers=headers,
            method=method,
        )
        try:
            with urllib.request.urlopen(request, timeout=600) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"SGLang {method} {path} failed with HTTP {exc.code}: {body}") from exc
        if not body:
            return None
        return json.loads(body)

    def sample_responses(
        self,
        batch_inputs: dict[str, torch.Tensor],
        generation_config: GenerationConfig,
        seed: int = 42,
    ) -> dict[str, torch.Tensor]:
        input_ids = batch_inputs["input_ids"]
        attention_mask = batch_inputs["attention_mask"]
        prompt_ids: list[list[int]] = []
        for ids, mask in zip(input_ids, attention_mask):
            trimmed_ids = ids[mask.bool()].tolist()
            for _ in range(generation_config.num_return_sequences):
                prompt_ids.append(trimmed_ids)

        stop_token_ids = []
        if self.tokenizer.eos_token_id is not None:
            stop_token_ids.append(self.tokenizer.eos_token_id)

        sampling_params = {
            "max_new_tokens": generation_config.max_new_tokens,
            "temperature": generation_config.temperature,
            "top_p": generation_config.top_p,
            "top_k": generation_config.top_k,
            "repetition_penalty": generation_config.repetition_penalty,
            "skip_special_tokens": True,
        }
        if stop_token_ids:
            sampling_params["stop_token_ids"] = stop_token_ids

        del seed
        response_items = []
        request_batch_size = getattr(self, "request_batch_size", None) or len(prompt_ids)
        if request_batch_size <= 0:
            raise ValueError("request_batch_size must be positive when set")

        for start in range(0, len(prompt_ids), request_batch_size):
            chunk_prompt_ids = prompt_ids[start:start + request_batch_size]
            payload = {
                "input_ids": chunk_prompt_ids,
                "sampling_params": sampling_params,
            }
            response = self._request("POST", "/generate", payload)
            chunk_items = response if isinstance(response, list) else [response]
            if len(chunk_items) != len(chunk_prompt_ids):
                raise RuntimeError(
                    f"SGLang returned {len(chunk_items)} responses for {len(chunk_prompt_ids)} prompts"
                )
            response_items.extend(chunk_items)

        if len(response_items) != len(prompt_ids):
            raise RuntimeError(f"SGLang returned {len(response_items)} responses for {len(prompt_ids)} prompts")

        output_token_ids: list[list[int]] = []
        completions: list[str] = []
        for item in response_items:
            text = _extract_text(item)
            output_ids = _extract_output_token_ids(item)
            if output_ids is None:
                output_ids = self.tokenizer.encode(text, add_special_tokens=False)
            output_token_ids.append(output_ids)
            completions.append(text)

        return build_sampled_response_tensors(
            tokenizer=self.tokenizer,
            prompt_ids=prompt_ids,
            output_token_ids=output_token_ids,
            completions=completions,
            device=batch_inputs["input_ids"].device,
        )

    def sync_weights_from_model(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        step: int,
    ) -> None:
        model_to_save = model.module if hasattr(model, "module") else model
        sync_plan = self.plan_weight_sync(step)
        target_path = sync_plan.target_path
        tmp_path = sync_plan.tmp_path
        if tmp_path.exists():
            shutil.rmtree(tmp_path)
        tmp_path.mkdir(parents=True)

        self._ensure_static_sync_artifacts(tokenizer, sync_plan.static_path)
        model_to_save.save_pretrained(tmp_path, safe_serialization=True)
        self._link_static_sync_artifacts(sync_plan.static_path, tmp_path)
        if target_path.exists():
            shutil.rmtree(target_path)
        tmp_path.rename(target_path)

        response = self._request(
            "POST",
            "/update_weights_from_disk",
            {"model_path": str(target_path)},
        )
        if not isinstance(response, dict) or response.get("success") is not True:
            raise RuntimeError(f"SGLang weight sync failed: {response}")

        previous_path = self._last_synced_path
        self._last_synced_path = target_path
        if previous_path is not None and previous_path != target_path and previous_path.exists():
            shutil.rmtree(previous_path)

    def close(self) -> None:
        process = getattr(self, "process", None)
        if process is None or process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=20)


def build_sampled_response_tensors(
    tokenizer: PreTrainedTokenizer,
    prompt_ids: list[list[int]],
    output_token_ids: list[list[int]],
    completions: list[str],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    if len(prompt_ids) != len(output_token_ids):
        raise ValueError("prompt_ids and output_token_ids must have the same length")

    pad_token_id = tokenizer.pad_token_id
    full_sequences = [prompt + output for prompt, output in zip(prompt_ids, output_token_ids)]
    max_length = max(len(sequence) for sequence in full_sequences)
    sequence_ids = torch.full(
        (len(full_sequences), max_length),
        fill_value=pad_token_id,
        dtype=torch.long,
        device=device,
    )
    action_mask = torch.zeros(
        (len(full_sequences), max_length - 1),
        dtype=torch.bool,
        device=device,
    )
    for row_idx, (prompt, output, sequence) in enumerate(zip(prompt_ids, output_token_ids, full_sequences)):
        sequence_length = len(sequence)
        sequence_ids[row_idx, :sequence_length] = torch.tensor(sequence, dtype=torch.long, device=device)
        if output:
            start = max(len(prompt) - 1, 0)
            action_mask[row_idx, start:start + len(output)] = True

    return {
        "num_samples": len(full_sequences),
        "input_ids": sequence_ids,
        "sequence_ids": sequence_ids,
        "action_mask": action_mask,
        "completions": completions,
    }


def _extract_text(item: Any) -> str:
    if not isinstance(item, dict):
        return ""
    text = item.get("text")
    if isinstance(text, str):
        return text
    return ""


def _extract_output_token_ids(item: Any) -> list[int] | None:
    if not isinstance(item, dict):
        return None

    for key in ("output_ids", "output_token_ids", "token_ids"):
        value = item.get(key)
        if _is_int_list(value):
            return list(value)

    meta_info = item.get("meta_info")
    if isinstance(meta_info, dict):
        for key in ("output_ids", "output_token_ids", "token_ids"):
            value = meta_info.get(key)
            if _is_int_list(value):
                return list(value)

    return None


def _is_int_list(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(token_id, int) for token_id in value)
