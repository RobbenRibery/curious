import argparse
import html
import json
import os
import sys
from pathlib import Path
from typing import Any

import modal


APP_NAME = "curious-saliency-comparison"
ARTIFACTS_DIR = "/modal-artifacts"
ARTIFACTS_VOLUME_NAME = "curious-training-artifacts"
DEFAULT_GPU = "H100"
DEFAULT_TIMEOUT_SECONDS = 2 * 60 * 60
LOCAL_SECRET_ENV_KEYS = (
    "WANDB_API_KEY",
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
)

DEFAULT_MODEL = "Qwen/Qwen3-1.7B"
DEFAULT_OUTPUT_DIR = "experiments/saliency_reasoning_traces"

REASONING_PROMPTS = [
    "Solve this carefully. A train leaves at 2:15 PM traveling 54 mph. Another train leaves the same station at 3:00 PM on the same track traveling 72 mph. When does the second train catch the first? Show your reasoning, revisit assumptions, and self-check the answer.",
    "Solve this carefully. Three notebooks and two pens cost $13.50. Five notebooks and four pens cost $24.50. Find the price of one notebook and one pen. Use deductive reasoning and verify the system.",
    "Solve this carefully. A rectangle has perimeter 74 cm. Its length is 5 cm more than twice its width. Find both dimensions. Reflect on each algebra step and check the result.",
    "Solve this carefully. A jar has red, blue, and green marbles in the ratio 3:4:5. After adding 8 green marbles, the ratio of blue to green becomes 2:3. How many red marbles are in the jar originally? Revisit the setup if needed.",
    "Solve this carefully. The average of five numbers is 18. Removing one number makes the average of the remaining four equal to 16. What number was removed? Explain the invariant and check it.",
    "Solve this carefully. A worker can finish a job in 12 hours and another can finish it in 18 hours. They work together for 4 hours, then the first worker leaves. How long does the second worker need to finish the job? Include a sanity check.",
]

CURATED_REASONING_COMPLETIONS = [
    """Let the first train's head start be the distance it covers from 2:15 to 3:00. That is 45 minutes = 0.75 hours, so the head start is 54 * 0.75 = 40.5 miles.

Now compare speeds after 3:00. The second train gains at 72 - 54 = 18 mph. Time to close 40.5 miles is 40.5 / 18 = 2.25 hours.

Revisiting the setup: both trains are on the same track and the second starts later from the same station, so relative speed is the right invariant. 2.25 hours after 3:00 PM is 5:15 PM. Answer: 5:15 PM.""",
    """Let n be the notebook price and p be the pen price. The equations are 3n + 2p = 13.50 and 5n + 4p = 24.50.

I want to eliminate p, so double the first equation: 6n + 4p = 27.00. Subtract the second equation from this: n = 2.50.

Substitute back: 3(2.50) + 2p = 13.50, so 7.50 + 2p = 13.50 and p = 3.00.

Check: 5(2.50) + 4(3.00) = 12.50 + 12.00 = 24.50. Answer: notebook $2.50, pen $3.00.""",
    """Let the width be w. The length is 2w + 5. The perimeter equation is 2(length + width) = 74, so length + width = 37.

Substitute: (2w + 5) + w = 37, hence 3w + 5 = 37, so 3w = 32 and w = 32/3 cm.

Then length = 2(32/3) + 5 = 64/3 + 15/3 = 79/3 cm.

Self-check: length + width = 79/3 + 32/3 = 111/3 = 37, so the perimeter is 74. Answer: width 32/3 cm and length 79/3 cm.""",
    """Use the ratio as actual counts scaled by x: red = 3x, blue = 4x, green = 5x.

After adding 8 green marbles, blue:green becomes 2:3, so 4x / (5x + 8) = 2/3. Cross multiply: 12x = 10x + 16, so 2x = 16 and x = 8.

The red marbles originally are 3x = 24.

Check: blue = 32 and green after adding is 40 + 8 = 48. The ratio 32:48 reduces to 2:3. Answer: 24 red marbles.""",
    """The invariant is total sum. Five numbers average 18, so their total is 5 * 18 = 90.

After removing one number, four numbers average 16, so the remaining total is 4 * 16 = 64.

The removed number is the difference: 90 - 64 = 26.

Self-check: if 26 is removed from 90, the remaining sum is 64, and 64/4 = 16. Answer: 26.""",
    """The first worker's rate is 1/12 job per hour. The second worker's rate is 1/18 job per hour. Together they work at 1/12 + 1/18 = 3/36 + 2/36 = 5/36 job per hour.

In 4 hours they complete 4 * 5/36 = 20/36 = 5/9 of the job. The remaining work is 4/9.

Only the second worker remains, at 1/18 job per hour. Time needed is (4/9) / (1/18) = (4/9) * 18 = 8 hours.

Sanity check: after the first worker leaves, less than half the job remains, and the second worker alone needs 18 hours for a full job, so 8 hours is plausible. Answer: 8 hours.""",
]

app = modal.App(APP_NAME)
artifacts_volume = modal.Volume.from_name(ARTIFACTS_VOLUME_NAME, create_if_missing=True)

analysis_image = (
    modal.Image.from_registry(
        "nvidia/cuda:13.0.2-devel-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install(
        "build-essential",
        "git",
        "ninja-build",
    )
    .uv_sync(uv_version="0.9.18")
    .workdir("/root")
    .env(
        {
            "TOKENIZERS_PARALLELISM": "true",
            "NVIDIA_TF32_OVERRIDE": "0",
            "TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS": "1",
            "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:128,expandable_segments:True",
            "HF_HOME": f"{ARTIFACTS_DIR}/hf",
            "WANDB_DIR": f"{ARTIFACTS_DIR}/wandb",
            "WANDB_CACHE_DIR": f"{ARTIFACTS_DIR}/wandb_cache",
        }
    )
    .add_local_python_source("curious")
)


def _dotenv_values(dotenv_path: str) -> dict[str, str]:
    path = Path(dotenv_path)
    if not path.exists():
        return {}

    values: dict[str, str] = {}
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if value and value[0] in {"'", '"'} and value[-1:] == value[0]:
            value = value[1:-1]
        if key:
            values[key] = value
    return values


def _local_secrets(dotenv_path: str | None, include_env_secret: bool, secret_names: list[str]) -> list[modal.Secret]:
    secrets: list[modal.Secret] = []
    if dotenv_path is not None:
        dotenv_secret = _dotenv_values(dotenv_path)
        if dotenv_secret:
            secrets.append(modal.Secret.from_dict(dotenv_secret))

    if include_env_secret:
        env_secret = {
            key: value
            for key in LOCAL_SECRET_ENV_KEYS
            if (value := os.environ.get(key))
        }
        if env_secret:
            secrets.append(modal.Secret.from_dict(env_secret))

    secrets.extend(modal.Secret.from_name(name) for name in secret_names)
    return secrets


def _build_prompt(question: str) -> str:
    return (
        "You are solving a math problem. Write a compact but sophisticated reasoning trace: "
        "state assumptions, reason deductively, revisit any fragile step, and end with a clear answer.\n\n"
        f"Problem: {question}\n\nReasoning trace:"
    )


def _chat_prompt(tokenizer: Any, question: str) -> str:
    conversation = [
        {
            "role": "system",
            "content": (
                "You produce concise mathematical reasoning traces with explicit self-checks. "
                "Do not skip algebraic setup."
            ),
        },
        {"role": "user", "content": question},
    ]
    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if callable(apply_chat_template):
        try:
            return apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
        except TypeError:
            return apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    return _build_prompt(question)


def _ensure_pad_token(tokenizer: Any) -> int:
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    if tokenizer.pad_token_id is None:
        raise RuntimeError("Tokenizer must expose a pad, eos, or unk token for saliency batching.")
    return int(tokenizer.pad_token_id)


def _pad_sequences(
    sequences: list[list[int]],
    prompt_lengths: list[int],
    pad_token_id: int,
    device: Any,
) -> tuple[Any, Any, Any]:
    import torch

    max_len = max(len(sequence) for sequence in sequences)
    sequence_ids = torch.full((len(sequences), max_len), pad_token_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros_like(sequence_ids)
    action_mask = torch.zeros((len(sequences), max_len - 1), dtype=torch.bool, device=device)
    for row_idx, (sequence, prompt_length) in enumerate(zip(sequences, prompt_lengths)):
        length = len(sequence)
        sequence_ids[row_idx, :length] = torch.tensor(sequence, dtype=torch.long, device=device)
        attention_mask[row_idx, :length] = 1
        if length > prompt_length:
            action_mask[row_idx, prompt_length - 1 : length - 1] = True
    return sequence_ids, attention_mask, action_mask


def _clean_token_text(text: str) -> str:
    if text == "\n":
        return "\\n"
    if text == "\t":
        return "\\t"
    if text == " ":
        return "space"
    return text.replace("\n", "\\n").replace("\t", "\\t")


def _token_rows(tokenizer: Any, sequence_ids: Any, action_mask: Any) -> list[list[dict[str, Any]]]:
    rows: list[list[dict[str, Any]]] = []
    sequence_ids_cpu = sequence_ids.detach().cpu()
    action_mask_cpu = action_mask.detach().cpu()
    for row_idx in range(sequence_ids_cpu.shape[0]):
        row: list[dict[str, Any]] = []
        active_positions = action_mask_cpu[row_idx].nonzero(as_tuple=False).flatten().tolist()
        for action_pos in active_positions:
            token_index = action_pos + 1
            token_id = int(sequence_ids_cpu[row_idx, token_index].item())
            text = tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
            row.append(
                {
                    "action_pos": int(action_pos),
                    "token_index": int(token_index),
                    "token_id": token_id,
                    "text": _clean_token_text(text),
                }
            )
        rows.append(row)
    return rows


def _scores_for_tokens(scores: Any, action_mask: Any) -> list[list[float]]:
    score_cpu = scores.detach().to(dtype=__import__("torch").float32).cpu()
    mask_cpu = action_mask.detach().cpu()
    rows: list[list[float]] = []
    for row_idx in range(score_cpu.shape[0]):
        rows.append([float(score_cpu[row_idx, action_pos].item()) for action_pos in mask_cpu[row_idx].nonzero(as_tuple=False).flatten()])
    return rows


def _normalize(values: list[float]) -> list[float]:
    finite_values = [value for value in values if value == value and value not in {float("inf"), float("-inf")}]
    if not finite_values:
        return [0.0 for _ in values]
    min_value = min(finite_values)
    max_value = max(finite_values)
    if max_value <= min_value:
        return [0.0 for _ in values]
    return [max(0.0, min(1.0, (value - min_value) / (max_value - min_value))) for value in values]


def _top_tokens(token_rows: list[list[dict[str, Any]]], score_rows: list[list[float]], top_k: int = 12) -> list[list[dict[str, Any]]]:
    summaries: list[list[dict[str, Any]]] = []
    for tokens, scores in zip(token_rows, score_rows):
        ranked = sorted(
            (
                {
                    "text": token["text"],
                    "token_id": token["token_id"],
                    "token_index": token["token_index"],
                    "score": score,
                }
                for token, score in zip(tokens, scores)
            ),
            key=lambda item: item["score"],
            reverse=True,
        )
        summaries.append(ranked[:top_k])
    return summaries


def _method_heatmap_html(
    *,
    title: str,
    method_name: str,
    prompts: list[str],
    completions: list[str],
    token_rows: list[list[dict[str, Any]]],
    score_rows: list[list[float]],
    top_tokens: list[list[dict[str, Any]]],
    color: tuple[int, int, int],
) -> str:
    rows_html = []
    red, green, blue = color
    for trace_idx, (prompt, completion, tokens, scores, top_row) in enumerate(
        zip(prompts, completions, token_rows, score_rows, top_tokens)
    ):
        normalized = _normalize(scores)
        token_html = []
        for token, score, norm in zip(tokens, scores, normalized):
            alpha = 0.08 + 0.82 * norm
            token_html.append(
                "<span class='tok' style='background: rgba(%d, %d, %d, %.3f)' title='idx=%s id=%s raw=%.6g norm=%.3f'>%s</span>"
                % (
                    red,
                    green,
                    blue,
                    alpha,
                    token["token_index"],
                    token["token_id"],
                    score,
                    norm,
                    html.escape(token["text"]),
                )
            )
        top_html = ", ".join(
            f"{html.escape(item['text'])} ({item['score']:.4g})"
            for item in top_row
        )
        rows_html.append(
            f"""
            <section class="trace">
              <h2>Trace {trace_idx + 1}</h2>
              <details>
                <summary>Prompt and generated reasoning trace</summary>
                <p><strong>Prompt:</strong> {html.escape(prompt)}</p>
                <pre>{html.escape(completion)}</pre>
              </details>
              <p class="top"><strong>Top tokens:</strong> {top_html}</p>
              <div class="heatrow" aria-label="{html.escape(method_name)} trace {trace_idx + 1} heatmap">
                {''.join(token_html)}
              </div>
            </section>
            """
        )
    return _page_html(title=title, body="\n".join(rows_html))


def _head_to_head_html(
    *,
    prompts: list[str],
    completions: list[str],
    token_rows: list[list[dict[str, Any]]],
    attention_scores: list[list[float]],
    causal_scores: list[list[float]],
) -> str:
    rows_html = []
    for trace_idx, (prompt, completion, tokens, attention_row, causal_row) in enumerate(
        zip(prompts, completions, token_rows, attention_scores, causal_scores)
    ):
        method_rows = []
        for method_name, scores, color in (
            ("future_attention_in_degree", attention_row, (220, 113, 32)),
            ("causal_tangent", causal_row, (17, 128, 122)),
        ):
            red, green, blue = color
            normalized = _normalize(scores)
            token_html = []
            for token, score, norm in zip(tokens, scores, normalized):
                alpha = 0.08 + 0.82 * norm
                token_html.append(
                    "<span class='tok' style='background: rgba(%d, %d, %d, %.3f)' title='%s idx=%s id=%s raw=%.6g norm=%.3f'>%s</span>"
                    % (
                        red,
                        green,
                        blue,
                        alpha,
                        html.escape(method_name),
                        token["token_index"],
                        token["token_id"],
                        score,
                        norm,
                        html.escape(token["text"]),
                    )
                )
            method_rows.append(
                f"""
                <div class="method-row">
                  <div class="method-label">{html.escape(method_name)}</div>
                  <div class="heatrow">{''.join(token_html)}</div>
                </div>
                """
            )
        rows_html.append(
            f"""
            <section class="trace">
              <h2>Trace {trace_idx + 1}</h2>
              <details>
                <summary>Prompt and generated reasoning trace</summary>
                <p><strong>Prompt:</strong> {html.escape(prompt)}</p>
                <pre>{html.escape(completion)}</pre>
              </details>
              {''.join(method_rows)}
            </section>
            """
        )
    return _page_html(title="AD-CISPO Saliency Head-to-Head Heatmaps", body="\n".join(rows_html))


def _page_html(*, title: str, body: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.45;
      margin: 24px;
      color: #151515;
      background: #fafafa;
    }}
    h1 {{ font-size: 24px; margin-bottom: 8px; }}
    h2 {{ font-size: 17px; margin: 0 0 8px; }}
    pre {{
      white-space: pre-wrap;
      background: #fff;
      border: 1px solid #ddd;
      padding: 12px;
      overflow-x: auto;
    }}
    .trace {{
      margin: 22px 0;
      padding: 16px 0;
      border-top: 1px solid #d8d8d8;
    }}
    .top {{ color: #333; font-size: 13px; }}
    .method-row {{
      display: grid;
      grid-template-columns: 210px 1fr;
      gap: 12px;
      align-items: start;
      margin: 10px 0;
    }}
    .method-label {{
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      font-size: 12px;
      color: #333;
      padding-top: 3px;
    }}
    .heatrow {{
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      font-size: 13px;
      line-height: 2.05;
      overflow-wrap: anywhere;
    }}
    .tok {{
      display: inline-block;
      margin: 1px;
      padding: 1px 3px;
      border-radius: 3px;
      border: 1px solid rgba(0, 0, 0, 0.06);
      white-space: pre-wrap;
    }}
  </style>
</head>
<body>
  <h1>{html.escape(title)}</h1>
  <p>Each row uses the same generated reasoning trace and token mask. Color intensity is normalized within each trace and method.</p>
  {body}
</body>
</html>
"""


@app.function(
    image=analysis_image,
    gpu=DEFAULT_GPU,
    volumes={ARTIFACTS_DIR: artifacts_volume},
    cpu=16,
    memory=96 * 1024,
    timeout=DEFAULT_TIMEOUT_SECONDS,
)
def run_saliency_comparison(
    model_name: str,
    trace_source: str,
    num_traces: int,
    max_new_tokens: int,
    temperature: float,
    seed: int,
    top_layers: int,
    attention_block_size: int,
    apply_sink_guard: bool,
) -> dict[str, Any]:
    import torch

    from curious.policy_gradient.ad_cispo import (
        ADCispoThresholdConfig,
        RawTokenSaliency,
        SequenceTokenBatch,
        build_saliency_masks,
        compute_token_clip_thresholds,
        extract_causal_tangent_saliency,
        extract_future_attention_indegree_saliency,
    )
    from curious.policy_gradient.ad_cispo import collect_ad_cispo_sink_token_ids
    from curious.utils.utils import configure_bf16_precision, load_model_tokenizer

    configure_bf16_precision()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model, tokenizer = load_model_tokenizer(
        model_name_or_path=model_name,
        dtype_=torch.bfloat16,
        device_map="auto",
        compile_model=False,
        use_liger=True,
    )
    model.eval()
    pad_token_id = _ensure_pad_token(tokenizer)
    device = next(model.parameters()).device

    prompts = REASONING_PROMPTS[:num_traces]
    prompt_texts = [_build_prompt(prompt) for prompt in prompts]
    sequences: list[list[int]] = []
    prompt_lengths: list[int] = []
    completions: list[str] = []

    if trace_source == "curated":
        for prompt_text, completion in zip(prompt_texts, CURATED_REASONING_COMPLETIONS[:num_traces]):
            prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
            completion_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]
            sequences.append([*prompt_ids, *completion_ids])
            prompt_lengths.append(len(prompt_ids))
            completions.append(completion)
    elif trace_source == "model":
        chat_prompt_texts = [_chat_prompt(tokenizer, prompt) for prompt in prompts]
        for prompt_text in chat_prompt_texts:
            inputs = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
            inputs = {key: value.to(device) for key, value in inputs.items()}
            prompt_length = int(inputs["input_ids"].shape[1])
            generation_kwargs: dict[str, Any] = {
                "max_new_tokens": max_new_tokens,
                "pad_token_id": pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
                "do_sample": temperature > 0,
            }
            if temperature > 0:
                generation_kwargs["temperature"] = temperature
                generation_kwargs["top_p"] = 0.95
            with torch.no_grad():
                output_ids = model.generate(**inputs, **generation_kwargs)
            sequence = output_ids[0].detach().cpu().tolist()
            sequences.append(sequence)
            prompt_lengths.append(prompt_length)
            completions.append(
                tokenizer.decode(sequence[prompt_length:], skip_special_tokens=True, clean_up_tokenization_spaces=False)
            )
    else:
        raise ValueError(f"Unsupported trace_source: {trace_source}")

    sequence_ids, attention_mask, action_mask = _pad_sequences(
        sequences=sequences,
        prompt_lengths=prompt_lengths,
        pad_token_id=pad_token_id,
        device=device,
    )
    sink_token_ids = collect_ad_cispo_sink_token_ids(tokenizer) if apply_sink_guard else frozenset()
    masks = build_saliency_masks(
        sequence_ids=sequence_ids,
        attention_mask=attention_mask,
        action_mask=action_mask,
        sink_token_ids=sink_token_ids,
    )
    token_batch = SequenceTokenBatch(sequence_ids=sequence_ids, attention_mask=attention_mask)

    with torch.no_grad():
        _, attention_raw = extract_future_attention_indegree_saliency(
            model=model,
            token_batch=token_batch,
            masks=masks,
            top_layers=top_layers,
            block_size=attention_block_size,
            return_logits=False,
        )

    advantages = torch.ones(sequence_ids.shape[0], dtype=torch.float32, device=device)
    _, causal_raw = extract_causal_tangent_saliency(
        model=model,
        token_batch=token_batch,
        masks=masks,
        advantages=advantages,
        top_layers=top_layers,
        return_logits=False,
    )

    threshold_config = ADCispoThresholdConfig(clip_high=1.2, min_multiplier=0.25, max_multiplier=1.5)
    _, attention_thresholds, _ = compute_token_clip_thresholds(
        raw_saliency=RawTokenSaliency(values=attention_raw.values),
        action_mask=masks.adaptive_clip_mask,
        threshold_config=threshold_config,
    )
    _, causal_thresholds, _ = compute_token_clip_thresholds(
        raw_saliency=RawTokenSaliency(values=causal_raw.values),
        action_mask=masks.adaptive_clip_mask,
        threshold_config=threshold_config,
    )

    token_rows = _token_rows(tokenizer, sequence_ids, action_mask)
    attention_scores = _scores_for_tokens(attention_raw.values[:, 1:], action_mask)
    causal_scores = _scores_for_tokens(causal_raw.values[:, 1:], action_mask)
    attention_multipliers = _scores_for_tokens(attention_thresholds.multipliers, action_mask)
    causal_multipliers = _scores_for_tokens(causal_thresholds.multipliers, action_mask)

    summary = {
        "model_name": model_name,
        "seed": seed,
        "trace_source": trace_source,
        "num_traces": len(prompts),
        "max_new_tokens": max_new_tokens,
        "top_layers": top_layers,
        "attention_block_size": attention_block_size,
        "sink_guard_applied": apply_sink_guard,
        "advantages": "unit positive advantage for every generated trace",
        "traces": [
            {
                "trace_index": idx,
                "prompt": prompt,
                "completion": completion,
                "num_saliency_tokens": len(tokens),
                "attention_top_tokens": attention_top,
                "causal_tangent_top_tokens": causal_top,
            }
            for idx, (prompt, completion, tokens, attention_top, causal_top) in enumerate(
                zip(
                    prompts,
                    completions,
                    token_rows,
                    _top_tokens(token_rows, attention_scores),
                    _top_tokens(token_rows, causal_scores),
                )
            )
        ],
    }
    data = {
        **summary,
        "token_rows": token_rows,
        "future_attention_in_degree": {
            "scores": attention_scores,
            "multipliers": attention_multipliers,
        },
        "causal_tangent": {
            "scores": causal_scores,
            "multipliers": causal_multipliers,
        },
    }

    attention_html = _method_heatmap_html(
        title="Future-Attention In-Degree Saliency Heatmaps",
        method_name="future_attention_in_degree",
        prompts=prompts,
        completions=completions,
        token_rows=token_rows,
        score_rows=attention_scores,
        top_tokens=_top_tokens(token_rows, attention_scores),
        color=(220, 113, 32),
    )
    causal_html = _method_heatmap_html(
        title="Causal-Tangent Saliency Heatmaps",
        method_name="causal_tangent",
        prompts=prompts,
        completions=completions,
        token_rows=token_rows,
        score_rows=causal_scores,
        top_tokens=_top_tokens(token_rows, causal_scores),
        color=(17, 128, 122),
    )
    head_to_head_html = _head_to_head_html(
        prompts=prompts,
        completions=completions,
        token_rows=token_rows,
        attention_scores=attention_scores,
        causal_scores=causal_scores,
    )

    remote_dir = Path(ARTIFACTS_DIR) / "saliency_reasoning_traces"
    remote_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "summary.json": json.dumps(summary, indent=2),
        "saliency_data.json": json.dumps(data, indent=2),
        "attention_heatmap.html": attention_html,
        "causal_tangent_heatmap.html": causal_html,
        "head_to_head_heatmap.html": head_to_head_html,
    }
    for filename, content in files.items():
        (remote_dir / filename).write_text(content)
    artifacts_volume.commit()

    return {
        "remote_dir": str(remote_dir),
        "files": files,
        "summary": summary,
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate reasoning traces and compare AD-CISPO saliency heatmaps on Modal.")
    parser.add_argument("--gpu", default=DEFAULT_GPU, help="Modal GPU request, e.g. H100 or A100-80GB.")
    parser.add_argument("--cloud", default=None, help="Optional Modal cloud selector.")
    parser.add_argument("--region", default=None, help="Optional Modal region selector.")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_SECONDS, help="Function timeout in seconds.")
    parser.add_argument("--secret", action="append", default=[], help="Modal Secret name to inject. Repeatable.")
    parser.add_argument("--dotenv", default=".env", help="Local dotenv path to send as a Modal Secret.")
    parser.add_argument("--no-dotenv", action="store_true", help="Do not read a local dotenv file.")
    parser.add_argument("--no-local-env-secret", action="store_true", help="Do not forward local auth env vars.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model used to generate traces and compute saliency.")
    parser.add_argument(
        "--trace-source",
        choices=("curated", "model"),
        default="curated",
        help="Use curated reasoning traces by default, or ask the model to generate traces first.",
    )
    parser.add_argument("--num-traces", type=int, default=6, choices=range(1, len(REASONING_PROMPTS) + 1))
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--top-layers", type=int, default=4)
    parser.add_argument("--attention-block-size", type=int, default=128)
    parser.add_argument(
        "--apply-sink-guard",
        action="store_true",
        help="Remove configured separator/special tokens from the saliency target mask before scoring.",
    )
    parser.add_argument("--local-output-dir", default=DEFAULT_OUTPUT_DIR)
    return parser


@app.local_entrypoint()
def main(*arglist: str) -> None:
    args = _build_arg_parser().parse_args(list(arglist))
    dotenv_path = None if args.no_dotenv else args.dotenv
    function = run_saliency_comparison.with_options(
        gpu=args.gpu,
        cloud=args.cloud,
        region=args.region,
        timeout=args.timeout,
        secrets=_local_secrets(
            dotenv_path=dotenv_path,
            include_env_secret=not args.no_local_env_secret,
            secret_names=args.secret,
        ),
    )
    result = function.remote(
        model_name=args.model,
        trace_source=args.trace_source,
        num_traces=args.num_traces,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        seed=args.seed,
        top_layers=args.top_layers,
        attention_block_size=args.attention_block_size,
        apply_sink_guard=args.apply_sink_guard,
    )

    output_dir = Path(args.local_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, content in result["files"].items():
        (output_dir / filename).write_text(content)
    print(f"Wrote saliency comparison artifacts to {output_dir.resolve()}")
    print(f"Modal volume artifacts: {result['remote_dir']}")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    _build_arg_parser().print_help()
