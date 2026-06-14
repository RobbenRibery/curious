import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from curious.utils.utils import (  # noqa: E402
    DEFAULT_LIGER_ATTN_IMPLEMENTATION,
    load_model_tokenizer,
    validate_flash_attention_3_runtime,
)


DTYPES = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
}


def _attn_implementation(model) -> str | None:
    return getattr(model.config, "_attn_implementation", None) or getattr(model.config, "attn_implementation", None)


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test Liger + Transformers FlashAttention-3 on an H100 host.")
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--dtype", choices=DTYPES, default="bf16")
    parser.add_argument("--prompt", default="What is 1 + 1?")
    args = parser.parse_args()

    validate_flash_attention_3_runtime()
    model, tokenizer = load_model_tokenizer(
        model_name_or_path=args.model,
        dtype_=DTYPES[args.dtype],
        device_map="auto",
        compile_model=False,
        use_liger=True,
    )

    attn_implementation = _attn_implementation(model)
    if attn_implementation != DEFAULT_LIGER_ATTN_IMPLEMENTATION:
        raise RuntimeError(
            f"Expected {DEFAULT_LIGER_ATTN_IMPLEMENTATION}, but model config reports {attn_implementation!r}."
        )

    inputs = tokenizer(args.prompt, return_tensors="pt")
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs, use_cache=False)

    if not torch.isfinite(outputs.logits).all():
        raise RuntimeError("Liger FlashAttention-3 smoke forward produced non-finite logits.")

    print(
        "Liger FlashAttention-3 smoke test passed: "
        f"model={args.model} dtype={args.dtype} device={model.device} attention={attn_implementation}"
    )


if __name__ == "__main__":
    main()
