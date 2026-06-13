#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

export WANDB_ENTITY="${WANDB_ENTITY:-autocurriculum}"
export WANDB_PROJECT="${WANDB_PROJECT:-curious}"
export RUN_NAME="${RUN_NAME:-baseline-gsm8k-grpo-qwen3-1p7b-sglang-fa3-h100-seed42-v1}"
export MODAL_APP="${MODAL_APP:-curious-training}"
export MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-1.7B}"
export GENERATION_BACKEND="${GENERATION_BACKEND:-sglang}"
export SGLANG_ATTENTION_BACKEND="${SGLANG_ATTENTION_BACKEND:-fa3}"

exec uv run python scripts/monitor_modal_wandb.py "$@"
