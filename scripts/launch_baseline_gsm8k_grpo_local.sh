#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export RUN_NAME="${RUN_NAME:-baseline-gsm8k-grpo-qwen3-0p6b-sglang-fa3-h100-seed42-v3}"
export WANDB_GROUP="${WANDB_GROUP:-baseline-grpo-gsm8k}"
export OBJECTIVE_NAME="${OBJECTIVE_NAME:-GRPO}"
export USE_CISPO_LOSS="${USE_CISPO_LOSS:-0}"

exec "${SCRIPT_DIR}/launch_baseline_gsm8k_training_local.sh" "$@"
