#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export RUN_NAME="${RUN_NAME:-cispo-baseline-gsm8k-grpo-qwen3-0p6b-sglang-fa3-h100-seed42-v1}"
export WANDB_GROUP="${WANDB_GROUP:-cispo-baseline-grpo-gsm8k}"

exec "${SCRIPT_DIR}/launch_baseline_gsm8k_grpo_modal.sh" "$@"
