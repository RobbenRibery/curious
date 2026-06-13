#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Share the CISPO config; override only AD-CISPO identity plus saliency/multiplier controls.
export RUN_NAME="${RUN_NAME:-gsm8k-ad-cispo-qwen3-1p7b-sglang-fa3-h100-seed42-v1}"
export WANDB_GROUP="${WANDB_GROUP:-ad-cispo-gsm8k}"
export OBJECTIVE_NAME="AD-CISPO"
export USE_CISPO_LOSS="1"
export USE_AD_CISPO="1"
export AD_CISPO_SALIENCY_METHOD="${AD_CISPO_SALIENCY_METHOD:-future_attention_in_degree}"
export AD_CISPO_TOP_LAYERS="${AD_CISPO_TOP_LAYERS:-4}"
export AD_CISPO_MIN_MULTIPLIER="${AD_CISPO_MIN_MULTIPLIER:-0.0}"
export AD_CISPO_MAX_MULTIPLIER="${AD_CISPO_MAX_MULTIPLIER:-}"
export AD_CISPO_EPS="${AD_CISPO_EPS:-1e-8}"
export AD_CISPO_ATTENTION_BLOCK_SIZE="${AD_CISPO_ATTENTION_BLOCK_SIZE:-256}"

exec "${SCRIPT_DIR}/launch_baseline_gsm8k_training_local.sh" "$@"
