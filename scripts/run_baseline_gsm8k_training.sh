#!/usr/bin/env bash
set -euo pipefail

if [ -z "${BASH_VERSION:-}" ]; then
  exec /usr/bin/env bash "$0" "$@"
fi

SCRIPT_SOURCE="${BASH_SOURCE[0]:-$0}"
SCRIPT_DIR="$(cd "$(dirname "${SCRIPT_SOURCE}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

if [[ -z "${WANDB_API_KEY:-}" ]]; then
  if [[ -f "${REPO_ROOT}/.env" ]]; then
    export WANDB_API_KEY="$(
      awk -F= '
        function trim(s) { gsub(/^[ \t"]+|[ \t"]+$/, "", s); return s }
        $0 ~ /^[ \t]*(export[ \t]+)?WANDB_API_KEY[ \t]*=/ {
          sub(/^[ \t]*(export[ \t]+)?WANDB_API_KEY[ \t]*=/, "", $0)
          print trim($0)
          exit
        }
      ' "${REPO_ROOT}/.env"
    )"
  fi
fi

if [[ -z "${WANDB_API_KEY:-}" ]]; then
  echo "WANDB_API_KEY not set and not found in ${REPO_ROOT}/.env"
  echo "Export WANDB_API_KEY first, or add WANDB_API_KEY=... to .env."
  exit 2
fi

export WANDB_API_KEY
export TRAINING_WANDB_KEY_EXPORTED_AT="$(date -u +%s)"

exec "${SCRIPT_DIR}/launch_baseline_gsm8k_grpo_modal.sh" "$@"
