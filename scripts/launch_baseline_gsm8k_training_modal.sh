#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

RUN_NAME="${RUN_NAME:-baseline-gsm8k-grpo-qwen3-1p7b-sglang-fa3-h100-seed42-v1}"
WANDB_ENTITY="${WANDB_ENTITY:-autocurriculum}"
WANDB_PROJECT="${WANDB_PROJECT:-curious}"
WANDB_GROUP="${WANDB_GROUP:-baseline-grpo-gsm8k}"
OBJECTIVE_NAME="${OBJECTIVE_NAME:-GRPO}"
USE_CISPO_LOSS="${USE_CISPO_LOSS:-0}"
USE_AD_CISPO="${USE_AD_CISPO:-0}"
if [[ "${USE_AD_CISPO}" == "1" ]]; then
  USE_CISPO_LOSS=1
fi
MODAL_GPU="${MODAL_GPU:-H100}"
MODAL_TIMEOUT="${MODAL_TIMEOUT:-86400}"
BACKGROUND="${BACKGROUND:-0}"
MODAL_SECRET="${MODAL_SECRET:-}"
DRY_RUN="${DRY_RUN:-0}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-1.7B}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-64}"
GROUP_SIZE="${GROUP_SIZE:-8}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1536}"
MAX_TRAIN_BATCHES="${MAX_TRAIN_BATCHES:-130}"
MINI_BATCH_SIZE="${MINI_BATCH_SIZE:-32}"
BACKWARD_MICRO_BATCH_SIZE="${BACKWARD_MICRO_BATCH_SIZE:-8}"
LOGITS_MINIBATCH_SIZE="${LOGITS_MINIBATCH_SIZE:-16}"
COMPILE_TRAIN_MODEL="${COMPILE_TRAIN_MODEL:-0}"
EPOCHS_PER_STEP="${EPOCHS_PER_STEP:-1}"
TRAIN_ENTROPY_LOG_INTERVAL="${TRAIN_ENTROPY_LOG_INTERVAL:-10}"
EVAL_INTERVAL="${EVAL_INTERVAL:-10}"
TRAIN_TEXT_LOG_INTERVAL="${TRAIN_TEXT_LOG_INTERVAL:-10}"
EVAL_TEXT_LOG_INTERVAL="${EVAL_TEXT_LOG_INTERVAL:-10}"
COMPLETION_LOG_SAMPLE_SIZE="${COMPLETION_LOG_SAMPLE_SIZE:-8}"
KL_WEIGHT="${KL_WEIGHT:-0.001}"
if [[ "${USE_CISPO_LOSS}" == "1" ]]; then
  KL_WEIGHT=0
fi
REF_MODEL_UPDATE_FREQ="${REF_MODEL_UPDATE_FREQ:-0}"
SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.12}"
SGLANG_ATTENTION_BACKEND="${SGLANG_ATTENTION_BACKEND:-fa3}"
SGLANG_DTYPE="${SGLANG_DTYPE:-bfloat16}"
SGLANG_REQUEST_BATCH_SIZE="${SGLANG_REQUEST_BATCH_SIZE:-8}"
SGLANG_PORT="${SGLANG_PORT:-30000}"
SGLANG_WEIGHT_SYNC_DIR="${SGLANG_WEIGHT_SYNC_DIR:-/tmp/curious-sglang-weight-sync}"
SGLANG_WEIGHT_SYNC_INTERVAL="${SGLANG_WEIGHT_SYNC_INTERVAL:-1}"
AD_CISPO_SALIENCY_METHOD="${AD_CISPO_SALIENCY_METHOD:-future_attention_in_degree}"
AD_CISPO_TOP_LAYERS="${AD_CISPO_TOP_LAYERS:-4}"
AD_CISPO_MIN_MULTIPLIER="${AD_CISPO_MIN_MULTIPLIER:-0.0}"
AD_CISPO_MAX_MULTIPLIER="${AD_CISPO_MAX_MULTIPLIER:-}"
AD_CISPO_EPS="${AD_CISPO_EPS:-1e-8}"
AD_CISPO_ATTENTION_BLOCK_SIZE="${AD_CISPO_ATTENTION_BLOCK_SIZE:-256}"

while (($#)); do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --background)
      BACKGROUND=1
      shift
      ;;
    --foreground|--no-background)
      BACKGROUND=0
      shift
      ;;
    *)
      echo "Unknown launch option: $1" >&2
      echo "Supported options: --dry-run, --background, --foreground, --no-background" >&2
      exit 2
      ;;
  esac
done

ROLLOUT_BATCH_SIZE=$((TRAIN_BATCH_SIZE * GROUP_SIZE))
if (( MINI_BATCH_SIZE % GROUP_SIZE != 0 )); then
  echo "MINI_BATCH_SIZE must be divisible by GROUP_SIZE: ${MINI_BATCH_SIZE} % ${GROUP_SIZE} != 0" >&2
  exit 2
fi
if (( ROLLOUT_BATCH_SIZE % MINI_BATCH_SIZE != 0 )); then
  echo "TRAIN_BATCH_SIZE * GROUP_SIZE must be divisible by MINI_BATCH_SIZE: ${ROLLOUT_BATCH_SIZE} % ${MINI_BATCH_SIZE} != 0" >&2
  exit 2
fi
if [[ "${COMPILE_TRAIN_MODEL}" != "0" && "${COMPILE_TRAIN_MODEL}" != "1" ]]; then
  echo "COMPILE_TRAIN_MODEL must be 0 or 1; got ${COMPILE_TRAIN_MODEL}" >&2
  exit 2
fi

if [[ "${DRY_RUN}" != "1" && -z "${WANDB_API_KEY:-}" && -z "${MODAL_SECRET}" && ! -f ".env" ]]; then
  echo "WANDB_API_KEY is not set, MODAL_SECRET is empty, and .env was not found." >&2
  echo "Export WANDB_API_KEY, create .env, or set MODAL_SECRET to a Modal Secret name before launching." >&2
  exit 2
fi

modal_args=(--gpu "${MODAL_GPU}" --timeout "${MODAL_TIMEOUT}")
if [[ "${BACKGROUND}" == "1" ]]; then
  modal_args+=(--background)
fi
if [[ -n "${MODAL_SECRET}" ]]; then
  modal_args+=(--secret "${MODAL_SECRET}")
fi

echo "Preparing Modal baseline ${OBJECTIVE_NAME} launch:"
echo "  run: ${RUN_NAME}"
echo "  W&B: ${WANDB_ENTITY}/${WANDB_PROJECT}"
echo "  gpu: ${MODAL_GPU}"
echo "  background: ${BACKGROUND}"
echo "  dry_run: ${DRY_RUN}"
echo "  model: ${MODEL_NAME}"
echo "  generation_backend: sglang"
echo "  sglang_attention_backend: ${SGLANG_ATTENTION_BACKEND}"
echo "  sglang_dtype: ${SGLANG_DTYPE}"
echo "  sglang_mem_fraction_static: ${SGLANG_MEM_FRACTION_STATIC}"
echo "  sglang_request_batch_size: ${SGLANG_REQUEST_BATCH_SIZE}"
echo "  sglang_weight_sync_interval: ${SGLANG_WEIGHT_SYNC_INTERVAL}"
echo "  train_batch_size: ${TRAIN_BATCH_SIZE}"
echo "  eval_batch_size: ${EVAL_BATCH_SIZE}"
echo "  group_size: ${GROUP_SIZE}"
echo "  rollout_batch_size: ${ROLLOUT_BATCH_SIZE}"
echo "  max_train_batches: ${MAX_TRAIN_BATCHES}"
echo "  mini_batch_size: ${MINI_BATCH_SIZE}"
echo "  backward_micro_batch_size: ${BACKWARD_MICRO_BATCH_SIZE}"
echo "  logits_minibatch_size: ${LOGITS_MINIBATCH_SIZE}"
echo "  compile_train_model: ${COMPILE_TRAIN_MODEL}"
echo "  epochs_per_step: ${EPOCHS_PER_STEP}"
echo "  train_entropy_log_interval: ${TRAIN_ENTROPY_LOG_INTERVAL}"
echo "  eval_interval: ${EVAL_INTERVAL}"
echo "  train_text_log_interval: ${TRAIN_TEXT_LOG_INTERVAL}"
echo "  eval_text_log_interval: ${EVAL_TEXT_LOG_INTERVAL}"
echo "  completion_log_sample_size: ${COMPLETION_LOG_SAMPLE_SIZE}"
echo "  kl_weight: ${KL_WEIGHT}"
echo "  ref_model_update_freq: ${REF_MODEL_UPDATE_FREQ} (frozen reference)"
echo "  use_cispo_loss: ${USE_CISPO_LOSS}"
echo "  use_ad_cispo: ${USE_AD_CISPO}"
if [[ "${USE_AD_CISPO}" == "1" ]]; then
  echo "  ad_cispo_saliency_method: ${AD_CISPO_SALIENCY_METHOD}"
  echo "  ad_cispo_top_layers: ${AD_CISPO_TOP_LAYERS}"
  echo "  ad_cispo_min_multiplier: ${AD_CISPO_MIN_MULTIPLIER}"
  echo "  ad_cispo_max_multiplier: ${AD_CISPO_MAX_MULTIPLIER:-None}"
  echo "  ad_cispo_eps: ${AD_CISPO_EPS}"
  echo "  ad_cispo_attention_block_size: ${AD_CISPO_ATTENTION_BLOCK_SIZE}"
fi

command=(scripts/modal_train.sh "${modal_args[@]}" -- \
  --wandb-config.entity "${WANDB_ENTITY}" \
  --wandb-config.project "${WANDB_PROJECT}" \
  --wandb-config.group "${WANDB_GROUP}" \
  --wandb-config.name "${RUN_NAME}" \
  --base-config.model-name "${MODEL_NAME}" \
  --base-config.device-index 0 \
  --base-config.dataset-name "openai/gsm8k" \
  --base-config.train-batch-size "${TRAIN_BATCH_SIZE}" \
  --base-config.eval-batch-size "${EVAL_BATCH_SIZE}" \
  --base-config.num-epochs 1 \
  --base-config.max-train-batches "${MAX_TRAIN_BATCHES}" \
  --base-config.num-workers 16 \
  --base-config.seed 42 \
  --base-config.checkpoint-interval 50 \
  --base-config.eval-interval "${EVAL_INTERVAL}" \
  --base-config.train-text-log-interval "${TRAIN_TEXT_LOG_INTERVAL}" \
  --base-config.train-entropy-log-interval "${TRAIN_ENTROPY_LOG_INTERVAL}" \
  --base-config.eval-text-log-interval "${EVAL_TEXT_LOG_INTERVAL}" \
  --base-config.completion-log-sample-size "${COMPLETION_LOG_SAMPLE_SIZE}" \
  --sampling-config.model-prompt-length 1024 \
  --sampling-config.max-new-tokens "${MAX_NEW_TOKENS}" \
  --sampling-config.temperature 0.7 \
  --sampling-config.top-p 0.9 \
  --sampling-config.top-k 50 \
  --sampling-config.do-sample \
  --sampling-config.use-cache \
  --sampling-config.repetition-penalty 1.0 \
  --sampling-config.system-prompt "qwen_system_prompt" \
  --sampling-config.generation-backend sglang \
  --sampling-config.sglang-attention-backend "${SGLANG_ATTENTION_BACKEND}" \
  --sampling-config.sglang-dtype "${SGLANG_DTYPE}" \
  --sampling-config.sglang-mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC}" \
  --sampling-config.sglang-request-batch-size "${SGLANG_REQUEST_BATCH_SIZE}" \
  --sampling-config.sglang-port "${SGLANG_PORT}" \
  --sampling-config.sglang-weight-sync-dir "${SGLANG_WEIGHT_SYNC_DIR}" \
  --sampling-config.sglang-weight-sync-interval "${SGLANG_WEIGHT_SYNC_INTERVAL}" \
  --reward-config.no-use-format-reward \
  --reward-config.no-use-overlong-penalty \
  --rl-config.group-size "${GROUP_SIZE}" \
  --rl-config.lr 3e-6 \
  --rl-config.weight-decay 0.01 \
  --rl-config.kl-weight "${KL_WEIGHT}" \
  --rl-config.ref-model-update-freq "${REF_MODEL_UPDATE_FREQ}" \
  --rl-config.clip-eps 0.2 \
  --rl-config.no-use-clip-high \
  --rl-config.no-use-fixed-response-length \
  --rl-config.use-surrogate-loss \
  --rl-config.mini-batch-size "${MINI_BATCH_SIZE}" \
  --rl-config.backward-micro-batch-size "${BACKWARD_MICRO_BATCH_SIZE}" \
  --rl-config.epochs-per-step "${EPOCHS_PER_STEP}" \
  --rl-config.max-grad-norm 0.5 \
  --rl-config.normalize-centered-returns \
  --rl-config.no-use-rloo-scalar \
  --rl-config.logits-minibatch-size "${LOGITS_MINIBATCH_SIZE}")

if [[ "${COMPILE_TRAIN_MODEL}" == "1" ]]; then
  command+=(--base-config.compile-train-model)
else
  command+=(--base-config.no-compile-train-model)
fi

if [[ "${USE_CISPO_LOSS}" == "1" ]]; then
  command+=(--rl-config.use-cispo-loss)
  command+=(--rl-config.use-token-level-loss)
else
  command+=(--rl-config.no-use-token-level-loss)
fi

if [[ "${USE_AD_CISPO}" == "1" ]]; then
  command+=(--rl-config.use-ad-cispo)
  command+=(--rl-config.ad-cispo-saliency-method "${AD_CISPO_SALIENCY_METHOD}")
  command+=(--rl-config.ad-cispo-top-layers "${AD_CISPO_TOP_LAYERS}")
  command+=(--rl-config.ad-cispo-min-multiplier "${AD_CISPO_MIN_MULTIPLIER}")
  if [[ -n "${AD_CISPO_MAX_MULTIPLIER}" ]]; then
    command+=(--rl-config.ad-cispo-max-multiplier "${AD_CISPO_MAX_MULTIPLIER}")
  fi
  command+=(--rl-config.ad-cispo-eps "${AD_CISPO_EPS}")
  command+=(--rl-config.ad-cispo-attention-block-size "${AD_CISPO_ATTENTION_BLOCK_SIZE}")
else
  command+=(--rl-config.no-use-ad-cispo)
fi

if [[ "${DRY_RUN}" == "1" ]]; then
  printf 'Command:'
  printf ' %q' "${command[@]}"
  printf '\n'
  exit 0
fi

exec "${command[@]}"
