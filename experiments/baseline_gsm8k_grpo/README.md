# Baseline GSM8K GRPO Modal Run

This is the prepared baseline GRPO run configuration for GSM8K on Modal H100.
It is intentionally not launched by default.

## Launch

```bash
scripts/launch_baseline_gsm8k_grpo_modal.sh
```

The launch script defaults to a background Modal run so the terminal returns
after Modal accepts the call. It requires one of:

- `WANDB_API_KEY` exported locally
- a local `.env` containing `WANDB_API_KEY`
- `MODAL_SECRET=<secret-name>` pointing to a Modal Secret that contains W&B credentials

Useful overrides:

```bash
RUN_NAME=baseline-gsm8k-grpo-qwen3-0p6b-sglang-fa3-h100-seed42-v2 \
WANDB_ENTITY=autocurriculum \
WANDB_PROJECT=curious \
MODAL_GPU=H100 \
BACKGROUND=1 \
scripts/launch_baseline_gsm8k_grpo_modal.sh
```

## Monitor

```bash
scripts/monitor_baseline_gsm8k_grpo.sh
```

The monitor polls:

- W&B run state and summary metrics under `autocurriculum/curious`
- Modal app state for `curious-training`

Useful overrides:

```bash
POLL_SECONDS=30 scripts/monitor_baseline_gsm8k_grpo.sh --show-modal-logs
```

## Baseline Config

- Model: `Qwen/Qwen3-0.6B`
- Dataset: `openai/gsm8k`
- Modal GPU: `H100`
- Generation backend: `SGLang`
- SGLang attention backend: `fa3`
- SGLang static memory fraction: `0.40`
- DeepSpeed: disabled
- Train batch size: `4`
- GRPO group size: `16`
- Minibatch size: `64`
- PPO epochs per rollout: `1`
- Learning rate: `5e-6`
- Weight decay: `0.01`
- Clip epsilon: `0.2`
- KL weight: `0.0`
- AD-CISPO: disabled
- RLOO: disabled
- Max prompt length: `1024`
- Max new tokens: `384`
- Sampling: temperature `0.7`, top-p `0.9`, top-k `50`

This starts from Qwen3 0.6B so the H100 run can use SGLang's
FlashAttention-3 path for fast rollout inference. The small train batch and
shorter decode cap keep memory pressure conservative while KL/AD-CISPO are off.
