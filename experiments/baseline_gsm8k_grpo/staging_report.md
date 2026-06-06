# Baseline GSM8K GRPO Staging Report

Generated: 2026-06-02T21:31:03Z

## Access Checks

- W&B access works through the project `uv` environment.
- W&B user: `rundong-liu-eric` / Rundong Liu.
- W&B project access: `autocurriculum/curious`.
- Modal profile: `rundong-liu-eric`.
- Modal workspace: `rundong-liu-eric`, environment `main`.
- Modal CLI used for launch: `/Users/ericliu/.local/bin/modal`, version `1.4.3`.
- Modal H100 access is confirmed by W&B run metadata and live `nvidia-smi` output: `NVIDIA H100 80GB HBM3`.

## Launches

- `v3`: reached a Modal H100 container and initialized W&B, but produced no metrics. The local launcher had been interrupted and no Modal app/container was listed by the time it was inspected.
- `v4`: launched with `--background`, created app `ap-WSQbKvxABlz3oX78u7VfVz`, printed function call `fc-01KT51ZN66S7YFZ46CHQYNGB5M`, then stopped before any W&B run appeared. The background path is not reliable for this launcher.
- `v5`: attached run, app `ap-ecf0fEY1S30jfdG0WaNyGF`, function call `fc-01KT5210W4MVNQZ921SCJQSEAG`, W&B run `r4dw7413`. It reached epoch 0 but stalled before completing the first update. Evidence: no metrics after about 14 minutes, H100 memory resident at about 24 GB, GPU utilization 0%, and logs showed Torch Dynamo/Inductor compile warnings in the `sequences_log_probs` path.
- `v6`: attached run after removing `torch.compile` wrappers from the log-prob/advantage helpers. App `ap-hoZIYGe2xQpNTWwwe0L6ET`, function call `fc-01KT530MNE5K1YQM6HGWBVP9EN`, W&B run `npw100co`.

## Current v6 Status

- State: running.
- W&B URL: https://wandb.ai/autocurriculum/curious/runs/npw100co
- Modal URL: https://modal.com/apps/rundong-liu-eric/main/ap-hoZIYGe2xQpNTWwwe0L6ET
- Latest observed batch: `num_batches_visited = 55` as of the final W&B snapshot after stopping the Modal app.
- Latest metrics:
  - `train/mean_batch_returns = 0.078125`
  - `train/mean_batch_solved_rate = 0.484375`
  - `train/mean_batch_outcome_returns = 0.078125`
  - `train/mean_num_words_in_completions = 230`
  - `train/mean_action_entropy = 0.49609375`
  - `train/loss = 0.00040435791015625`
  - `train/actor_loss = 0.00040435791015625`
  - `train/mean_kl = 0`
  - `train/grad_norm = 2.03125`
  - `train/lr = 5e-6`
- Modal memory at rollout logs remained stable around 4.5 GB allocated and 7.0 GB reserved through batch 55, with max reserved 26.8 GB.
- Modal app `ap-hoZIYGe2xQpNTWwwe0L6ET` was stopped manually after collecting the staged baseline snapshot; no Modal containers are currently active.

## Notes

- The Homebrew `wandb` CLI is broken locally because it imports an older W&B package against NumPy 2.x. The project `uv` environment works and is what the monitor uses.
- `scripts/monitor_modal_wandb.py` can poll W&B and Modal, but its default run name is still `v2`; override `RUN_NAME=baseline-gsm8k-grpo-qwen3-0p6b-sglang-fa3-h100-seed42-v6`.
- The full `PolicyGradientTrainer` path does not currently checkpoint or evaluate, despite launch flags setting checkpoint/eval intervals.
- The observed throughput is roughly three completed batches per five minutes in the latest sample, so a full one-epoch GSM8K run may exceed the 24h Modal timeout unless throughput improves.
