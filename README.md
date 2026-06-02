+ ![Curious Icon](icon.svg)
# Curious
A simple and effective framework for training LLMs using RL and rule-based reward models.

Curious is built to help researchers and developers train large language models with reinforcement learning combined with a simple rule-based reward model. Our aim is to keep the framework clear, efficient, and easy to use. Currently, we support GRPO training on the GSM8K dataset. In the future, we plan to add support for Dr.GRPO, DAPO, MATH AIME, and RAG-RL.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Training](#training)
  - [Evaluation](#evaluation)
- [Configuration](#configuration)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Introduction
Curious is a straightforward framework designed to train LLMs using reinforcement learning and a rule-based reward model. It provides a simple yet powerful training loop based on GRPO and uses the GSM8K dataset for evaluation. The focus is on clarity and efficiency, making it easy for anyone to get started and experiment.

## Features
- **Simple RL-based Training:** A clear and easy-to-follow policy gradient training loop.
- **Configurable Settings:** Adjust parameters using Python data classes and command line options.

## Installation
This project uses [uv](https://docs.astral.sh/uv/) to manage dependencies.

### Steps:
1. **Clone the Repository:**
   ```bash
   git clone https://github.com/RobbenRibery/curious.git
   cd curious
   ```
2. **Install Dependencies:**
   ```bash
   uv sync --group dev
   ```

## Usage
Curious provides separate scripts for training and evaluation. Both use command line options (via tyro) for easy configuration.

### Training
```bash
uv run python -m curious.training --help
```

### Modal Training
Install and authenticate the Modal CLI:
```bash
uv tool install modal
uv tool run modal setup
```

Launch training from your local checkout onto Modal:
```bash
scripts/modal_train.sh \
  --wandb-config.project curious \
  --wandb-config.name modal-grpo-smoke \
  --base-config.model-name Qwen/Qwen2.5-0.5B-Instruct \
  --base-config.train-batch-size 8 \
  --rl-config.group-size 16 \
  --rl-config.kl-weight 0
```

The Modal launcher defaults to `gpu=H100` and persists train logs, eval logs, checkpoints, Hugging Face cache, and W&B cache in the `curious-training-artifacts` Modal Volume. The wrapper forwards unknown flags to `curious.training`; use `--` only when you want to separate launcher options from training options:
```bash
scripts/modal_train.sh --gpu A100-80GB --timeout 86400 -- \
  --base-config.checkpoint-interval 20
```

You can call Modal directly as well; Modal's own CLI needs the first `--` before app arguments:
```bash
uv tool run modal run -m curious.modal_train -- --gpu H100 -- \
  --wandb-config.project curious
```

Secrets are not committed. The launcher forwards local `.env`, `WANDB_API_KEY`, `HF_TOKEN`, and `HUGGING_FACE_HUB_TOKEN` when present. You can also inject named Modal Secrets:
```bash
uv tool run modal secret create wandb WANDB_API_KEY=...
scripts/modal_train.sh --secret wandb -- --wandb-config.project curious
```

### Evaluation
```bash
uv run python -m curious.evaluate --help
```

## Configuration
Settings are managed using Python data classes (via tyro). You can adjust them in the configuration files or override them via command line options.

## Roadmap
- **Planned:** Add support for Dr.GRPO and DAPO.
- **Planned:** Add support for MATH AIME.
- **Planned:** Extend the framework to include RAG-RL methods.
- **Planned:** Reimplement using JAX! 

## License
This project is licensed under the [MIT License](LICENSE).
