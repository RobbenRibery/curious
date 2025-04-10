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
This project uses [poetry](https://python-poetry.org/) to manage dependencies.

### Steps:
1. **Clone the Repository:**
   ```bash
   git clone https://github.com/RobbenRibery/curious.git
   cd curious
   ```
2. **Install Dependencies:**
   ```bash
   pip install -U poetry
   poetry install
   ```

## Usage
Curious provides separate scripts for training and evaluation. Both use command line options (via tyro) for easy configuration.

### Training
```bash
python training.py --help
```

### Evaluation
```bash
python evaluate.py --help
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