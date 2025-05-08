from typing import Dict

import torch

from curious.training.training_setup import TrainState, TrainingSetup
from curious.config import TrainingConfig
from curious.sampling.sampling import rollout

class Trainer:
    def __init__(self, training_setup: TrainingSetup):
        self.training_setup = training_setup

        self.tokenizer = training_setup["tokenizer"]
        self.reward_model = training_setup["reward_model"]
        self.generation_config = training_setup["generation_config"]

        self.model = training_setup["target_policy"]
        self.optimizer = training_setup["optimizer"]
        self.scheduler = training_setup["scheduler"]
        self.device = training_setup["device"]

        self.rollout_data_loader = training_setup["rollout_data_loader"]

    def train(self):
        pass

    def collect_trajectories(self, batch_inputs: Dict[str, torch.Tensor]) -> TrainState:

        self.model.eval()
        self.model.gradient_checkpointing_disable()

        rollout_out = rollout(
            model=self.model,
            tokenizer=self.tokenizer,
            batch_inputs=batch_inputs,
            reward_model=self.reward_model, 
            generation_config=self.generation_config,
            group_size=self.args.grpo_config.group_size,
            seed=self.args.base_config.seed,
            normalize_centered_returns=args.grpo_config.normalize_centered_returns,
            use_rloo_scalar=args.grpo_config.use_rloo_scalar,
        )

    def learn_trajectories(self) -> TrainState:
        pass
