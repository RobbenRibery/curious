import torch

from curious.training.training_setup import TrainState
from curious.config import TrainingConfig


class Trainer:
    def __init__(self, config: TrainingConfig):
        self.config = config

    def train(self):
        pass

    def collect_trajectories(self) -> TrainState:
        pass

    def learn_trajectories(self) -> TrainState:
        pass
