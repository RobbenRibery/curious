
from dataclasses import dataclass
from uuid import uuid4

from curious.reward import THINK_PATTERN, ANSWER_PATTERN

@dataclass
class WandbConfig:
    """
    A dataclass for storing the wandb configuration.
    """

    project: str = "curious"
    """
    The project to use for the wandb.
    """

    name: str = f"curious-{uuid4()}"
    """
    The name to use for the wandb.
    """
    group: str = "grpo-reproduction"
    """
    The group to use for the wandb.
    """


@dataclass
class BaseConfig:
    """
    A dataclass for storing the evaluation configuration.
    """

    # Model and dataset
    model_name: str = "Qwen/Qwen2-0.5B-Instruct"
    """
    The name of the model to use for the evaluation.
    """
    device_index: int = 0
    """
    The device index to use for cuda
    """
    dataset_name: str = "openai/gsm8k"
    """
    The name of the dataset to use for the evaluation.
    """
    batch_size: int = 256
    """
    The batch size to use for the evaluation.
    """
    num_workers: int = 8
    """
    The number of cpu workers to use for the evaluation.
    """
    mode: str = "test"
    """
    The mode to use for the evaluation.
    """
    log_dir: str = "logs"
    """
    The directory to use for the evaluation.
    """
    seed: int = 42
    """
    The seed to use for the evaluation.
    """
    model_max_length_inuse: int = 2048
    """
    The maximum length of the model to use for the evaluation.
    """
    checkpoint_path: str = None
    """
    The path to the checkpoint to use for the evaluation.
    """
    checkpoint_interval: int = 100
    """
    The interval to use for the checkpoint.
    """

@dataclass
class SamplingConfig:
    """
    A dataclass for storing the sampling configuration.
    """

    max_new_tokens: int = 1024
    """
    The maximum number of new tokens to use for the evaluation.
    """
    temperature: float = 0.7
    """
    The temperature to use for the evaluation.
    """
    top_p: float = 0.9
    """
    The top p to use for the evaluation.
    """
    top_k: int = 50
    """
    The top k to use for the evaluation.
    """
    do_sample: bool = True
    """
    Whether to sample from the model.
    """


@dataclass
class RewardConfig:
    """
    A dataclass for storing the reward configuration.
    """

    answer_pattern: str = ANSWER_PATTERN
    """
    The pattern to use for the answer.
    """
    think_pattern: str = THINK_PATTERN
    """
    The pattern to use for the think.
    """
    use_format_reward: bool = False
    """
    Whether to use the format reward.
    """


@dataclass
class GRPOConfig:
    """
    A dataclass for storing the GRPO configuration.
    """

    group_size: int = 12
    """
    The group size to use for the GRPO.
    """
    lr: float = 1e-6
    """
    The learning rate to use for the GRPO.
    """
    kl_weight: float = 0.01
    """
    The KL weight to use for the GRPO.
    """
    clip_eps: float = 0.2
    """
    The clip epsilon to use for the GRPO.
    """
    mini_batch_size: int = 64
    """
    The mini batch size to use for the GRPO.
    """
    epochs_per_step: int = 1
    """
    The number of epochs to use for the GRPO.
    """