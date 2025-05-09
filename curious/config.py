from dataclasses import dataclass
from uuid import uuid4

from curious.reward.rule.gsm8k import * 

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
    
    group: str = "grpo-iterate"
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
    The name of the model to use for the training.
    """
    
    device_index: int = 0
    """
    The device index to use for cuda
    """
    
    dataset_name: str = "openai/gsm8k"
    """
    The name of the dataset to use for the training.
    """
    
    train_batch_size: int = 16
    """
    The batch size to use for training 
    """

    eval_batch_size: int = 256
    """
    The batch size to use for evaluation
    """

    num_epochs:int = 1
    """
    The number of epochs to use for the training.
    """
    
    num_workers: int = 8
    """
    The number of cpu workers to use for the data loader
    """
    
    mode: str = "train"
    """
    The mode to use for the training/testing.
    """
    
    train_log_dir: str = "train_logs"
    """
    The directory to use for the train logs.
    """
    
    eval_log_dir: str = "eval_logs"
    """
    The directory to use for the eval logs.
    """
    
    seed: int = 42
    """
    The seed to use for the evaluation.
    """
    checkpoint_dir: str = "checkpoints/"
    """
    The directory to use for the checkpoint.
    """
    
    checkpoint_interval: int = 100
    """
    The interval to use for the checkpoint and to evaluate the model.
    """

    eval_interval: int = 50
    """
    The interval to use for the evaluation.
    """

    train_text_log_interval: int = 500
    """
    The interval to use for the train text log.
    """

    eval_text_log_interval: int = 500
    """
    The interval to use for the eval text log.
    """

    return_entropy: bool = False
    """
    Whether to return the entropy of the tokens.
    """

    deepspeed_config: str = "deepspeed_config.json"
    """
    The path to the deepspeed config file.
    """


@dataclass
class SamplingConfig:
    """
    A dataclass for storing the sampling configuration.
    """
    
    model_prompt_length: int = 2048
    """
    The maximum length of the model to use for the evaluation.
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
    
    use_cache:bool = True
    """
    Whether to use kv cache during inference
    """
    
    repetition_penalty: float = 1.0
    """
    The repetition penalty to use for the sampling.
    """
    
    system_prompt: str = "deepseek_system_prompt"
    """
    The system prompt to use for the sampling.
    """

@dataclass
class RewardConfig:
    """
    A dataclass for storing the reward configuration.
    """

    answer_pattern: str = QWEN_ANSWER_PATTERN
    """
    The pattern to use for the answer.
    """
    think_pattern: str = THINK_PATTERN
    """
    The pattern to use for the think.
    """
    use_format_reward: bool = True
    """
    Whether to use the format reward.
    """
    use_overlong_penalty: bool = True
    """
    Whether to use the overlong penalty.
    """
    l_max: int = 300
    """
    The maximum length of the completion, beyond which the penalty is applied.
    """
    l_cache: int = 100
    """
    The cache length of the completion, which indicates the peanlizable span of the completion.
    """

@dataclass
class RLConfig:
    """
    A dataclass for storing the RL configuration.
    """

    group_size: int = 16
    """
    The group size to use for the GRPO.
    """
    
    lr: float = 1e-6
    """
    The learning rate to use for the GRPO.
    """

    anneling_lr: bool = False
    """
    Whether to use the anneling learning rate.
    """

    anneling_temperature: bool = False
    """
    Whether to use the anneling temperature.
    """

    weight_decay: float = 0.0
    """
    The weight decay to use for the GRPO.
    """
    
    kl_weight: float = 0.01
    """
    The KL weight to use for the GRPO.
    """

    kl_controller: str = "constant"
    """
    The KL controller to use for the GRPO.
    """

    kl_target: float = 0.01
    """
    The KL target to use for the GRPO.
    """

    kl_horizon_factor: int = 5
    """
    The KL horizon factor to use for the GRPO.
    """
    
    clip_eps: float = 0.2
    """
    The clip epsilon to use for the GRPO.
    """
    
    clip_eps_high: float = 0.28
    """
    The clip epsilon high to use for the GRPO.
    """
    
    use_clip_high: bool = False
    """
    Whether to use the clip epsilon high.
    """
    
    use_token_level_loss: bool = False
    """
    Whether to use the token level loss.
    """
    
    use_fixed_response_length: bool = False
    """
    Whether to use fixed response length to aggregate loss.
    """

    use_surrogate_loss: bool = True
    """
    Whether to use the surrogate loss (ppo) or policy gradient loss (pg).
    """

    mini_batch_size: int = 16 * 2
    """
    The mini batch size to use for the GRPO.
    """
    
    epochs_per_step: int = 1
    """
    The number of epochs to use for the GRPO.
    """
    
    max_grad_norm: float = 1.0
    """
    The maximum gradient norm to use for the GRPO.
    """

    normalize_centered_returns: bool = True
    """
    Whether to normalize the returns.
    """

    use_rloo_scalar: bool = False
    """
    Whether to use the rloo scalar.
    """

    ref_model_update_freq: int = 0
    """
    The interval to update the reference model.
    """

    logits_minibatch_size: int = 64
    """
    The minibatch size to use for the logits.
    """

@dataclass
class SFLConfig:
    """
    A dataclass for storing the SFL configuration.
    """
    
    sfl_enabled: bool = False
    """
    Whether to use Sampling for Learnability (No-regret paper).
    """

    sfl_total_steps: int = 200
    """
    The total number of steps to use for the SFL.
    """

    sfl_sampling_batch_size: int = 256
    """
    The batch size to use for the SFL sampling.
    """
    
    sfl_total_scanning_size: int = 1024
    """
    The total size of the dataset to scan.
    """

    sfl_num_samples_to_collect: int = 256
    """
    The number of samples to collect.
    """


@dataclass
class TrainingConfig:
    """
    A dataclass for storing the training configuration.
    """

    rl_config: RLConfig
    """
    The RL configuration.
    """

    wandb_config: WandbConfig
    """
    The wandb configuration.
    """

    base_config: BaseConfig
    """
    The base configuration.
    """

    sampling_config: SamplingConfig
    """
    The sampling configuration.
    """

    reward_config: RewardConfig
    """
    The reward configuration.
    """

    sfl_config: SFLConfig
    """
    The SFL configuration.
    """
