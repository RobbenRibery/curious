from dataclasses import dataclass
from typing import Optional
from uuid import uuid4

from curious.reward.rule.gsm8k import * 

@dataclass
class WandbConfig:
    """
    A dataclass for storing the wandb configuration.
    """

    entity: str = "autocurriculum"
    """
    The entity to use for the wandb.
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
    model_name: str = "Qwen/Qwen3-1.7B"
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

    eval_batch_size: int = 64
    """
    The batch size to use for evaluation
    """

    num_epochs:int = 1
    """
    The number of epochs to use for the training.
    """

    max_train_batches: int = 130
    """
    The maximum number of training batches to visit. Set to 0 or lower to disable the cap.
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

    eval_interval: int = 10
    """
    The interval to use for the evaluation.
    """

    train_text_log_interval: int = 10
    """
    The interval to use for the train text log.
    """

    train_entropy_log_interval: int = 1
    """
    The interval to compute and log train action entropy. Set to 0 or lower to disable.
    """

    perf_log_interval: int = 10
    """
    The interval to log phase timing and CUDA memory diagnostics. Set to 0 or lower to disable.
    """

    memory_cleanup_batch_interval: int = 1
    """
    The interval for Python GC and conditional CUDA cleanup at training batch boundaries.
    """

    memory_cleanup_minibatch_empty_cache_interval: int = 0
    """
    The minibatch interval for conditional CUDA cache cleanup. Set to 0 to avoid hot-loop cache flushes.
    """

    memory_cleanup_reserved_ratio_threshold: float = 0.90
    """
    Force CUDA cache cleanup when reserved memory approaches this fraction of device memory.
    """

    memory_cleanup_fragmentation_ratio: float = 1.35
    """
    Run CUDA cache cleanup when reserved memory is this much larger than allocated memory.
    """

    memory_cleanup_force_after_eval: bool = True
    """
    Whether to force Python and CUDA cleanup after evaluation.
    """

    memory_cleanup_force_after_checkpoint: bool = True
    """
    Whether to force Python and CUDA cleanup after checkpoint saves.
    """

    eval_text_log_interval: int = 10
    """
    The interval to use for the eval text log.
    """

    completion_log_sample_size: int = 8
    """
    The maximum number of train/eval sample completions to log at each text log interval.
    """

    return_entropy: bool = False
    """
    Whether to return the entropy of the tokens.
    """

    deepspeed_config: Optional[str] = None
    """
    The path to the deepspeed config file.
    """

    compile_train_model: bool = False
    """
    Whether to wrap the train policy forward pass in torch.compile.
    Keep this disabled by default because compiled model forward can conflict with
    gradient checkpointing and Liger custom autograd on the H100 training path.
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
    
    max_new_tokens: int = 1536
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

    generation_backend: str = "hf"
    """
    The generation backend to use for rollouts. One of: hf, sglang.
    """

    sglang_attention_backend: str = "fa3"
    """
    The attention backend to use for SGLang rollout generation.
    """

    sglang_dtype: str = "bfloat16"
    """
    The floating dtype to use for SGLang rollout generation.
    """

    sglang_mem_fraction_static: float = 0.12
    """
    The static memory fraction reserved by SGLang on the H100.
    """

    sglang_host: str = "127.0.0.1"
    """
    The host for the local SGLang server.
    """

    sglang_port: int = 30000
    """
    The port for the local SGLang server.
    """

    sglang_log_level: str = "warning"
    """
    The SGLang server log level.
    """

    sglang_max_running_requests: Optional[int] = None
    """
    The optional maximum number of running requests for SGLang.
    """

    sglang_chunked_prefill_size: Optional[int] = None
    """
    The optional chunked prefill size for SGLang.
    """

    sglang_request_batch_size: Optional[int] = None
    """
    The optional number of prompt instances to send per SGLang /generate request.
    """

    sglang_startup_timeout: int = 600
    """
    The number of seconds to wait for SGLang startup.
    """

    sglang_weight_sync: str = "disk"
    """
    The SGLang policy weight synchronization method. Currently only disk is supported.
    """

    sglang_weight_sync_dir: str = "sglang_weight_sync"
    """
    The local directory used for disk-based SGLang weight sync.
    """

    sglang_weight_sync_interval: int = 1
    """
    The number of completed training batches between SGLang policy weight syncs.
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
    use_format_reward: bool = False
    """
    Whether to use the format reward.
    """
    use_overlong_penalty: bool = False
    """
    Whether to use the overlong penalty.
    """
    l_max: int = 300
    """
    The maximum length of the completion, beyond which the penalty is applied. (in number of words)
    """
    l_cache: int = 100
    """
    The cache length of the completion, which indicates the peanlizable span of the completion. (in number of words)
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

    use_cispo_loss: bool = False
    """
    Whether to use CISPO's clipped importance-sampling-weight policy gradient loss.
    """

    use_ad_cispo: bool = False
    """
    Whether to use AD-CISPO token-level upper clipping thresholds.
    """

    ad_cispo_saliency_method: str = "future_attention_in_degree"
    """
    The saliency method to use for AD-CISPO. Supported values are future_attention_in_degree, kv_norm, causal_tangent,
    and causal_tangent_smoothed.
    """

    ad_cispo_top_layers: int = 4
    """
    The number of final decoder layers to use for AD-CISPO saliency.
    """

    ad_cispo_min_multiplier: float = 0.0
    """
    The lower bound for token-level AD-CISPO clip multipliers.
    """

    ad_cispo_max_multiplier: Optional[float] = None
    """
    The optional upper bound for token-level AD-CISPO clip multipliers.
    """

    ad_cispo_eps: float = 1e-8
    """
    The numerical epsilon used by AD-CISPO saliency normalization.
    """

    ad_cispo_attention_block_size: int = 256
    """
    The query block size used by exact future-attention in-degree saliency.
    """

    mini_batch_size: int = 16 * 2
    """
    The mini batch size to use for the GRPO.
    """

    backward_micro_batch_size: int = 0
    """
    The microbatch size to use for memory-safe backward accumulation inside each optimizer minibatch.
    Set to 0 or lower to disable backward microbatching.
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
