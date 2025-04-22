from typing import Dict, Any, TypedDict, Optional
import os

from curious.data.gsm8k import GSM8KDataset
from curious.utils.utils import load_model_tokenizer
from curious.config import TrainingConfig
from curious.policy_gradient.loss import AdaptiveKLController, ConstantKLController
from curious.policy_gradient.loss import ActorLoss
from curious.reward import GSM8KRewardModel
from curious.evaluate import EvaluationConfig, FixedSamplingConfig
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from transformers import PreTrainedModel, PreTrainedTokenizer, GenerationConfig

class TrainingSetup(TypedDict):
    """
    A typed dictionary for storing the training setup.
    """
    run_name: str
    device: torch.device
    cpu_device: torch.device

    train_log_dir: str
    eval_log_dir: str

    target_policy: PreTrainedModel
    tokenizer: PreTrainedTokenizer
    pad_token_id: int
    
    dataset: GSM8KDataset
    rollout_data_loader: DataLoader

    optimizer: optim.Optimizer
    lr_scheduler: optim.lr_scheduler.CosineAnnealingLR

    actor_loss: ActorLoss
    reward_model: GSM8KRewardModel
    generation_config: GenerationConfig
    eval_config: EvaluationConfig

    kl_controller: Optional[AdaptiveKLController | ConstantKLController] = None
    reference_model: Optional[PreTrainedModel] = None   

def set_up_training(config:TrainingConfig) -> TrainingSetup: 
    """
    Set up the training.
    """
    # out 
    training_setup:TrainingSetup = {}

    # check that the data mode is train
    assert config.base_config.mode == "train"
    assert config.grpo_config.mini_batch_size % config.grpo_config.group_size == 0
    assert config.base_config.train_batch_size * config.grpo_config.group_size % config.grpo_config.mini_batch_size == 0
    
    # get the run name
    run_name = config.wandb_config.name.replace("-", "_")
    training_setup["run_name"] = run_name

    # create the logging directory for training logs
    train_log_dir = os.path.join(config.base_config.train_log_dir, run_name)
    os.makedirs(train_log_dir, exist_ok=True)
    training_setup["train_log_dir"] = train_log_dir

    # create the logging directory for evaluation logs
    eval_log_dir = os.path.join(config.base_config.eval_log_dir, run_name)
    os.makedirs(eval_log_dir, exist_ok=True)
    training_setup["eval_log_dir"] = eval_log_dir

    # device & seeding
    device = torch.device("cuda", config.base_config.device_index)
    cpu_device = torch.device("cpu")
    training_setup["device"] = device
    training_setup["cpu_device"] = cpu_device

    # target policy
    model, tokenizer = load_model_tokenizer(
        config.base_config.model_name, 
        device_map=device, 
        freeze_model=True,
    )
    tokenizer.padding_side  = 'left'
    pad_token_id = tokenizer.eos_token_id
    
    training_setup["target_policy"] = model
    training_setup["tokenizer"] = tokenizer
    training_setup["pad_token_id"] = pad_token_id

    # dataset 
    dataset = GSM8KDataset(
        tokenizer=tokenizer,
        dataset_name=config.base_config.dataset_name,
        seed=config.base_config.seed,
        mode=config.base_config.mode,
        max_prompt_length=config.sampling_config.model_prompt_length,
    )
    training_setup["dataset"] = dataset
    rollout_data_loader = DataLoader(
        dataset,
        batch_size=config.base_config.train_batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=config.base_config.num_workers,
    )
    training_setup["rollout_data_loader"] = rollout_data_loader

    if config.grpo_config.kl_weight > 0:
        # kl controller
        if config.grpo_config.kl_controller == "adaptive":
            kl_controller = AdaptiveKLController(
                init_kl_coef=config.grpo_config.kl_weight,
                target_kl=config.grpo_config.kl_target,
                horizon=config.grpo_config.kl_horizon_factor * config.grpo_config.mini_batch_size,
            )
        elif config.grpo_config.kl_controller == "constant":
            kl_controller = ConstantKLController(
                init_kl_coef=config.grpo_config.kl_weight,
            )
        else:
            raise ValueError(f"Invalid KL controller: {config.grpo_config.kl_controller}")
        
        # reference model
        reference_model, _ = load_model_tokenizer(
            config.base_config.model_name, 
            device_map=device, 
            freeze_model=True,
        )
        reference_model.eval()
        
        training_setup["kl_controller"] = kl_controller
        training_setup["reference_model"] = reference_model

    ## Objective
    objective = ActorLoss(
        epsilon=config.grpo_config.clip_eps,
        epsilon_high=config.grpo_config.clip_eps_high,
        kl_weight=config.grpo_config.kl_weight,
        use_clip_high=config.grpo_config.use_clip_high,
        use_token_level_loss=config.grpo_config.use_token_level_loss,
        use_fixed_response_length=config.grpo_config.use_fixed_response_length,
        use_surrogate_loss=config.grpo_config.use_surrogate_loss,
    )
    training_setup["objective"] = objective

    ## Reward model
    reward_model = GSM8KRewardModel(
        answer_pattern=config.reward_config.answer_pattern,
        think_pattern=config.reward_config.think_pattern,
        use_format_reward=config.reward_config.use_format_reward,
        use_overlong_penalty=config.reward_config.use_overlong_penalty,
        l_max=config.reward_config.l_max,
        l_cache=config.reward_config.l_cache,
    )
    training_setup["reward_model"] = reward_model
    
    ## Sampling config
    generation_config = GenerationConfig(
        num_return_sequences=config.grpo_config.group_size,
        max_new_tokens=config.sampling_config.max_new_tokens,
        temperature=config.sampling_config.temperature,
        top_p=config.sampling_config.top_p,
        top_k=config.sampling_config.top_k,
        do_sample =config.sampling_config.do_sample,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id= tokenizer.eos_token_id,
        use_cache=config.sampling_config.use_cache,
        repetition_penalty=config.sampling_config.repetition_penalty,
    )
    training_setup["generation_config"] = generation_config

    ## Evaluation config
    eval_config = EvaluationConfig(
        wandb_config=config.wandb_config,
        base_config=config.base_config,
        sampling_config=FixedSamplingConfig(
            max_new_tokens=config.sampling_config.max_new_tokens,
            system_prompt=config.sampling_config.system_prompt,
            model_prompt_length=config.sampling_config.model_prompt_length,
        ),
        reward_config=config.reward_config,
    )
    training_setup["eval_config"] = eval_config
    
    # optimizer 
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=config.grpo_config.lr,
        weight_decay=config.grpo_config.weight_decay,
    )
    training_setup["optimizer"] = optimizer
    
    # lr scheduler
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=len(rollout_data_loader),
        eta_min=config.grpo_config.lr * 1e-03,
    )
    training_setup["lr_scheduler"] = lr_scheduler

    return training_setup