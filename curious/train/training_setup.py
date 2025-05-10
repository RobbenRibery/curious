from typing import TypedDict, Optional, Tuple
import os
import json
from curious.data.gsm8k import GSM8KDataset
from curious.utils.utils import load_model_tokenizer
from curious.config import TrainingConfig, BaseConfig, RLConfig, SamplingConfig, RewardConfig
from curious.policy_gradient.loss import AdaptiveKLController, ConstantKLController
from curious.policy_gradient.loss import ActorLoss
from curious.reward import GSM8KRewardModel
from curious.evaluate import EvaluationConfig, FixedSamplingConfig

from transformers import PreTrainedModel, PreTrainedTokenizer, GenerationConfig

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

class TrainState(TypedDict):
    """
    A typed dictionary for storing the training state.
    """
    run_name: str
    device: torch.device
    seed:int
    
    model: PreTrainedModel
    optimizer: optim.Optimizer
    lr_scheduler: optim.lr_scheduler.CosineAnnealingLR

    reference_model: Optional[PreTrainedModel] = None
    kl_controller: Optional[AdaptiveKLController | ConstantKLController] = None
    
class TrainingSetup(TypedDict):
    """
    A typed dictionary for storing the training setup.
    """
    run_name: str
    num_epochs: int
    device: torch.device
    cpu_device: torch.device

    train_log_dir: str
    eval_log_dir: str

    target_policy: PreTrainedModel
    tokenizer: PreTrainedTokenizer
    pad_token_id: int
    
    dataset: GSM8KDataset
    rollout_data_loader: DataLoader

    use_vllm: bool

    optimizer: optim.Optimizer
    lr_scheduler: optim.lr_scheduler.CosineAnnealingLR

    actor_loss: ActorLoss
    reward_model: GSM8KRewardModel
    generation_config: GenerationConfig
    eval_config: EvaluationConfig

    kl_controller: Optional[AdaptiveKLController | ConstantKLController] = None
    reference_model: Optional[PreTrainedModel] = None   

    base_config: BaseConfig
    rl_config: RLConfig
    sampling_config: SamplingConfig
    reward_config: RewardConfig

def set_up_training(config:TrainingConfig) -> Tuple[TrainingSetup, TrainState]: 
    """
    Set up the training.
    """
    # out 
    training_setup:TrainingSetup = {}
    training_setup["base_config"] = config.base_config
    training_setup["rl_config"] = config.rl_config
    training_setup["sampling_config"] = config.sampling_config
    training_setup["reward_config"] = config.reward_config

    # check that the data mode is train
    assert config.base_config.mode == "train"
    assert config.rl_config.mini_batch_size % config.rl_config.group_size == 0
    assert config.base_config.train_batch_size * config.rl_config.group_size % config.rl_config.mini_batch_size == 0
    
    # get the run name
    run_name = config.wandb_config.name.replace("-", "_")
    training_setup["run_name"] = run_name
    training_setup["num_epochs"] = config.base_config.num_epochs

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
    print(f"#### Loading target policy {config.base_config.model_name} ####")
    model, tokenizer = load_model_tokenizer(
        model_name_or_path=config.base_config.model_name, 
        trust_remote_code=True,
        dtype_=torch.bfloat16,
        device_map=device, 
        freeze_model=False,
        checkpoint_path=None,
    )
    assert model.generation_config.pad_token_id == tokenizer.pad_token_id, \
    print(model.generation_config.pad_token_id, tokenizer.pad_token_id, tokenizer.pad_token)
   
    tokenizer.padding_side  = 'left'
    pad_token_id = tokenizer.pad_token_id
    
    training_setup["target_policy"] = model
    training_setup["tokenizer"] = tokenizer
    training_setup["pad_token_id"] = pad_token_id

    # dataset 
    print(f"#### Loading dataset {config.base_config.dataset_name} ####")
    dataset = GSM8KDataset(
        tokenizer=tokenizer,
        dataset_name=config.base_config.dataset_name,
        seed=config.base_config.seed,
        mode=config.base_config.mode,
        max_prompt_length=config.sampling_config.model_prompt_length,
    )
    training_setup["dataset"] = dataset

    # rollout data loader
    print(f"#### Loading rollout data loader ####")
    rollout_data_loader = DataLoader(
        dataset,
        batch_size=config.base_config.train_batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=config.base_config.num_workers,
    )
    training_setup["rollout_data_loader"] = rollout_data_loader

    if config.rl_config.kl_weight > 0:
        # kl controller
        print(f"#### Loading KL controller ####")
        if config.rl_config.kl_controller == "adaptive":
            kl_controller = AdaptiveKLController(
                init_kl_coef=config.rl_config.kl_weight,
                target_kl=config.rl_config.kl_target,
                horizon=config.rl_config.kl_horizon_factor * config.rl_config.mini_batch_size,
            )
        elif config.rl_config.kl_controller == "constant":
            kl_controller = ConstantKLController(
                init_kl_coef=config.rl_config.kl_weight,
            )
        else:
            raise ValueError(f"Invalid KL controller: {config.rl_config.kl_controller}")
        
        # reference model
        print(f"#### Loading reference model ####")
        reference_model, _ = load_model_tokenizer(
            model_name_or_path=config.base_config.model_name, 
            trust_remote_code=True,
            dtype_=torch.bfloat16,
            device_map=device, 
            freeze_model=True,
            checkpoint_path=None,
        )
        reference_model.eval()
        training_setup["kl_controller"] = kl_controller
        training_setup["reference_model"] = reference_model
    else:
        training_setup["kl_controller"] = None
        training_setup["reference_model"] = None

    ## Objective
    print(f"#### Defining actor loss ####")
    objective = ActorLoss(
        epsilon=config.rl_config.clip_eps,
        epsilon_high=config.rl_config.clip_eps_high,
        kl_weight=config.rl_config.kl_weight,
        use_clip_high=config.rl_config.use_clip_high,
        use_token_level_loss=config.rl_config.use_token_level_loss,
        use_fixed_response_length=config.rl_config.use_fixed_response_length,
        use_surrogate_loss=config.rl_config.use_surrogate_loss,
    )
    training_setup["actor_loss"] = objective

    ## Reward model
    print(f"#### Defining reward model ####")
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
    print(f"#### Defining generation config ####")
    generation_config = GenerationConfig(
        num_return_sequences=config.rl_config.group_size,
        max_new_tokens=config.sampling_config.max_new_tokens,
        temperature=config.sampling_config.temperature,
        top_p=config.sampling_config.top_p,
        top_k=config.sampling_config.top_k,
        do_sample =config.sampling_config.do_sample,
        use_cache=config.sampling_config.use_cache,
        repetition_penalty=config.sampling_config.repetition_penalty,
    )
    training_setup["generation_config"] = generation_config

    ## Evaluation config
    print(f"#### Defining evaluation config ####")
    eval_config = EvaluationConfig(
        wandb_config=config.wandb_config,
        base_config=config.base_config,
        sampling_config=FixedSamplingConfig(
            system_prompt=config.sampling_config.system_prompt,
        ),
        reward_config=config.reward_config,
    )
    training_setup["eval_config"] = eval_config
    
    # optimizer 
    print(f"#### Defining optimizer ####")
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=config.rl_config.lr,
        weight_decay=config.rl_config.weight_decay,
    )
    
    # lr scheduler
    print(f"#### Defining lr scheduler ####")
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=len(rollout_data_loader),
        eta_min=config.rl_config.lr * 1e-03,
    )

    # deepspeed
    if config.base_config.deepspeed_config is not None:
        print(f"#### Setting up deepspeed from {config.base_config.deepspeed_config} ####")
        with open(config.base_config.deepspeed_config, "r") as f:
            deepspeed_config = json.load(f)
        f.close()

        deepspeed_config["train_micro_batch_size_per_gpu"] = config.rl_config.mini_batch_size
        assert deepspeed_config["bf16"]["enabled"], "Only bfloat16 is supported for now"

        import deepspeed
        engine, optimizer, _, lr_scheduler = deepspeed.initialize(
            model = model,
            model_parameters = model.parameters(),
            optimizer = optimizer,
            lr_scheduler = lr_scheduler,
            config = deepspeed_config,
        )
        training_setup["target_policy"] = engine 

    training_setup["optimizer"] = optimizer
    training_setup["lr_scheduler"] = lr_scheduler   

    return training_setup, TrainState(
        run_name=run_name,
        device=device,
        seed=config.base_config.seed,
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        reference_model=reference_model,
        kl_controller=kl_controller,
    )