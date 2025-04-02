python training.py \
    --wandb_config.name "grpo-reproduction" \
    --base_config.checkpoint_path "checkpoints" \
    --base_config.checkpoint_interval 100 \
    --base_config.mode "train" \
    --base_config.dataset_name "openai/gsm8k" \
    --base_config.model_name "Qwen/Qwen2-0.5B-Instruct" \
    --base_config.device_index 0 \
    --base_config.num_workers 8 \
    --base_config.seed 42 \
    --base_config.log_dir "train_logs" \
    --base_config.batch_size 8 \
    --grpo_config.mini_batch_size 8 \
    --grpo_config.epochs_per_step 2 \
    --grpo_config.group_size 12 \
    --grpo_config.lr 1e-6 \
    --grpo_config.kl_weight 0.01 \
    --grpo_config.clip_eps 0.2 \
    --reward_config.use_format_reward 