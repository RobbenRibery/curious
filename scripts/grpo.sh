make clean;
export TOKENIZERS_PARALLELISM=true;
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128,expandable_segments:True";


TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS=1 python3 training.py \
    --wandb_config.project "curious-grpo-gsm8k" \
    --wandb_config.group "grpo-test" \
    --wandb_config.name "grpo-qwen25-prompt[qwen]-reward[partial-solved]" \
    --base_config.model_name "Qwen/Qwen2.5-0.5B-Instruct" \
    --base_config.device_index 0 \
    --base_config.dataset_name "openai/gsm8k" \
    --base_config.train_batch_size 8 \
    --base_config.eval_batch_size 256 \
    --base_config.num_workers 16 \
    --base_config.train_log_dir "train_logs" \
    --base_config.eval_log_dir "eval_logs" \
    --base_config.seed 42 \
    --base_config.checkpoint_dir "checkpoints" \
    --base_config.checkpoint_interval 100 \
    --base_config.eval_interval 50 \
    --base_config.train_text_log_interval 50 \
    --base_config.eval_text_log_interval 100 \
    --sampling_config.model_prompt_length 1024 \
    --sampling_config.max_new_tokens 1024 \
    --sampling_config.temperature 1.0 \
    --sampling_config.top_p 1.0 \
    --sampling_config.top_k 100 \
    --sampling_config.do_sample \
    --sampling_config.use_cache \
    --sampling_config.repetition_penalty 1.0 \
    --sampling_config.system_prompt "qwen_system_prompt" \
    --reward_config.no-use-format-reward \
    --grpo_config.group_size 16 \
    --grpo_config.lr 1e-06 \
    --grpo_config.weight_decay 0.0 \
    --grpo_config.kl_weight 0.0 \
    --grpo_config.clip_eps 0.2 \
    --grpo_config.mini_batch_size 64 \
    --grpo_config.epochs_per_step 2 \