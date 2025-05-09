from curious.config import TrainingConfig
from curious.train.training_setup import set_up_training
from curious.train.trainer import PolicyGradientTrainer

import tyro
import wandb

if __name__ == "__main__":
    
    args = tyro.cli(TrainingConfig)
    training_setup, init_train_state = set_up_training(args)
    
    wandb.init(
        entity="moed",
        project=args.wandb_config.project,
        name=args.wandb_config.name,
        config=args,
    )
    wandb.define_metric("num_batches_visited")
    wandb.define_metric("train/mean_batch_returns", step_metric="num_batches_visited")
    wandb.define_metric("train/mean_batch_solved_rate", step_metric="num_batches_visited")
    wandb.define_metric("train/max_input_length", step_metric="num_batches_visited")
    wandb.define_metric("train/mean_num_words_in_completions", step_metric="num_batches_visited")
    wandb.define_metric("train/mean_batch_format_returns", step_metric="num_batches_visited")
    wandb.define_metric("train/mean_batch_outcome_returns", step_metric="num_batches_visited")

    trainer = PolicyGradientTrainer(training_setup)
    trainer.train(init_train_state)
   
        
        