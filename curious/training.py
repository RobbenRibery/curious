from curious.config import TrainingConfig
from curious.train.training_setup import set_up_training
from curious.train.trainer import PolicyGradientTrainer

import tyro
import wandb

if __name__ == "__main__":
    args = tyro.cli(TrainingConfig)
    training_setup, init_train_state = set_up_training(args)

    wandb.init(
        entity=args.wandb_config.entity,
        project=args.wandb_config.project,
        name=args.wandb_config.name,
        group=args.wandb_config.group,
        config=args,
    )
    wandb.define_metric("num_batches_visited")
    wandb.define_metric("train/*", step_metric="num_batches_visited")
    wandb.define_metric("eval/*", step_metric="num_batches_visited")
    wandb.define_metric("ad_cispo/*", step_metric="num_batches_visited")
    wandb.define_metric("cw_cispo/*", step_metric="num_batches_visited")

    trainer = PolicyGradientTrainer(training_setup)
    trainer.logger = wandb.log
    trainer.train(init_train_state)
    wandb.finish()
