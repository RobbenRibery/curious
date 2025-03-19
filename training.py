import os 
os.environ["TOKENIZERS_PARALLELIS"] = "true"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch 
from torch.utils.data import DataLoader 
from torch.nn.utils import clip_grad_norm_
import torch.optim as optim

from transformers import GenerationConfig

from curious.data import ReasoningGymDataset
from curious.utils import tokenize_questions, load_model_tokenizer
from curious.grpo import rollout, sequences_log_probs, group_advantages
from curious.buffer import ReplayBuffer, Experience, join_experience_batch
from curious.loss import GRPOLoss, approx_kl_divergence

from lightning import seed_everything
import wandb

from dataclasses import dataclass
from pathlib import Path
import tyro 

@dataclass
class CliArgs: 
    # wandb params
    wandb_project: str = "curious-training-grpo-test"
    
    # device params
    device_index: int = 0
    
    # model params
    model_name: str = "Qwen/Qwen2-0.5B-Instruct"
    model_max_length_inuse:int = 512
    checkpoint_path: Path = Path("output/")
    checkpoint_interval: int = 20


    # training params
    seed: int = 42
    group_size: int = 16
    lr: float = 5e-7
    kl_weight: float = 0.01
    clip_eps: float = 0.2
    train_batch_size: int = 8
    epochs_per_step: int = 4
    max_norm: float = 2.0

    # sampling params
    max_new_tokens: int = 512
    top_p: float = 0.9
    top_k: int = 50
    temperature: float = 0.7

    # data params
    each_dataset_size: int = 1000

def train(args:CliArgs) -> None:

    # datasets
    datasets_name = [
        "gsm_symbolic"
    ]

    device = torch.device("cuda", args.device_index)
    cpu_device = torch.device("cpu")
    seed_everything(args.seed)

    # load models
    reference_model, _ = load_model_tokenizer(args.model_name, device_map=device)
    model, tokenizer = load_model_tokenizer(args.model_name, device_map=device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    reference_model.eval()
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={
            "use_reentrant": False,
        }
    )

    pad_token_id = tokenizer.eos_token_id

    # data
    dataset = ReasoningGymDataset(
        datasets_name=datasets_name,
        each_dataset_size=args.each_dataset_size,
        seed=args.seed,
    )

    main_data_loader = DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        drop_last=True,
    )
    
    # replay buffer
    replay_buffer = ReplayBuffer()

    # objective
    objective = GRPOLoss(
        clip_eps=args.clip_eps,
        kl_weight=args.kl_weight,
    )

    if not args.wandb_project:
        wandb.init(mode="disabled")
    else:
        wandb.init(
            project=args.wandb_project,
            name= args.wandb_run_name,
        )

    for batch_idx, batch in enumerate(main_data_loader):
        replay_buffer.clear()
        questions = batch["question"]
        answers = batch["answer"]

        batch_inputs = tokenize_questions(
            tokenizer=tokenizer,
            questions=questions,
            trucation_max_length=args.model_max_length_inuse,
        )
        batch_inputs = {
            k:v.to(device) for k,v in batch_inputs.items()
        }

        # Rollout phase of GRPO
        with torch.no_grad():
            # rollout
            sequence_ids, returns, solved_rate, action_mask, completions = rollout(
                model,
                tokenizer,
                batch_inputs,
                oracle_answers=answers,
                generation_config=GenerationConfig(
                    num_return_sequences=args.group_size,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    do_sample =True,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id= tokenizer.eos_token_id,
                ),
            )
            # print(sequence_ids.shape)
            # sequence_ids: (num_samples * group_size, seq_len)
            # action_mask: (num_samples * group_size, seq_len)
            # completions: (num_samples * group_size)
            # returns: (num_samples * group_size)
            # solved_rate: (num_samples, )

            batch_mean_returns = returns.mean()
            batch_mean_solved_rate = solved_rate.mean() 
            print(f"batch_idx: {batch_idx} | returns: {batch_mean_returns.item()} | solved_rate: {batch_mean_solved_rate.item()}")

            advantages = group_advantages(returns) # (num_samples, group_size)
            attention_mask = sequence_ids != pad_token_id # (num_samples * group_size, seq_len)

            log_probs = sequences_log_probs(
                model=model,
                sequence_ids=sequence_ids,
                attention_mask=attention_mask,
            ) # (num_samples * group_size, seq_len-1)
            log_probs_ref = sequences_log_probs(
                model=reference_model,
                sequence_ids=sequence_ids,
                attention_mask=attention_mask,
            ) # (num_samples * group_size, seq_len-1)
            kl = approx_kl_divergence(
                log_probs=log_probs,
                log_probs_ref=log_probs_ref,
                action_mask=action_mask,
            ) # (num_samples * group_size, seq_len-1)

            experience = Experience(
                sequences=sequence_ids,
                action_log_probs=log_probs,
                log_probs_ref=log_probs_ref,
                returns=returns,
                solved_rate = solved_rate,
                advantages=advantages,
                attention_mask=attention_mask,
                action_mask=action_mask,
                kl=kl,
            )
            replay_buffer.append(experience.to(cpu_device))

        torch.cuda.empty_cache()

        # log the stats to wandb 
        wandb.log(
            {
                "train/batch_returns": batch_mean_returns,
                "train/batch_solved_rate": batch_mean_solved_rate,
            }
        )
        # TODO: log the text completions as artifacts
        with open(f"output/completions_{batch_idx}.txt", "w") as f:
            for i, completion in enumerate(completions):
                f.write(f"******** completion {i} *********\n {completion}\n")
        wandb.save(f"output/completions_{batch_idx}.txt")

        ### Training phase of GRPO
        experience_sampler = DataLoader(
            replay_buffer,
            batch_size=args.train_batch_size,
            shuffle=True,
            drop_last=False,
            collate_fn=join_experience_batch,
        )

        for step_epoch in range(args.epochs_per_step):
            model.train()

            for exp in experience_sampler:
                exp: Experience
                exp = exp.to(device)

                optimizer.zero_grad()

                log_probs = sequences_log_probs(
                    model, 
                    sequence_ids=exp.sequences, 
                    attention_mask=exp.attention_mask
                )

                loss, mean_kl = objective(log_probs=log_probs, experience=exp)

                if not loss.isfinite():
                    print(f"Loss not finite, skipping backward, loss={loss}")
                    print(f"experience.advantages={experience.advantages}")
                    continue
                
                loss: torch.Tensor
                loss.backward()

                grad_norm = clip_grad_norm_(
                    model.parameters(), 
                    max_norm=args.max_norm,
                )
                #print(f"{step_epoch}: kl={kl: .4f}, grad_norm={grad_norm: .4f}")
                wandb.log(
                    {
                        "train/mean_kl": mean_kl, 
                        "train/grad_norm": grad_norm,
                        "train/loss": loss,
                    }
                )
                optimizer.step()

        if (
            args.checkpoint_path is not None
            and args.checkpoint_interval is not None
            and (batch_idx + 1) % args.checkpoint_interval == 0
        ):
            model.save_pretrained(args.checkpoint_path / f"step_{batch_idx + 1}")

    if args.checkpoint_path is not None:
        model.save_pretrained(args.checkpoint_path / f"step_{batch_idx + 1}")

if __name__ == "__main__":
    
    args = tyro.cli(CliArgs)
    train(args)