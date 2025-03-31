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
    wandb_run_name: str = "defualt-run"
    wand_group_name:str = "no-ablation"
    
    # device params
    device_index: int = 0
    
    # model params
    model_name: str = "Qwen/Qwen2-0.5B-Instruct"
    model_max_length_inuse:int = 512
    checkpoint_path: Path = Path("output/")
    checkpoint_interval: int = 20


    # training params
    seed: int = 42
    group_size: int = 12
    lr: float = 1e-6 
    kl_weight: float = 0
    clip_eps: float = 0.2
    train_batch_size: int = 8
    epochs_per_step: int = 1
    max_norm: float = 1.0

    # sampling params
    max_new_tokens: int = 1024
    top_p: float = 0.9
    top_k: int = 50
    temperature: float = 1.0

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
    reference_model, _ = load_model_tokenizer(args.model_name, device_map=device, freeze_model=True)
    model, tokenizer = load_model_tokenizer(args.model_name, device_map=device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    reference_model.eval()
    model.gradient_checkpointing_enable()

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
        args.wandb_run_name = args.wandb_run_name + f"-batch_{args.train_batch_size}" + f"-group_{args.group_size}"
        wandb.init(
            project=args.wandb_project,
            name= args.wandb_run_name,
            group= args.wand_group_name,
        )

    for batch_idx, batch in enumerate(main_data_loader):
        
        print(f"Batch indx {batch_idx}")
        print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
        print("torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(0)/1024/1024/1024))
        print("torch.cuda.max_memory_reserved: %fGB"%(torch.cuda.max_memory_reserved(0)/1024/1024/1024))

        replay_buffer.clear()
        questions = batch["question"]
        answers = batch["answer"]

        batch_inputs = tokenize_questions(
            tokenizer=tokenizer,
            questions=questions,
            trucation_max_length=args.model_max_length_inuse,
        )
        batch_inputs = {
            k:v.to(device) \
            for k,v in batch_inputs.items()
        }
        max_input_length = batch_inputs["input_ids"].shape[1]

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

            batch_mean_returns = returns.mean().to(cpu_device)
            batch_mean_solved_rate = solved_rate.mean().to(cpu_device) 
            print(f"batch_idx: {batch_idx} | returns: {batch_mean_returns.item()} | solved_rate: {batch_mean_solved_rate.item()}")

            advantages = group_advantages(returns) # (num_samples, group_size)
            returns = returns.to(cpu_device)
            solved_rate = solved_rate.to(cpu_device)

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
            
            del sequence_ids
            del attention_mask
            del action_mask
            del log_probs
            del log_probs_ref
            del returns 
            del advantages

        torch.cuda.empty_cache()

        # log the stats to wandb 
        wandb.log(
            {
                "train/batch_returns": batch_mean_returns,
                "train/batch_solved_rate": batch_mean_solved_rate,
                "train/max_input_length": max_input_length,
            }
        )
        # TODO: log the text completions as artifacts
        out_dir = f"output/{args.wandb_run_name}"
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        file_name = os.path.join(
            out_dir,
            f"completions_{batch_idx}.txt"
        )
        text = ""
        with open(file_name, "w") as f:
            for i, completion in enumerate(completions):
                question = questions[i//args.group_size]
                text += "******" + "\n" + question + ":\n----\n" 
                text += completion + "\n******\n"
            f.write(text)
                
        wandb.save(file_name)

        ### Training phase of GRPO
        experience_sampler = DataLoader(
            replay_buffer,
            batch_size=args.train_batch_size,
            shuffle=True,
            drop_last=False,
            collate_fn=join_experience_batch,
        )

        for _ in range(args.epochs_per_step):
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
                del log_probs

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
                del grad_norm
                del exp 
                del loss 
                del mean_kl
                torch.cuda.empty_cache()

            del experience

        if (
            args.checkpoint_path is not None
            and args.checkpoint_interval is not None
            and (batch_idx + 1) % args.checkpoint_interval == 0
        ):
            model.save_pretrained(args.checkpoint_path / f"step_{batch_idx + 1}")

        del batch_inputs

    if args.checkpoint_path is not None:
        model.save_pretrained(args.checkpoint_path / f"step_{batch_idx + 1}")

if __name__ == "__main__":
    
    args = tyro.cli(CliArgs)
    train(args)