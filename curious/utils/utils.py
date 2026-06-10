from dataclasses import dataclass

from datasets.utils.tf_utils import minimal_tf_collate_fn_with_renaming
from transformers import (
    PreTrainedTokenizer,
    PreTrainedModel,
    AutoTokenizer,
    AutoModelForCausalLM,
)
import datasets
from typing import Callable, List, Dict, Optional, Union, Any, Tuple
import torch

from curious.prompt import * 

import gc 
import os


LOGGING_TEMPLATE = dedent(
"""
******************************
Question: 
{question}
----
Answer: 
{answer}
----
Completion: 
{completion}
----
Reward: 
{reward}
----
Info: 
{info}
******************************

"""
).strip()

COMPLETION_SAMPLE_COLUMNS = [
    "phase",
    "batch_idx",
    "sample_idx",
    "question",
    "answer",
    "completion",
    "reward",
    "info",
]


def configure_bf16_precision() -> None:
    os.environ.setdefault("NVIDIA_TF32_OVERRIDE", "0")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("highest")


@dataclass(frozen=True)
class CudaMemorySnapshot:
    allocated_bytes: int
    reserved_bytes: int
    max_reserved_bytes: int
    total_bytes: int

    @property
    def reserved_ratio(self) -> float:
        if self.total_bytes <= 0:
            return 0.0
        return self.reserved_bytes / self.total_bytes

    @property
    def fragmentation_ratio(self) -> float:
        if self.allocated_bytes <= 0:
            return 0.0 if self.reserved_bytes <= 0 else float("inf")
        return self.reserved_bytes / self.allocated_bytes


@dataclass(frozen=True)
class CleanupPolicy:
    batch_interval: int = 1
    minibatch_empty_cache_interval: int = 0
    reserved_ratio_threshold: float = 0.90
    fragmentation_ratio: float = 1.35
    force_after_eval: bool = True
    force_after_checkpoint: bool = True


@dataclass(frozen=True)
class CleanupDecision:
    drop_refs: bool
    run_gc: bool
    empty_cache: bool
    reason: str


@dataclass(frozen=True)
class PerfTrace:
    phase: str
    elapsed_seconds: float
    memory_before: CudaMemorySnapshot
    memory_after: CudaMemorySnapshot


def cuda_memory_snapshot(device: Optional[Union[torch.device, int, str]] = None) -> CudaMemorySnapshot:
    if not torch.cuda.is_available():
        return CudaMemorySnapshot(
            allocated_bytes=0,
            reserved_bytes=0,
            max_reserved_bytes=0,
            total_bytes=0,
        )
    if device is None:
        device_index = torch.cuda.current_device()
    elif isinstance(device, int):
        device_index = device
    else:
        resolved_device = torch.device(device)
        device_index = resolved_device.index
        if device_index is None:
            device_index = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device_index)
    return CudaMemorySnapshot(
        allocated_bytes=torch.cuda.memory_allocated(device_index),
        reserved_bytes=torch.cuda.memory_reserved(device_index),
        max_reserved_bytes=torch.cuda.max_memory_reserved(device_index),
        total_bytes=properties.total_memory,
    )


def should_empty_cuda_cache(snapshot: CudaMemorySnapshot, policy: CleanupPolicy) -> bool:
    return (
        snapshot.reserved_ratio >= policy.reserved_ratio_threshold
        or snapshot.fragmentation_ratio >= policy.fragmentation_ratio
    )


def decide_cleanup(
    snapshot: CudaMemorySnapshot,
    policy: CleanupPolicy,
    *,
    phase: str,
    batch_idx: int = 0,
    minibatch_idx: int = 0,
    force: bool = False,
) -> CleanupDecision:
    run_gc = False
    empty_cache = False
    reason = phase

    if force or phase == "oom":
        return CleanupDecision(drop_refs=True, run_gc=True, empty_cache=True, reason=f"{phase}:forced")

    if phase == "minibatch":
        interval = policy.minibatch_empty_cache_interval
        empty_cache = interval > 0 and minibatch_idx > 0 and minibatch_idx % interval == 0
        return CleanupDecision(
            drop_refs=True,
            run_gc=False,
            empty_cache=empty_cache and should_empty_cuda_cache(snapshot, policy),
            reason=reason,
        )

    if phase == "after_eval" and policy.force_after_eval:
        return CleanupDecision(drop_refs=True, run_gc=True, empty_cache=True, reason=reason)
    if phase == "after_checkpoint" and policy.force_after_checkpoint:
        return CleanupDecision(drop_refs=True, run_gc=True, empty_cache=True, reason=reason)

    interval = policy.batch_interval
    run_gc = interval > 0 and batch_idx > 0 and batch_idx % interval == 0
    empty_cache = should_empty_cuda_cache(snapshot, policy)
    return CleanupDecision(drop_refs=True, run_gc=run_gc, empty_cache=empty_cache, reason=reason)


def drop_refs(*objects: Any) -> None:
    del objects


def clear_python_memory() -> None:
    gc.collect()


def clear_cuda_cache_if_needed(decision: CleanupDecision) -> None:
    if decision.empty_cache and torch.cuda.is_available():
        torch.cuda.empty_cache()


def execute_cleanup(decision: CleanupDecision) -> None:
    if decision.run_gc:
        clear_python_memory()
    clear_cuda_cache_if_needed(decision)


def _to_loggable_scalar(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "item"):
        return value.item()
    return value


def build_completion_sample_rows(
    *,
    phase: str,
    batch_idx: int,
    questions: List[Any],
    answers: List[Any],
    completions: List[str],
    rewards: List[Any],
    infos: List[Any],
    max_samples: int,
    sample_offset: int = 0,
) -> List[Dict[str, Any]]:
    if max_samples <= 0:
        return []

    rows: List[Dict[str, Any]] = []
    for local_idx, (question, answer, completion, reward, info) in enumerate(
        zip(questions, answers, completions, rewards, infos)
    ):
        if len(rows) >= max_samples:
            break
        rows.append(
            {
                "phase": phase,
                "batch_idx": batch_idx,
                "sample_idx": sample_offset + local_idx,
                "question": str(question),
                "answer": str(answer),
                "completion": completion,
                "reward": _to_loggable_scalar(reward),
                "info": str(info),
            }
        )
    return rows


def format_completion_sample_rows(rows: List[Dict[str, Any]]) -> str:
    return "\n".join(
        LOGGING_TEMPLATE.format(
            question=row["question"],
            answer=row["answer"],
            completion=row["completion"],
            reward=row["reward"],
            info=row["info"],
        )
        for row in rows
    )


def log_completion_sample_table(
    *,
    logger: Callable[[Dict[str, Any]], None],
    key: str,
    rows: List[Dict[str, Any]],
) -> None:
    if not rows:
        return

    data = [[row[column] for column in COMPLETION_SAMPLE_COLUMNS] for row in rows]
    try:
        import wandb

        value: Any = wandb.Table(columns=COMPLETION_SAMPLE_COLUMNS, data=data)
    except Exception:
        value = data

    logger(
        {
            key: value,
            "num_batches_visited": rows[0]["batch_idx"],
        }
    )


def release_memory(vars:List[Any]):
    vars.clear()
    execute_cleanup(CleanupDecision(drop_refs=True, run_gc=True, empty_cache=True, reason="legacy_release_memory"))


def move_paddings_to_right(
    input_ids:torch.Tensor,
    attention_mask: torch.Tensor,
    sequence_ids: torch.Tensor,
    pad_token_id: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Moves padding tokens from the left side to the right side of sequences.
    And then truncate the unecessary padding tokens

    This function takes sequences with left padding and rearranges them to have right padding
    instead, while preserving the actual content of the sequences.

    Args:
        attention_mask (torch.Tensor): Binary mask indicating which tokens are padding (0) and which are content (1).
            Shape: [batch_size, prompt_seq_len]
        sequence_ids (torch.Tensor): Token IDs for the sequences, including both prompt and answer parts.
            Shape: [batch_size, prompt_seq_len + answer_seq_len]
        pad_token_id (int): The token ID used for padding.

    Returns:
        torch.Tensor: The rearranged sequence IDs with padding moved to the right.
            Shape: [batch_size, prompt_seq_len + answer_seq_len]

    Example:
        >>> attention_mask = torch.tensor([[0, 0, 1, 1, 1], [0, 1, 1, 1, 1]])
        >>> sequence_ids = torch.tensor([[128, 128, 1, 2, 3], [128, 1, 2, 3, 4]])
        >>> pad_token_id = 128
        >>> result = move_paddings_to_right(attention_mask, sequence_ids, pad_token_id)
    """
    left_padded_prompt_length = input_ids.shape[1]

    # expand the attention mask 
    group_size = sequence_ids.shape[0] // attention_mask.shape[0]
    attention_mask = attention_mask.repeat_interleave(
        repeats = group_size, 
        dim = 0 
    ).to(device=sequence_ids.device)
    left_padding_amounts = (attention_mask == 0).sum(dim=1)
    seq_len = sequence_ids.shape[1]
    positions = torch.arange(seq_len, device=sequence_ids.device).unsqueeze(0)
    gather_indices = (positions + left_padding_amounts.unsqueeze(1)).remainder(seq_len)
    right_padded_seq_ids = sequence_ids.gather(dim=1, index=gather_indices)

    new_action_mask = positions >= (left_padded_prompt_length - left_padding_amounts).unsqueeze(1)
    new_action_mask = new_action_mask.to(device=sequence_ids.device, dtype=torch.bool)
    new_action_mask[right_padded_seq_ids == pad_token_id] = False
    minimum_right_padding_tokens = (right_padded_seq_ids == pad_token_id).sum(dim=1).min().item()
    new_seq_length = right_padded_seq_ids.shape[1] - minimum_right_padding_tokens

    new_seq_ids = right_padded_seq_ids[:,:new_seq_length]
    new_action_mask = new_action_mask[:, :new_seq_length]

    new_action_mask = new_action_mask[:, 1:]
    return new_seq_ids, new_action_mask


def form_hf_dataset(
    tokenizer: PreTrainedTokenizer,
    data:List[Dict[str, Any]],
    seed: int = 42,
    max_prompt_length: int = 1024,
    system_prompt: str = qwen_system_prompt,
) -> datasets.Dataset:
    """
    Form a Hugging Face dataset from a dataset name.
    
    Args:
        tokenizer (PreTrainedTokenizer): The tokenizer to use.
        data (List[Dict[str, Any]]): The data to form the dataset from.
        seed (int): The seed to use for the dataset.
        max_prompt_length (int): The maximum length of the prompt.
        system_prompt (str): The system prompt to use.

    Returns:
        datasets.Dataset: The huggingface dataset.
    """
    df_dataset = datasets.Dataset.from_list(data)
    df_dataset = df_dataset.map(
        lambda x: tokenize_questions(
            tokenizer, 
            x["question"], 
            max_length=max_prompt_length, 
            allow_dynamic_padding=False, 
            system_prompt=system_prompt),
        batched=True,
    )
    df_dataset.set_format(
        type="pt",
        columns=["input_ids", "attention_mask"],
        output_all_columns=True,
    )
    df_dataset = df_dataset.shuffle(seed=seed)
    return df_dataset

def load_model_tokenizer(
    model_name_or_path: str,
    trust_remote_code: bool = True,
    dtype_: torch.dtype = torch.bfloat16,
    device_map: Optional[str] = "auto",
    freeze_model: bool = False,
    checkpoint_path: Optional[str] = None,
    compile_model: bool = True,
    use_liger: bool = True,
) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Loads a pre-trained model and its tokenizer.

    Args:
        model_name_or_path (str): The name or path of the pre-trained model.
        trust_remote_code (bool): Whether to trust remote code.
        dtype_ (torch.dtype): The floating dtype to use for model weights.
        device_map (str): The device map to use for the model.

    Returns:
        tuple: A tuple containing the model and its tokenizer.
    """
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    resolved_model_path = model_name_or_path if checkpoint_path is None else checkpoint_path

    if torch.cuda.is_available():
        if use_liger:
            from liger_kernel.transformers import AutoLigerKernelForCausalLM
            try:
                model: PreTrainedModel = AutoLigerKernelForCausalLM.from_pretrained(
                    resolved_model_path,
                    trust_remote_code=trust_remote_code,
                    attn_implementation="flash_attention_2",
                    torch_dtype=dtype_,
                    device_map=device_map,
                )
            except KeyError as exc:
                print(f"Liger does not support this model type ({exc}); falling back to AutoModelForCausalLM.")
                model = AutoModelForCausalLM.from_pretrained(
                    resolved_model_path,
                    trust_remote_code=trust_remote_code,
                    torch_dtype=dtype_,
                    device_map=device_map,
                )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                resolved_model_path,
                trust_remote_code=trust_remote_code,
                torch_dtype=dtype_,
                device_map=device_map,
            )
    else:
        model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            resolved_model_path,
            trust_remote_code=trust_remote_code,
            torch_dtype=dtype_,
        )
    model.to(dtype=dtype_)
    if compile_model:
        model.forward = torch.compile(model.forward, dynamic=True)
    model.generation_config.cache_implementation = "static"
    if freeze_model:
        for param in model.parameters():
            param.requires_grad = False

    return model, tokenizer

def tokenize_questions(
    tokenizer: PreTrainedTokenizer,
    questions: List[str],
    max_length: int = None,
    allow_dynamic_padding: bool = False,
    system_prompt: str = deepseek_system_prompt,
) -> Dict[str, torch.Tensor]:
    """
    Tokenize a list of questions and answers.
    """
    if not max_length:
        print(f"Using model max length: {tokenizer.model_max_length}")
        max_length = tokenizer.model_max_length

    if tokenizer.padding_side == "right":
        print(f"Adjusting padding side from right to left for training")
        # training padding should be on the left
        tokenizer.padding_side = "left"

    # add the DeepSeek system prompt to the conversation
    questions = [
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]
        for question in questions
    ]

    # Apply the chat template to the questions
    texts = tokenizer.apply_chat_template(
        conversation=questions,
        tokenize=False,
        add_generation_prompt=True,
    )

    # get encodings for the question
    encodings = tokenizer(
        texts,
        padding="longest" if allow_dynamic_padding else "max_length",  # padding to the longest sequence in the batch
        truncation=True,
        return_tensors="pt",
        max_length=max_length,
        padding_side="left",
        return_attention_mask=True,
    )
    
    # add the prompt to the output dict
    encodings["prompt"] = texts
    return encodings
