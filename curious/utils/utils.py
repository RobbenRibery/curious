from transformers import (
    PreTrainedTokenizer,
    PreTrainedModel,
    AutoTokenizer,
    AutoModelForCausalLM,
)
import datasets
from typing import List, Dict, Optional, Union, Any, Tuple
import torch

from curious.prompt import * 

import gc 


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


def release_memory(vars:List[Any]):
    for var in vars:
        del var
    gc.collect()
    torch.cuda.empty_cache()


def move_paddings_to_right(
    attention_mask: torch.Tensor,
    sequence_ids: torch.Tensor,
    pad_token_id: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Moves padding tokens from the left side to the right side of sequences.

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
    # get the least left padded amount 
    right_padded_seq_ids = []
    for attn_mask, seq_id in zip(attention_mask, sequence_ids):
        left_padding_amount = (attn_mask == 0).sum()
        right_padding_tensor = torch.full(
            (left_padding_amount,), 
            pad_token_id, 
            dtype=torch.long, 
            device=seq_id.device
        )
        new_seq_id = torch.cat(
            [
                seq_id[left_padding_amount:],
                right_padding_tensor,
            ],
            dim=0,
        )
        right_padded_seq_ids.append(new_seq_id)
    
    return torch.stack(right_padded_seq_ids)



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
) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Loads a pre-trained model and its tokenizer.

    Args:
        model_name_or_path (str): The name or path of the pre-trained model.
        trust_remote_code (bool): Whether to trust remote code.
        bf16 (bool): Whether to use bfloat16 precision.
        device_map (str): The device map to use for the model.

    Returns:
        tuple: A tuple containing the model and its tokenizer.
    """
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    if torch.cuda.is_available():
        from liger_kernel.transformers import AutoLigerKernelForCausalLM
        model: PreTrainedModel = AutoLigerKernelForCausalLM.from_pretrained(
            model_name_or_path if checkpoint_path is None else checkpoint_path,
            trust_remote_code=trust_remote_code,
            attn_implementation="flash_attention_2",
            torch_dtype=dtype_,
            device_map=device_map,
        )
    else:
        model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            model_name_or_path if checkpoint_path is None else checkpoint_path,
            trust_remote_code=trust_remote_code,
            torch_dtype=dtype_,
        )
    model.forward = torch.compile(model.forward, dynamic=True)
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
