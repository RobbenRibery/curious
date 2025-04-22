from transformers import (
    PreTrainedTokenizer,
    PreTrainedModel,
    AutoTokenizer,
    AutoModelForCausalLM,
)
import datasets
from typing import List, Dict, Optional, Union, Any
import torch

from curious.prompt import * 
from liger_kernel.transformers import (
    AutoLigerKernelForCausalLM
)

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
    bf16: bool = True,
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
    tokenizer.pad_token = tokenizer.eos_token


    model: PreTrainedModel = AutoLigerKernelForCausalLM.from_pretrained(
        model_name_or_path if checkpoint_path is None else checkpoint_path,
        trust_remote_code=trust_remote_code,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16 if bf16 else "auto",
        device_map=device_map,
    )
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
    )

    return encodings
