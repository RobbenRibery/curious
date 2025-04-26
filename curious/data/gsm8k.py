import datasets
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from curious.utils.utils import tokenize_questions
from curious.prompt import *

from typing import List, Dict


class GSM8KDataset(Dataset):
    """
    A dataset of GSM8K questions and answers.
    """

    def __init__(
        self, 
        tokenizer: PreTrainedTokenizer,
        dataset_name: str = "openai/gsm8k", 
        seed: int = 42, 
        mode: str = "train", 
        max_prompt_length: int = 1024,
        system_prompt: str = qwen_system_prompt,
    ):
        """
        Initialize the GSM8K dataset.

        Args:
            dataset_name (str, optional): The name of the dataset to load.
                Defaults to "openai/gsm8k".
            seed (int, optional): The seed to use for shuffling the dataset.
                Defaults to 42.
            mode (str, optional): The mode to use for the dataset, either "train" or "test".
                Defaults to "train".
        """
        self.ds: datasets.Dataset = load_dataset(dataset_name, "main")
        self.ds: datasets.Dataset = self.ds.map(
            lambda x: self.get_answer_from_gt(x["answer"]),
            batched=False,
        )
        self.seed = seed
        self.ds: datasets.Dataset = self.ds.shuffle(seed=self.seed)

        self.train: datasets.Dataset = self.ds["train"]
        self.test: datasets.Dataset = self.ds["test"]

        self.mode = mode

        ## pre-tokenize the dataset ## 
        self.max_prompt_length = max_prompt_length

        self.tokenizer = tokenizer
        self.tokenizer.padding_side = "left"

        ## Train dataset ##
        train_max_length = max(
            len(self.tokenizer(x["question"])["input_ids"])
            for x in self.train
        )
        print(f"Detected train_max_length: {train_max_length}")
        self.train_max_length = train_max_length if train_max_length <= self.max_prompt_length else self.max_prompt_length
        print(f"Setting train_max_length to {self.train_max_length}")

        self.train = self.train.map(
            lambda x: tokenize_questions(
                self.tokenizer, 
                x["question"],
                max_length=self.train_max_length,
                allow_dynamic_padding=False,
                system_prompt=system_prompt,
            ),
            batched=True,
        )
        self.train.set_format(
            type="pt",
            columns=["input_ids", "attention_mask"],
            output_all_columns=True,
        )

        ## Test dataset ##
        test_max_length = max(
            len(self.tokenizer(x["question"])["input_ids"])
            for x in self.test
        )
        print(f"Detected test_max_length: {test_max_length}")
        self.test_max_length = test_max_length if test_max_length <= self.max_prompt_length else self.max_prompt_length
        print(f"Setting test_max_length to {self.test_max_length}")
    
        self.test = self.test.map(
            lambda x: tokenize_questions(
                self.tokenizer, 
                x["question"],
                max_length=self.test_max_length,
                allow_dynamic_padding=False,
                system_prompt=system_prompt,
            ),
            batched=True,
        )

        self.test.set_format(
            type="pt",
            columns=["input_ids", "attention_mask"],
            output_all_columns=True,
        )

    def get_answer_from_gt(self, answer_text: str) -> Dict[str, str]:
        """
        This function is strict that it will guarantee to find a
        valid answer in the given answer_text, provided that the answer
        text from the GSM8K Dataset (not generated answer)
        Any violation of the format will raise an error.

        The ground truth format is a single string with the following rules:

        1. The last line should start with "####"
        2. The last line should contain only digits

        (Works only on GSM8K data)

        Args:
            answer_text (str): The answer text from the GMS8K Dataset

        Returns:
            A dictionary with a single key "answer_str_digit" and the
            corresponding value as the digit-only answer string.
        """
        lines = answer_text.strip().split("\n")

        if "####" not in lines[-1]:
            raise ValueError(f"Ill-formed answer provided: {answer_text}")

        answer_str: str = lines[-1].replace("####", "").strip()
        answer_str_digit = answer_str.replace(",", "")

        try:
            eval(answer_str_digit)
        except Exception as e:
            raise ValueError(f"Ill-formed answer provided: {answer_str}") from e

        return {"oracle_answer": answer_str_digit}

    def __len__(self):
        """
        Get the length of the dataset.

        Returns
        -------
        int : The length of the dataset.

        Raises
        ------
        ValueError : If the mode is invalid.
        """
        if self.mode == "train":
            return len(self.train)
        elif self.mode == "test":
            return len(self.test)
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

    def __getitem__(self, idx: int) -> Dict[str, str | List]:
        """
        Get an item from the dataset by index.

        Returns
        -------
        Dict[str, str|List]: The item at the given index.
        """
        if self.mode == "train":
            return self.train[idx]
        elif self.mode == "test":
            return self.test[idx]
        else:
            raise ValueError(f"Invalid mode: {self.mode}")
