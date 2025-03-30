import reasoning_gym
from datasets import load_dataset
from torch.utils.data import Dataset

from math_verify import parse

from typing import List, Dict, Any


class GSM8KDataset(Dataset):
    """
    A dataset of GSM8K questions and answers.
    """
    def __init__(self, dataset_name: str = "openai/gsm8k", seed: int = 42, mode:str="train"):
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
        self.ds = load_dataset(dataset_name, "main")
        self.ds = self.ds.map(
            lambda x: self.get_answer_from_gt(x["answer"]),
            batched=False,
        )
        self.seed = seed
        self.ds = self.ds.shuffle(seed=self.seed)
    
        self.train = self.ds["train"]
        self.test = self.ds["test"]

        self.mode = mode

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

    def __getitem__(self, idx: int) -> Dict[str, str|List]:
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


class ReasoningGymDataset(Dataset):

    def __init__(
        self,
        datasets_name: List[str], 
        each_dataset_size: int, 
        seed: int
    ):
        """
        Initialize a ReasoningGymDataset instance.

        Parameters
        ----------
        datasets_name : List[str]
            The names of the datasets to load.
        size : int
            The number of examples to load from each dataset.
        seed : int
            The random seed to use when loading the datasets.
        """
        self.datasets_name = datasets_name
        self.each_dataset_size = each_dataset_size
        self.seed = seed
        self.each_dataset_size = each_dataset_size
        self.total_size = len(datasets_name) * each_dataset_size
        
        # Cache for lazy loading
        self._datasets = {}  
        for dataset_idx in range(len(datasets_name)):
            self._datasets[dataset_idx] = reasoning_gym.create_dataset(
                name=self.datasets_name[dataset_idx],
                size=self.each_dataset_size,
                seed=self.seed
            )

    def __len__(self):
        """
        Get the total number of examples in the dataset.

        This is the product of the number of datasets and the number of examples
        in each dataset.

        Returns
        -------
        int
            The total number of examples in the dataset.
        """
        return self.total_size

    def __getitem__(self, idx: int) -> Dict:
        """
        Get an item from the dataset by index.

        Parameters
        ----------
        idx : int
            The index of the item to retrieve.

        Returns
        -------
        Dict
            The item at the given index.
        """
        dataset_idx = idx // self.each_dataset_size
        item_idx = idx % self.each_dataset_size
        
        return {
            "question": self._datasets[dataset_idx][item_idx]["question"],
            "answer": self._datasets[dataset_idx][item_idx]["answer"],
            "dataset_name": self.datasets_name[dataset_idx],
            "dataset_idx": dataset_idx,
            "item_idx": item_idx,
            "global_idx": idx
        }