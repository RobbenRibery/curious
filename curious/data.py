import reasoning_gym
from torch.utils.data import Dataset

from curious.utils import tokenize_questions

from typing import List, Dict, Callable


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