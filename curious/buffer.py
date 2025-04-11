from dataclasses import dataclass, fields
from typing import Optional, Self, List, Tuple

import torch
import torch.nn.functional as F

def zero_pad_sequences(sequences: List[torch.Tensor], side: str = "left") -> torch.Tensor:
    """
    Zero pad sequences to the same length.
    Args:
        sequences: A list of torch.Tensor objects.
        side: A string that can be either "left" or "right".
    Returns:
        A torch.Tensor object.
    """
    assert side in ("left", "right")
    max_len = max(
        seq.size(0) for seq in sequences
    )  # get the max length of the sequences
    padded_sequences = []
    for seq in sequences:
        pad_len = max_len - seq.size(0)  # get the padding length
        padding = (pad_len, 0) if side == "left" else (0, pad_len)  # get the padding
        padded_sequences.append(F.pad(seq, padding))  # pad the sequence
    return torch.stack(padded_sequences, dim=0)  # stack the padded sequences

@dataclass
class Experience:
    """
    A 1/2 Dimensional representation of a collection of experiences.
    """
    sequences: torch.Tensor  # (num_samples * group_size, seq_len)
    action_log_probs: torch.Tensor  # (num_samples * group_size, seq_len-1)
    attention_mask: torch.Tensor  # (num_samples * group_size, seq_len)
    action_mask: torch.Tensor  # (num_samples * group_size, seq_len-1)

    returns: torch.Tensor  # (num_samples * group_size)
    solved_mask: torch.Tensor  # (num_samples * group_size)
    advantages: torch.Tensor  # (num_samples * group_size, 1)

    kl: Optional[torch.Tensor]  = None # (num_samples * group_size, seq_len-1) (boolean)
    log_probs_ref: Optional[torch.Tensor] = None # (num_samples * group_size, seq_len-1)

    def to(self, device: torch.device) -> Self:
        members = {}
        for field in fields(self):
            v = getattr(self, field.name)
            if isinstance(v, torch.Tensor):
                v = v.to(device=device)
            members[field.name] = v
        return Experience(**members)
    
    keys = [
        "sequences",
        "action_log_probs",
        "log_probs_ref",
        "returns",
        "solved_mask",
        "advantages",
        "attention_mask",
        "action_mask",
    ]


def split_experience_batch(experience: Experience) -> List[Experience]:
    """
    Splits a single `Experience` object into a list of `Experience` objects,
    where each object corresponds to a single instance.

    Args:
        experience (Experience): The `Experience` object containing batched data.
            Each field in `experience` should be a tensor where the first dimension
            is the batch size, or None.

    Returns:
        List[Experience]: A list of `Experience` objects, each representing a
            single element from the batch. If a field in `experience` is None,
            the corresponding field in the output `Experience` objects will also be None.
    """
    # batch size is not the real batch size here
    # it is the number of samples in a batch (i.e. number of groups * num_samples)
    batch_size = experience.sequences.size(0)
    batch_data = [{} for _ in range(batch_size)]
    keys = experience.keys

    for key in keys:
        value:torch.Tensor = getattr(experience, key)
        if value is None:
            vals: List[None] = [None] * batch_size
        else:
            vals: Tuple[torch.Tensor, ...] = torch.unbind(value)
       
        for i, v in enumerate(vals):
            batch_data[i][key] = v

    # convert the batch data to a list of experiences
    return [Experience(**data) for data in batch_data]


def join_experience_batch(items: List[Experience]) -> Experience:
    """
    Joins a list of `Experience` objects into a single `Experience` object.

    Args:
        items (list[Experience]): A list of `Experience` objects, each containing
            individual batch data. Each field in these objects should be a tensor
            or None.

    Returns:
        Experience: A single `Experience` object with each field containing the
            concatenated batch data from the input `Experience` objects. If a
            field in the input objects is None, the corresponding field in the
            output `Experience` will also be None.
    """
    batch_data = {}
    keys = items[0].keys

    for key in keys:
        # get the values for this key from all the experiences within the batch
        vals: List[torch.Tensor] = [getattr(item, key) for item in items]
        
        # if all the values are not None, concatenate them
        if all(v is not None for v in vals):            
            data = torch.stack(vals, dim=0) #zero_pad_sequences(vals, "left")
            # for 1 dimensional data, stack the values and reshape them
            if key in {"returns", "solved_mask"}:
                data = data.reshape(-1)
            elif key == "advantages":
                data = data.reshape(-1, 1)
        else:
            data = None
        batch_data[key] = data

    return Experience(**batch_data)


class ReplayBuffer:
    """
    A buffer that stores experiences.
    """
    def __init__(self, limit: int = 0) -> None:
        """
        Initialize a ReplayBuffer with a given buffer limit.

        Args:
            limit: The maximum number of experiences to store in the buffer. If 0, no limit is applied.
        """
        self.limit = limit
        self.items: list[Experience] = []

    def append(self, experience: Experience) -> None:
        """
        Append an experience to the buffer.

        Args:
            experience (Experience): The experience to append.

        Returns:
            None
        """
        items = split_experience_batch(experience)
        self.items.extend(items)
        if self.limit > 0:
            samples_to_remove = len(self.items) - self.limit
            if samples_to_remove > 0:
                # remove the oldest experiences
                self.items = self.items[samples_to_remove:]

    def clear(self) -> None:
        """
        Clear the buffer.
        """
        self.items.clear()

    def __len__(self) -> int:
        """
        Get the number of experiences in the buffer.
        """
        return len(self.items)

    def __getitem__(self, idx: int) -> Experience:
        """
        Get an experience from the buffer.
        """
        return self.items[idx]
