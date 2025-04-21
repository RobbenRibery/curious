from dataclasses import dataclass, fields
from typing import Optional, Self, List, Tuple, Any

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
    The first dimension is the batch size, the second dimension is the sequence length.
    The batch size is the number of samples in a batch (i.e. number of groups * num_samples)
    The sequence length is the length of the sequences in the batch.
    """
    sequences: torch.Tensor  # (num_samples * group_size, seq_len)
    attention_mask: torch.Tensor  # (num_samples * group_size, seq_len)
    action_mask: torch.Tensor  # (num_samples * group_size, seq_len-1)

    returns: torch.Tensor  # (num_samples * group_size)
    solved_mask: torch.Tensor  # (num_samples * group_size)
    advantages: torch.Tensor  # (num_samples * group_size, 1)

    action_log_probs: Optional[torch.Tensor]  = None # (num_samples * group_size, seq_len-1)
    kl: Optional[torch.Tensor]  = None # (num_samples * group_size, seq_len-1) (boolean)
    log_probs_ref: Optional[torch.Tensor] = None # (num_samples * group_size, seq_len-1)
    learnability: Optional[torch.Tensor] = None # (num_samples, )

    completion: Optional[List[str]] = None

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
        elif isinstance(value, torch.Tensor):
            vals: Tuple[torch.Tensor, ...] = torch.unbind(value)
        elif isinstance(value, list):
            vals: List[Any] = value
        else:
            raise ValueError(f"Unsupported type: {type(value)}")
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
    def __init__(self, items: List[Experience] = None, limit: int = 0) -> None:
        """
        Initialize a ReplayBuffer with a given buffer limit.

        Args:
            items: A list of experiences to initialize the buffer with.
            limit: The maximum number of experiences to store in the buffer. If 0, no limit is applied.
        """
        self.limit = limit
        self.items: list[Experience] = [item.to("cpu") for item in items] if items else []

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

        Returns:
            None
        """
        self.items.clear()

    def __len__(self) -> int:
        """
        Get the number of experiences in the buffer.

        Returns:
            int: The number of experiences in the buffer.
        """
        return len(self.items)

    def __getitem__(self, idx: int) -> Experience:
        """
        Get an experience from the buffer.

        Args:
            idx (int): The index of the experience to get.

        Returns:
            Experience: The experience at the given index.
        """
        return self.items[idx]
    
    def sample_for_learnability(self, num_samples:int, use_top_k:bool=False) -> Self:
        """
        Sample experiences from the buffer.

        Args:
            num_samples (int): The number of experiences to sample.
            use_top_k (bool): Whether to use the top k experiences.

        Returns:
            ReplayBuffer: A replay buffer of sampled experiences.
        """
        learnability = torch.stack([item.learnability for item in self.items], dim=0).reshape(-1)

        # sample the experiences
        if use_top_k:
            # sample the top k experiences
            indices = torch.topk(learnability, num_samples).indices
        else:
            # sample the experiences randomly
            learnability = F.softmax(learnability, dim=0)
            indices = torch.multinomial(
                weights=learnability,
                num_samples=num_samples,
                replacement=False,
            )

        sampled_experiences = [self.items[i] for i in indices]
        return ReplayBuffer(items=sampled_experiences)
