from typing import List, Self, Tuple

import torch
from dataclasses import dataclass, fields
from torch.utils.data import Dataset

@dataclass
class Curriculum:
    """
    A dataclass for storing the curriculum.
    """
    question: List[str]
    """The quesitons provided"""

    oracle_answer: List[str]
    """The oracle answers"""
    
    learnability: torch.Tensor
    """The learnability scores"""

    def to(self, device: torch.device) -> Self:
        members = {}
        for field in fields(self):
            v = getattr(self, field.name)
            if isinstance(v, torch.Tensor):
                v = v.to(device=device)
            members[field.name] = v
        return Curriculum(**members)
    

class CurriculumBuffer:
    """
    A buffer for storing the curriculum.
    """ 
    def __init__(self, buffer: List[Curriculum] = None) -> None:
        self.buffer = buffer if buffer is not None else []
        self.keys = [
            "question",
            "oracle_answer",
            "learnability",
        ]

    def append(self, curriculum: Curriculum) -> None:
        """
        Append a curriculum to the buffer.
        """
        num_samples = len(curriculum.question)
        individual_curriculums:List[Curriculum] = [{} for _ in range(num_samples)]

        for key in self.keys:
            print(key)
            if key not in curriculum.__dict__:
                raise ValueError(f"Curriculum must have a {key} attribute")
            
            values = getattr(curriculum, key)
            if isinstance(values, torch.Tensor):
                values: Tuple[torch.Tensor, ...] = torch.unbind(values)
            
            for i, v in enumerate(values):
                individual_curriculums[i][key] = v

        self.buffer.extend([Curriculum(**data) for data in individual_curriculums])

    def clear(self) -> None:
        self.buffer.clear()

    def __len__(self) -> int:
        return len(self.buffer) 
    
    def __getitem__(self, idx: int) -> Curriculum:
        return self.buffer[idx]
    
    def to_dataset(self) -> Dataset:
        pass 