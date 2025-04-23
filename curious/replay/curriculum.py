from typing import List, Self, Tuple, Dict, Any

import torch
from dataclasses import dataclass, fields
from torch.utils.data import Dataset

from textwrap import dedent

@dataclass
class Curriculum:
    """
    A dataclass for storing the curriculum.
    """
    question: List[str]
    """The quesitons provided"""

    answer: List[str]
    """The answers provided"""

    oracle_answer: List[str]
    """The oracle answers"""

    completion: List[str]
    """The completion of the model"""
    
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
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "oracle_answer": self.oracle_answer,
            "completion": self.completion,
            "learnability": self.learnability.tolist(),
        }
    

    def __repr__(self) -> str:
        return dedent(
        f"""
        Curriculum(
        Question:
        {self.question}
        -------------------------
        Answer:
        {self.answer}
        -------------------------
        Oracle_answer:
        {self.oracle_answer}
        -------------------------
        Completion:
        {self.completion}
        -------------------------
        Learnability:
        {self.learnability}
        )
        """
        ).strip()

class CurriculumBuffer:
    """
    A buffer for storing the curriculum.
    """ 
    def __init__(self, buffer: List[Curriculum] = None) -> None:
        self.buffer = buffer if buffer is not None else []
        self.keys = [
            "question",
            "answer",
            "oracle_answer",
            "learnability",
            "completion",
        ]

    def append(self, curriculum: Curriculum) -> None:
        """
        Append a curriculum to the buffer.
        """
        num_samples = len(curriculum.question)
        individual_curriculums:List[Curriculum] = [{} for _ in range(num_samples)]

        for key in self.keys:
            if key not in curriculum.__dict__:
                raise ValueError(f"Curriculum must have a {key} attribute")
            
            values = getattr(curriculum, key)
            if isinstance(values, torch.Tensor):
                values: Tuple[torch.Tensor, ...] = torch.unbind(values)
            
            for i, v in enumerate(values):
                individual_curriculums[i][key] = v

        self.buffer.extend(
            [
                Curriculum(**data) for data \
                in individual_curriculums
            ]
        )

    def clear(self) -> None:
        self.buffer.clear()

    def __len__(self) -> int:
        return len(self.buffer) 
    
    def __getitem__(self, idx: int) -> Curriculum:
        return self.buffer[idx]
    