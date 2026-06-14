import torch
from datasets import Dataset
from torch.utils.data import DataLoader

from curious.data.gsm8k import GSM8KDataset
from curious.utils.utils import tensorize_model_inputs


def test_tensorize_model_inputs_preserves_text_columns():
    row = {
        "question": "What is 1 + 1?",
        "oracle_answer": "2",
        "input_ids": [101, 102],
        "attention_mask": [1, 1],
    }

    formatted = tensorize_model_inputs(row)

    assert formatted["question"] == "What is 1 + 1?"
    assert formatted["oracle_answer"] == "2"
    assert torch.equal(formatted["input_ids"], torch.tensor([101, 102]))
    assert formatted["input_ids"].dtype == torch.long
    assert torch.equal(formatted["attention_mask"], torch.tensor([1, 1]))
    assert formatted["attention_mask"].dtype == torch.long


def test_gsm8k_dataset_getitem_tensorizes_raw_hf_rows():
    dataset = GSM8KDataset.__new__(GSM8KDataset)
    dataset.mode = "test"
    dataset.train = Dataset.from_list([])
    dataset.test = Dataset.from_list(
        [
            {
                "question": "What is 2 + 2?",
                "oracle_answer": "4",
                "input_ids": [201, 202],
                "attention_mask": [1, 1],
            }
        ]
    )

    item = dataset[0]

    assert item["question"] == "What is 2 + 2?"
    assert torch.equal(item["input_ids"], torch.tensor([201, 202]))
    assert torch.equal(item["attention_mask"], torch.tensor([1, 1]))


def test_hf_dataset_transform_collates_model_inputs_as_tensors():
    hf_dataset = Dataset.from_list(
        [
            {
                "question": "q1",
                "oracle_answer": "1",
                "input_ids": [1, 2],
                "attention_mask": [1, 1],
            },
            {
                "question": "q2",
                "oracle_answer": "2",
                "input_ids": [3, 4],
                "attention_mask": [1, 1],
            },
        ]
    )
    hf_dataset.set_transform(tensorize_model_inputs)

    batch = next(iter(DataLoader(hf_dataset, batch_size=2, num_workers=0)))

    assert batch["question"] == ["q1", "q2"]
    assert torch.equal(batch["input_ids"], torch.tensor([[1, 2], [3, 4]]))
    assert torch.equal(batch["attention_mask"], torch.tensor([[1, 1], [1, 1]]))
