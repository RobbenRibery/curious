import os 
os.environ["TOKENIZERS_PARALLELIS"] = "true"
import torch 
from torch.utils.data import DataLoader 
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from curious.data import ReasoningGymDataset
from curious.utils import tokenize_questions
from curious.grpo import rollout


EACH_DATASET_SIZE = 100
SEED = 42
BATCH_SIZE = 2
MODEL_NAME = "Qwen/Qwen2-0.5B-Instruct"

datasets = [
    "complex_arithmetic",
    "intermediate_integration",
    "polynomial_equations",
    "simple_equations",
]

dataset = ReasoningGymDataset(
    datasets_name=datasets,
    size=EACH_DATASET_SIZE,
    seed=SEED
)

print(len(dataset))

data_loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=0
)
print(len(data_loader))    


for batch in data_loader:
    print(len(batch["question"]))
    print(batch["dataset_name"])
    print('*'*100)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, 
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
    #attn_implementation="flash_attention_2",
)


samples = next(iter(data_loader))
encodings = tokenize_questions(
    tokenizer,
    questions= samples["question"],
    max_length= 1024,
)

generation_config = GenerationConfig(
    max_new_tokens=512,
    do_sample=True,
    top_p=0.9,
    top_k=50,
    temperature=0.7,
    num_return_sequences=2,
)

encodings["input_ids"] = encodings["input_ids"].to(model.device)
encodings["attention_mask"] = encodings["attention_mask"].to(model.device)

seq_ids, returns, solved_rate, action_mask, completions = rollout(
    model,
    tokenizer,
    batch_inputs=encodings,
    oracle_answers=samples["answer"],
    generation_config=generation_config,
)

print(returns.shape)
print(returns)


