from typing import List
from transformers import PreTrainedTokenizer

class Injector:

    def __init__(self, tokenizer: PreTrainedTokenizer, intervention_string: str) -> None:
        self.tokenizer = tokenizer
        self.intervention_string:str = intervention_string
        self.intervention_token:List[int] = tokenizer.encode(intervention_string)

    def inject(self, text: str) -> str:
        return text.replace(self.intervention_token, self.tokenizer.eos_token)
    
    