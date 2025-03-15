from transformers import PreTrainedTokenizer
from typing import List, Dict

from curious.prompt import system_prompt

def tokenize_questions(
    tokenizer: PreTrainedTokenizer, 
    questions: List[str],
    max_length: int = None
) -> List[Dict]:
    """
    Tokenize a list of questions and answers.
    """
    if not max_length:
        print(f"Using model max length: {tokenizer.model_max_length}")
        max_length = tokenizer.model_max_length

    if tokenizer.padding_side == "right":
        print(f"Adjusting padding side from right to left for training")
        # training padding should be on the left 
        tokenizer.padding_side = "left"

    # add the DeepSeek system prompt to the conversation
    questions = [
        [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": question
            }
        ] for question in questions
    ]

    # Apply the chat template to the questions 
    texts = tokenizer.apply_chat_template(
        conversation= questions,
        tokenize= False,
        add_generation_prompt= True,
    )

    # get encodings for the question
    encodings = tokenizer(
        texts,
        padding= "max_length",
        truncation= True,
        return_tensors= "pt",
        max_length= max_length,
    )

    return encodings
        