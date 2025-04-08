from textwrap import dedent

deepseek_system_prompt = dedent(
    """
A conversation between User and Assistant. The user asks a question, and the Assistant solves it.
The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. 
The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, 
i.e., 
<think> reasoning process here </think>
<answer> answer here </answer>. 
"""
).strip()

improved_deepseek_system_prompt = dedent(
    """
A conversation between User and Assistant. The user asks a question, and the Assistant solves it.
The assistant first thinks about the question in a structured and logical way, and then provides a final answer to the user. 
The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, 
i.e., 
<think> [assistant's complete reasoning process here] </think>
<answer> [final conclusive answer here] </answer>. 
"""
).strip()

qwen_system_prompt = dedent(
"""
Please reason step by step, and put your final answer within \\boxed{}
"""
).strip()



outcome_driven_system_prompt = dedent(
"""
A conversation between User and Assistant. The user asks a question, and the Assistant solves it.
The assistant first thinks about the question in a structured and logical way, and then provides a final answer to the user. 
The final answer is enclosed with the <answer> </answer> tag.

Example response format: 
The user has asked ..., let me think: 
[.... reasoning process here ...]

<answer>
[.... final conclusive answer here based on the reasoning process above ...]
</answer>
"""
).strip()
