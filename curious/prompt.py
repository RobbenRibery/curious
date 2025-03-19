from textwrap import dedent

system_prompt = dedent(
"""
You are a helpful assistant.

Respond in the following format: 
Start with a reasoning process and then quote the final answer inside the <answer> tag.

<reasoning>
... 
</reasoning> 

<answer>Final answer</answer>

"""
).strip()