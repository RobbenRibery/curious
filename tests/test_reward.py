from curious.reward import GSM8KRewardModel, SOLVED_REWARD, NEGATIVE_REWARD, ZERO_REWARD
from curious.data import GSM8KDataset
from curious.reward import QWEN_ANSWER_PATTERN
from curious.prompt import qwen_system_prompt
from textwrap import dedent

def test_outcome_reward():
    reward_model = GSM8KRewardModel(use_format_reward=False)
    pred1 = """
    <think> 
    If x + y = 10, x - y = 2, what is x?
    </think>

    <answer>
    final answer: x = 6
    </answer>
    """
    oracle_answer = "6"

    pred_answer, reward, info = reward_model.outcome_reward(pred1, oracle_answer)
    assert reward == SOLVED_REWARD
    assert info["outcome"] == None

    pred2 = """
    <think> 
    If x + y = 10, x - y = 2, what is x?
    </think>

    <answer>
    x = 5
    </answer>
    """
    pred_answer, reward, info = reward_model.outcome_reward(pred2, oracle_answer)
    assert reward == ZERO_REWARD
    assert info["outcome"] == "wrong_answer"

    pred3 = """
    <think> 
    If x + y = 10, x - y = 2, what is x?
    </think>

    answer:x = 6
    """
    pred_answer, reward, info = reward_model.outcome_reward(pred3, oracle_answer)
    assert reward == NEGATIVE_REWARD
    assert info["outcome"] == "no_answer_in_required_format"

def test_format_reward():
    pred1 = dedent(
    """
    <think> 
    If f(x) = x^2, what is f(x) for x = 10?
    <answer> 
    final answer: 1.0e02
    </answer>
    </think>
    
    <answer>
    1.0e02
    </answer>
    """
    ).strip()

    reward_model = GSM8KRewardModel(use_format_reward=True, use_strict_format_reward=False)
    _, reward, info = reward_model.format_reward(pred1)
    assert reward == SOLVED_REWARD
    assert info["format_"] == None

    reward_model = GSM8KRewardModel(use_format_reward=True, use_strict_format_reward=True)
    _, reward, info = reward_model.format_reward(pred1)
    assert reward == NEGATIVE_REWARD
    assert info["format_"] == "wrong_format"

    pred2 = dedent(
    """
    <think>If f(x) = x^2, what is f(x) for x = 10?</think>
    <answer>1.0e02</answer>

    <think>let's multiply the answer by 2</think>
    <answer>2.0e02</answer> 
    """
    ).strip()

    oracle_answer = "200"
    reward_model = GSM8KRewardModel(use_format_reward=True, use_strict_format_reward=True)
    parsed_answer, reward, info = reward_model.format_reward(pred2)
    assert reward == SOLVED_REWARD
    assert info["format_"] == None


    pred2 = dedent(
    """
    <think>If f(x) = x^2, what is f(x) for x = 10?</think>
    <answer>1.0e02</answer>
    <think>let's multiply the answer by 2</think>
    <answer><think> let's think about it</think>2.0e02</answer> 
    """
    ).strip()

    oracle_answer = "200"
    reward_model = GSM8KRewardModel(use_format_reward=True, use_strict_format_reward=True)
    parsed_answer, reward, info = reward_model.format_reward(pred2)
    assert reward == NEGATIVE_REWARD
    assert info["format_"] == "wrong_ending_format"

def test_instance_reward():
    pred1 = dedent(
    """
    <think>If f(x) = x^2, what is f(x) for x = 10?</think>
    <answer>100</answer>
    <think>let's multiply the answer by 2</think>
    <answer>the final answer is 200 = 100 * 2</answer> 
    """ 
    ).strip()

    oracle_answer = "200"
    reward_model = GSM8KRewardModel(use_format_reward=True, use_strict_format_reward=True)
    reward, info = reward_model.instance_reward(pred1, oracle_answer)
    assert info['parsed_answer'][1] == "200"
    assert reward == 2*SOLVED_REWARD
    assert info['parsed_reasoning'] == pred1
    assert info["format_"] == None
    assert info["outcome"] == None


    pred1 = dedent(
    """
    <think>If f(x) = x^2, what is f(x) for x = 10?</think>
    <answer>1.0e02</answer>
    <think>let's multiply the answer by 2</think>
    <answer>the final answer is 200 = 2.0e02</answer> 
    <answer>the final answer is 400 = 4.0e02</answer> 
    """ 
    ).strip()

    oracle_answer = "200"
    reward_model = GSM8KRewardModel(use_format_reward=True, use_strict_format_reward=True)
    reward, info = reward_model.instance_reward(pred1, oracle_answer)
    assert reward == NEGATIVE_REWARD
    assert info['parsed_answer'][1] == "400"
    assert info["format_"] == "wrong_ending_format"
    assert info["outcome"] == "wrong_answer"

def test_batch_reward():
    gsm8k = GSM8KDataset(mode="train")
    completions = gsm8k[:10]["answer"]
    completions = [
        completion.replace("####", "<answer>").strip() + "</answer>" \
        for completion in completions
    ] 
    oracle_answers = gsm8k[:10]["oracle_answer"]

    reward_model = GSM8KRewardModel(use_format_reward=False, use_strict_format_reward=False)
    rewards, infos, solved_rate = reward_model(completions, oracle_answers)
    assert len(rewards) == len(completions)
    assert len(infos) == len(completions)
    assert rewards == [SOLVED_REWARD] * len(completions)
    assert solved_rate == 1.0

def test_qwen_reward():
    reward_model = GSM8KRewardModel(use_format_reward=False, use_strict_format_reward=False, answer_pattern=QWEN_ANSWER_PATTERN)
    
    completion = dedent(
    """
    In summary:
    - Her total earnings were $960;
    - She had $180 over her goal ($610 minus $790);
    - The difference was $180; and
    - Thus, she made an additional $180 above her goal, as calculated previously.
    The final answer is $\\boxed{180}$ dollars.
    """
    ).strip()
    #print(qwen_system_prompt)

    oracle_answer = "180"
    answer_parsed, reward, info = reward_model.outcome_reward(completion, oracle_answer)
    assert reward == SOLVED_REWARD
    assert info["outcome"] == None