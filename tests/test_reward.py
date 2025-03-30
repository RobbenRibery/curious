from curious.reward import GSM8KRewardModel, SOLVED_REWARD, NEGATIVE_REWARD, ZERO_REWARD
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
    assert info["format_"] == "wrong_format"

    
    
    