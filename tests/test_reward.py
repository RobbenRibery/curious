from curious.reward import GSM8KRewardModel, SOLVED_REWARD, NEGATIVE_REWARD, PARTIAL_REWARD
from textwrap import dedent 

THINK_PATTERN = r"<think>(.*?)</think>"
ANSWER_PATTERN = r"<answer>(.*?)</answer>"


def test_perfect_answer():
    perfect_completion = dedent(
    """
    <think> 
    because the number is too big, we need to use scientific notation
    </think>

    <answer>
    5e8
    </answer>
    """
    ).strip()
    
    reward_model = GSM8KRewardModel(think_pattern=THINK_PATTERN, answer_pattern=ANSWER_PATTERN)

    oracle_answer = "5e8"
    parsed_answer, reward, _ = reward_model(perfect_completion, oracle_answer)
    assert reward == [SOLVED_REWARD]
    assert parsed_answer == ["5e8"]

    oracle_answer = "500,000,000"
    parsed_answer, reward, _ = reward_model(perfect_completion, oracle_answer)
    assert reward == [SOLVED_REWARD]
    assert parsed_answer == ["5e8"]

    oracle_answer = "50"
    parsed_answer, reward, _ = reward_model(perfect_completion, oracle_answer)
    assert reward == [NEGATIVE_REWARD]
    assert parsed_answer == ["5e8"]


def test_no_answer():
    completion = dedent(
    """
    <think> 
    because the number is too big, we need to use scientific notation
    </think>
    """
    ).strip()

    oracle_answer = "123"
    reward_model = GSM8KRewardModel(think_pattern=THINK_PATTERN, answer_pattern=ANSWER_PATTERN)
    parsed_answer, reward, _ = reward_model(completion, oracle_answer)
    assert reward == [NEGATIVE_REWARD]
    assert parsed_answer == [None]


    completion = dedent(
    """
    <think> 
    because the number is too big, we need to use scientific notation
    </think>

    <answer> 

    </answer>
    """
    ).strip()

    oracle_answer = "567"
    parsed_answer, reward, _ = reward_model(completion, oracle_answer)
    assert reward == [NEGATIVE_REWARD]
    assert parsed_answer == [None]


    completion = dedent(
    """
    <think> 
    because the number is too big, we need to use scientific notation
    </think>

    <answer> 567 </answer>

    <answer> 
    </answer>
    """
    ).strip()

    oracle_answer = "567"
    parsed_answer, reward, _ = reward_model(completion, oracle_answer)
    assert reward == [NEGATIVE_REWARD]
    assert parsed_answer == [None]

def test_partial_answer():
    completion = dedent(
    """
    <think> 
    what is 3*40
    </think>

    <answer> 
    120,00
    </answer>
    """
    ).strip()
    
    oracle_answer = "120"
    reward_model = GSM8KRewardModel(think_pattern=THINK_PATTERN, answer_pattern=ANSWER_PATTERN)
    parsed_answer, reward, _ = reward_model(completion, oracle_answer)
    assert reward == [PARTIAL_REWARD]
    assert parsed_answer == ["12000"]

    completion = dedent(
    """
    <think> 
    what is 3*40
    </think>

    <answer> 
    120.00
    </answer>
    """
    ).strip()
    oracle_answer = "120"
    reward_model = GSM8KRewardModel(think_pattern=THINK_PATTERN, answer_pattern=ANSWER_PATTERN)
    parsed_answer, reward, _ = reward_model(completion, oracle_answer)
    assert reward == [SOLVED_REWARD]
    assert parsed_answer == ["120.00"]
    
    
    