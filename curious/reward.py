import re
from typing import List, Tuple, Optional, Dict
from concurrent.futures import ThreadPoolExecutor, Future

import sympy 
from math_verify import verify, parse
from dataclasses import dataclass

ZERO_REWARD = 0.0
NEGATIVE_REWARD = -1.0 
SOLVED_REWARD = 1.0

THINK_PATTERN = r"<think>(.*?)</think>"
ANSWER_PATTERN = r"<answer>(.*?)</answer>"
BASE_FORMAT_PATTERN = r"^<think>.*?</think>\s*<answer>.*?</answer>$"
STRICT_FORMAT_PATTERN = (
    r"(<think>(?:(?!<answer>|</answer>).)*?</think>\s*"
    r"<answer>(?:(?!<think>|</think>).)*?</answer>)"
)


def normalize_number(answer:str) -> str:
    """
    Normalizes the answer by removing commas and converting to lowercase.
    """
    return answer.lower().strip().replace(",", "")

@dataclass
class FailureMode: 
    
    # answer failure mode
    NO_ANSWER = "no_answer_in_required_format"
    NO_NUMBER_IN_ANSWER = "no_number_in_answer"
    WRONG_ANSWER = "wrong_answer"

    # format failure mode
    WRONG_FORMAT = "wrong_format"
    WRONG_BEGINNING_FORMAT = "wrong_beginning_format"
    WRONG_ENDING_FORMAT = "wrong_ending_format"
    MULTIPLE_FORMATS = "multiple_required_formats"


class GSM8KRewardModel:

    def __init__(
        self, 
        answer_pattern:str = ANSWER_PATTERN, 
        think_pattern:str = THINK_PATTERN,
        format_pattern:str = BASE_FORMAT_PATTERN,
        use_format_reward:bool = False,
        use_strict_format_reward:bool = False,
    ) -> None: 
        """
        Initialize the reward model.

        Args:
            answer_pattern (str): The pattern to extract the answer from the completion.
            think_pattern (str): The pattern to extract the think from the completion.
            format_pattern (str): The pattern to check the format of the completion.
            use_format_reward (bool): Whether to use the format reward.
            use_strict_format_reward (bool): Whether to use the strict format reward.
        """
        # initialize the patterns
        self.answer_pattern = answer_pattern
        self.think_ptttern = think_pattern
        self.format_pattern = format_pattern

        self.use_strict_format_reward = use_strict_format_reward
        if use_strict_format_reward:
            self.format_pattern = STRICT_FORMAT_PATTERN

        self.use_format_reward = use_format_reward
        
    def outcome_reward(self, completion: str, oracle_answer: str) -> Tuple[Optional[List[sympy.Expr | str]], float, Dict[str, str]]:
        """
        Computes the reward for a given completion and oracle answer.

        Args:
            completion (str): The string containing the completion with an embedded answer.
            oracle_answer (str): The correct answer to compare against.

        Returns:
            Tuple[Optional[List[sympy.Expr | str]], float, Dict[str, str]]: 
            A tuple containing the parsed answer, the reward value and a dictionary of failure mode.
        """
        # normalize the oracle answer 
        oracle_answer:List[sympy.Expr | str] = parse(oracle_answer,)

        # find all the answer matches
        answer_match:List[str] | None = re.findall(self.answer_pattern, completion, flags=re.DOTALL)

        # return negative reward in case no answer is found
        if not answer_match:
            return None, NEGATIVE_REWARD, {"outcome": FailureMode.NO_ANSWER}

        # get the last answer as the final answer and normalize the parsed answer
        answer_parsed:List[sympy.Expr | str] = parse(answer_match[-1], extraction_mode="any_match")
        if not answer_parsed:
            return None, NEGATIVE_REWARD, {"outcome": FailureMode.NO_NUMBER_IN_ANSWER}

        # return positive reward in case 
        # the answer is exactly the same as the oracle answer
        if verify(answer_parsed, oracle_answer):
            return answer_parsed, SOLVED_REWARD, {"outcome": None}
        else:
            return answer_parsed, ZERO_REWARD, {"outcome": FailureMode.WRONG_ANSWER}

    def format_reward(self, completion:str) -> Tuple[str|None, float, Dict[str, str|None]]:
        """
        Computes the reward for the format of the completion.

        Args:
            completion (str): The string containing the completion with an embedded answer.

        Returns:
            Tuple[str|None, float, Dict[str, str|None]]: A tuple containing the parsed format, the reward value and a dictionary of failure mode.
        """
        section_parsed = None 
        if not self.use_format_reward:
            return section_parsed, ZERO_REWARD, {"format_": None}

        # find all the format matches
        format_matches = list(re.finditer(self.format_pattern, completion, flags=re.DOTALL))

        # return negative reward in case no format is found
        if not format_matches:
            return section_parsed, NEGATIVE_REWARD, {"format_": FailureMode.WRONG_FORMAT}
        
        if format_matches[0].start() != 0:
            return section_parsed, NEGATIVE_REWARD, {"format_": FailureMode.WRONG_BEGINNING_FORMAT}
        
        if format_matches[-1].end() != len(completion):
            return section_parsed, NEGATIVE_REWARD, {"format_": FailureMode.WRONG_ENDING_FORMAT}

        # return the last format as the final format
        section_parsed = "\n".join([mathch_.group(0) for mathch_ in format_matches]) if len(format_matches) > 1 else format_matches[0].group(0)
        return section_parsed, SOLVED_REWARD, {"format_": None}
             
    def instance_reward(self, completion: str, oracle_answer: str) -> Tuple[float, Dict[str, str]]:
        """
        Computes the reward for a given completion and oracle answer.

        Args:
            completion (str): The string containing the completion with an embedded answer.
            oracle_answer (str): The correct answer to compare against.

        Returns:
            Tuple[float, Dict[str, str]]: A tuple containing the reward value and a dictionary of failure mode.
        """
        info  = {}

        parsed_answer, outcome_reward, outcome_failure_mode = self.outcome_reward(completion, oracle_answer)
        info.update(outcome_failure_mode)
        info.update({"parsed_answer": parsed_answer})

        section_parsed, format_reward, format_failure_mode = self.format_reward(completion)
        info.update(format_failure_mode)
        info.update({"parsed_reasoning": section_parsed})

        reward = outcome_reward + format_reward
        return reward, info
    
    def __call__(self, completions: List[str] | str, oracle_answers: List[str] | str) -> Tuple[List[str], List[float], List[Dict[str, str]], float]:
        """
        Computes the reward for a given completion and oracle answer.

        Args:
            completions (List[str] | str): The list of completions or a single completion.
            oracle_answers (List[str] | str): The list of oracle answers or a single oracle answer.

        Returns:
            Tuple[List[str], List[float], List[Dict[str, str]], float]: A tuple containing the parsed answers, rewards, infos and the solved rate.
        """
        if isinstance(completions, str):
            completions = [completions]
        if isinstance(oracle_answers, str):
            oracle_answers = [oracle_answers]

        rewards, infos = [], []
        sovled_times = 0
        for completion, oracle_answer in zip(completions, oracle_answers):
            reward, info = self.instance_reward(completion, oracle_answer)
            rewards.append(reward)
            infos.append(info)

            # update the solved times
            if info["outcome"] is None and info["format_"] is None:
                sovled_times += 1

        # return the rewards, infos and the solved rate
        return rewards, infos, sovled_times/len(completions)
