import re
from typing import List, Tuple, Optional

from concurrent.futures import ThreadPoolExecutor, Future

PARTIAL_REWARD = 0.5
NEGATIVE_REWARD = -1.0 
SOLVED_REWARD = 1.0

def normalize_number(answer:str) -> str:
    """
    Normalizes the answer by removing commas and converting to lowercase.
    """
    return answer.lower().strip().replace(",", "")

class GSM8KRewardModel:

    def __init__(
        self, 
        answer_pattern:str, 
        think_pattern:str,
        use_format_reward:bool = False,
    ) -> None: 
        """
        Initializes the RuleRewardModel with the given patterns and options.

        Args:
            answer_pattern (str): The regex pattern for extracting the answer.
            think_pattern (str): The regex pattern for extracting the think.
            use_process_reward (bool): Whether to use the process reward.
        """
        self.answer_pattern = answer_pattern
        self.think_ptttern = think_pattern

        if use_format_reward:
            raise NotImplementedError("Format reward is not implemented yet")
        self.use_format_reward = use_format_reward

        self.executor = ThreadPoolExecutor()
        
    def outcome_reward(self, completion: str, oracle_answer: str) ->Tuple[Optional[str], float]:
        """
        Computes the reward based on the completion's answer compared to the oracle answer.

        Args:
            completion (str): The string containing the completion with an embedded answer.
            oracle_answer (str): The correct answer to compare against.

        Returns:
            float: The reward value calculated based on the match of the answer to the oracle answer.
                A reward of 1.0 is given for an exact match.
                A reward of 0.5 is given if the oracle answer is contained within the answer.
                A reward of 0.01 is given otherwise.
        """
        # normalize the oracle answer 
        oracle_answer = normalize_number(oracle_answer)

        # find all the answer matches
        answer_match:List[str] | None = re.findall(self.answer_pattern, completion, flags=re.DOTALL)

        # return negative reward in case no answer is found
        if not answer_match:
            return None, NEGATIVE_REWARD

        # get the last answer as the final answer
        answer_parsed = normalize_number(answer_match[-1])
        if not answer_parsed:
            return None, NEGATIVE_REWARD

        # return positive reward in case 
        # the answer is exactly the same as the oracle answer
        if eval(answer_parsed) == eval(oracle_answer):
            return answer_parsed, SOLVED_REWARD
        elif oracle_answer in answer_parsed:
            return answer_parsed, PARTIAL_REWARD
        else:
            return answer_parsed, NEGATIVE_REWARD

    def format_reward(self, completion: str) -> Tuple[int|None, int|None, float]:
        """
        # TODO: refine this function later
        Processes the reward for a given completion.

        Args:
            completion (str): The string containing the completion with an embedded think and answer.

        Returns:
            tuple[int|None, int|None, float]: A tuple containing the reward value, 
            the start index of the think, and the end index of the think.
        """
        # find the <think> section
        think_match = re.search(self.think_ptttern, completion, flags=re.DOTALL)
        if not think_match:
            return None, None, NEGATIVE_REWARD
        
        think_start, think_end = think_match.span()

        # <answer> should not appear within the <think> section.
        if "<answer>" in think_match.group(1) or "</answer>" in think_match.group(1):
            return None, None, NEGATIVE_REWARD

        # Find the <answer> section.
        answer_match = re.search(self.answer_pattern, completion, flags=re.DOTALL)
        if not answer_match:
            return think_start, think_end, NEGATIVE_REWARD

        # <answer> must appear after the <think> section.
        if answer_match.start() < think_end:
            return think_start, think_end, NEGATIVE_REWARD

        # <think> should not appear within the <answer> section.
        if "<think>" in answer_match.group(0) or "</think>" in answer_match.group(0):
            return think_start, think_end, NEGATIVE_REWARD

        return think_start, think_end, SOLVED_REWARD
             
    def instance_reward(self, completion: str, oracle_answer: str) -> Tuple[str, float]:
        """
        Computes the reward for a given completion and oracle answer.

        Args:
            completion (str): The string containing the completion with an embedded answer.
            oracle_answer (str): The correct answer to compare against.

        Returns:
            Tuple[str, float]: A tuple containing the parsed answer and the reward value.
        """
        parsed_answer, reward = self.outcome_reward(completion, oracle_answer)

        return parsed_answer, reward
    
    def __call__(
        self,
        completions: List[str] | str,
        oracle_answers: List[str] | str,
    ) -> Tuple[List[str], List[float], float]:
        """
        Computes the outcome reward for a batch of completions and oracle answers.

        Args:
            completions (List[str] | str): The completions to compute the reward for.
            oracle_answers (List[str] | str): The oracle answers to compare against.

        Returns:
            Tuple[List[str], List[float], float]: A tuple containing the parsed answers, the reward values, and the solved rate.
        """
        if isinstance(completions, str):
            completions = [completions]
        if isinstance(oracle_answers, str):
            oracle_answers = [oracle_answers]

        futures:List[Future] = []
        parsed_answers, rewards = [], []
        sovled_times = 0
        for completion, oracle_answer in zip(completions, oracle_answers):
            # submit the task to the executor
            futures.append(
                self.executor.submit(
                    self.instance_reward, 
                    completion, 
                    oracle_answer
                )
            )

        for future_ in futures:
            # get the parsed answer and reward
            parsed_answer, reward = future_.result()
            # update the solved times
            sovled_times += 1 if reward == SOLVED_REWARD  else 0

            # append the parsed answer and reward
            rewards.append(reward)
            parsed_answers.append(parsed_answer)

        return parsed_answers, rewards, sovled_times/len(completions)