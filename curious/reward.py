import re
from concurrent.futures import ThreadPoolExecutor, Future

class RewardModel:

    @classmethod
    def outcome_reward(cls, completion: str, oracle_answer: str) -> float:
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
        answer_match = re.search(
            r"<answer>(.*?)</answer>",
            completion,
            flags=re.DOTALL,
        )

        answer = answer_match.group(1) if answer_match else None
        reward = 0
        if answer is not None:
            if answer == oracle_answer:
                reward = 1.0
            elif oracle_answer in answer:
                reward = 0.5
            else:
                reward = 0.01
        return reward, 

    @classmethod
    def process_reward(cls, completion: str) -> tuple[float, int|None, int|None]:
        """
        Processes the reward for a given completion.

        Args:
            completion (str): The string containing the completion with an embedded think and answer.

        Returns:
            tuple[float, int|None, int|None]: A tuple containing the reward value, 
            the start index of the think, and the end index of the think.
        """
        think_match = re.search(
            r"<think>(.*?)</think>",
            completion,
            flags=re.DOTALL,
        )
        think = think_match.group(1) if think_match else None
        think_start = think_match.start() if think_match else None
        think_end = think_match.end() if think_match else None

        reward = 0 
        if think is not None:
            reward = 0.5 
        
        return reward, think_start, think_end
        
    
    