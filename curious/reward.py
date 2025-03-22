import re
from typing import List, Tuple

SOLVED_REWARD = 1.0

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
        oracle_answer = oracle_answer.lower()
        answer_matchs = re.findall(
            r"<answer>(.*?)</answer>",
            completion,
            flags=re.DOTALL,
        )
        # TODO: 
        # must only consider the final reward 
        # TODO: 
        # why? the LLM could produce intermediate answers 
        # TODO: 
        # ADDITIONAL PARSING IS REQUIRED TO HANDLE 000,000 patterns 
        # we need to use some symbolic methods to validate i.e. Sympy

        answer = answer_matchs[-1] if answer_matchs else None
        reward = 0
        if answer is not None:
            if answer == oracle_answer:
                reward = SOLVED_REWARD
            elif oracle_answer in answer:
                reward = 0.5
            else:
                reward = 0.1

        return reward

    @classmethod
    def process_reward(cls, completion: str, illeagel_contents:List[str]) -> tuple[float, int|None, int|None]:
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
        think = think_match.group(1).lower().strip() if think_match else None
        think_start = think_match.start() if think_match else None
        think_end = think_match.end() if think_match else None

        reward = 0 
        # neg_reward = -0.1
        # if think is not None:
        #     for illeagel_content in illeagel_contents:
        #         illeagel_content = illeagel_content.lower().strip()
        #         if illeagel_content == think:
        #             return neg_reward

        if think is not None:
            reward = (think_end - think_start) / len(completion) * 0.1
        
        return reward
        
    
    @classmethod
    def reward_batch(
        cls, 
        completions: List[str], 
        oracle_answers: List[str],
        illeagel_contents:List[str],
    ) -> Tuple[List[float], float]:
        """
        Computes the outcome reward for a batch of completions and oracle answers.
        """
        rewards = []
        sovled_times = 0
        for completion, oracle_answer in zip(completions, oracle_answers):
            outcome_reward = cls.outcome_reward(completion, oracle_answer)
            sovled_times += 1 if outcome_reward == SOLVED_REWARD  else 0
            # TODO: Test only using the outcome reward 
            process_reward = cls.process_reward(completion, illeagel_contents=illeagel_contents)
            reward = outcome_reward + process_reward
            rewards.append(reward)

        return rewards, sovled_times/len(completions)