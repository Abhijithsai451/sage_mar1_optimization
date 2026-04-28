import operator
from typing import Annotated, List
from langchain_core.messages import BaseMessage


class SAGEAgentState:

    #Challenger
    challenger_reward: int
    threshold: int = 0.7
    reward_diff: int

    #Planner
    reg_beta: int = 0.3
    weight_coeff: int = 0.5
    score_planner: int
    reward_planner : int









