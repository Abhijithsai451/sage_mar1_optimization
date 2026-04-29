import operator
from typing import Annotated, List, TypedDict
from langchain_core.messages import BaseMessage


class SAGEAgentState(TypedDict):

    #Challenger
    reward_challenger: int
    alpha: int
    reward_diff: int

    #Planner
    beta: int
    lambda_plan: int
    reward_planner : int

    #Solver
    reward_solver: int
    lambda_format: int
    w_p : int
    w_c : int
    w_f : int

    #Critic
    reward_critic: int
    reward_format: int
    score_quality: int
    score_planner : int










