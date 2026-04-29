import operator
from typing import Annotated, List, TypedDict
from langchain_core.messages import BaseMessage


class SAGEAgentState(TypedDict):
    messages: List[BaseMessage]
    #Challenger
    reward_challenger: float
    alpha: float
    reward_diff: float

    #Planner
    beta: float
    lambda_plan: float
    reward_planner : float

    #Solver
    reward_solver: float
    lambda_format: float
    w_p : float
    w_c : float
    w_f : float

    #Critic
    reward_critic: float
    reward_format: float
    score_quality: float
    score_planner : float
    score_ground_truth : float

    status: str










