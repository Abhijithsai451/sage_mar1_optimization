import operator
from typing import Annotated, List
from langchain_core.messages import BaseMessage


class SAGEAgentState:
    num_tasks: int
    questions: list[str]
    answers: list[str]
    planner_steps: List[str]
    critic_score: float
    history: List[str]






