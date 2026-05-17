from pydantic import BaseModel, Field

from states.rewards import RewardState
from states.scores import ScoreState


class TasksState(BaseModel):
    question: str
    rewards: RewardState = Field(default_factory=RewardState)
    score: ScoreState = Field(default_factory=ScoreState)
    plan: str = ""
    solution: str = ""
