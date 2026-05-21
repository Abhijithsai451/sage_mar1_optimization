import uuid

from pydantic import BaseModel, Field

from states.rewards import RewardState
from states.scores import ScoreState


class TasksState(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str
    rewards: RewardState = Field(default_factory=RewardState)
    score: ScoreState = Field(default_factory=ScoreState)
    plan: str = ""
    solution: str = ""
