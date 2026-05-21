import uuid

from pydantic import BaseModel, Field


class RewardState(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    reward_challenger: float = 0.0
    reward_planner: float = 0.0
    reward_solver: float = 0.0
    reward_format: float = 0.0
    reward_diff: float = 0.0
