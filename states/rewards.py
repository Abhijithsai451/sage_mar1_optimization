from pydantic import BaseModel


class RewardState(BaseModel):
    reward_challenger: float = 0.0
    reward_planner: float = 0.0
    reward_solver: float = 0.0
    reward_format: float = 0.0
    reward_diff: float = 0.0
