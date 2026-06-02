import uuid
from pydantic import BaseModel, Field

class ScoreState(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    score_quality: float = 0.0
    score_planner: float = 0.0
    score_ground_truth: float = 0.0
