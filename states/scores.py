import uuid
from pydantic import BaseModel, Field

class ScoreState(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    score_quality: str
    score_planner: str
    score_ground_truth: str
