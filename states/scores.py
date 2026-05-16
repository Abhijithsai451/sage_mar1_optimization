from pydantic import BaseModel

class ScoreState(BaseModel):
    score_quality: float
    score_planner: float
    score_ground_truth: float