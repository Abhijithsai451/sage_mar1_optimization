from pydantic import BaseModel


class ScoreState(BaseModel):
    score_quality: str
    score_planner: str
    score_ground_truth: str
