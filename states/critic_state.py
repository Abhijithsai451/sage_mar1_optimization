from pydantic import BaseModel

class CriticState(BaseModel):
    # Challenger
    reward_challenger: float
    alpha: float = 0.7
    reward_diff: float

    # Planner
    beta: float = 0.3
    lambda_plan: float = 0.5
    reward_planner: float

    # Solver
    reward_solver: float
    lambda_format: float
    w_p: float = 0.2
    w_c: float = 0.6
    w_f: float = 0.2

    # Critic
    reward_critic: float
    reward_format: float
    score_quality: float
    score_planner: float
    score_ground_truth: float

