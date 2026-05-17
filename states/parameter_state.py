from pydantic import BaseModel


class ParameterState(BaseModel):
    # Challenger
    alpha: float = 0.7

    # Planner
    beta: float = 0.3
    lambda_plan: float = 0.5

    # Solver
    lambda_format: float = 0.0
    w_p: float = 0.2
    w_c: float = 0.6
    w_f: float = 0.2
