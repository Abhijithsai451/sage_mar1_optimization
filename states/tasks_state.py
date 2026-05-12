from pydantic import BaseModel

class TasksState(BaseModel):
    question: str
    reward_challenger: float


