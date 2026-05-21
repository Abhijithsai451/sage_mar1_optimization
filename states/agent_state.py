import uuid

from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field
from typing_extensions import List, Literal

from states.parameter_state import ParameterState
from states.tasks_state import TasksState


class SAGEAgentState(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    messages: List[BaseMessage]
    input: List[str]
    parameter_state: ParameterState = Field(default_factory=ParameterState)
    tasks: List[TasksState] = Field(default_factory=list)
    status: Literal["Initialized", "challenged", "planned", "solved"]
