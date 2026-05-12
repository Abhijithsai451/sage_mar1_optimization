from pydantic import BaseModel
from typing_extensions import List, TypedDict, Literal
from langchain_core.messages import BaseMessage

from states.critic_state import CriticState
from states.tasks_state import TasksState


class SAGEAgentState(TypedDict):
    messages: List[BaseMessage]
    input: str
    critic_state: CriticState
    tasks : TasksState
    plan: str

    status: Literal["challenged","planned","solved"]










