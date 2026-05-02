from typing_extensions import List, TypedDict, Literal
from langchain_core.messages import BaseMessage

from states.critic_state import CriticState


class SAGEAgentState(TypedDict):
    messages: List[BaseMessage]
    input: str
    critic_state: CriticState
    tasks : List[str]
    plan: str

    status: Literal["challenged","planned","solved"]










