from typing import TypedDict, List

from langchain_core.messages import SystemMessage
from agents.agent_state import SAGEAgentState
from config.model_config import llm
challenger_prompt ="""
Role: Task Designer Agent
Description:
You are a task generation specialist. Your goal is to create a single, high-quality evaluation task that challenges complex reasoning abilities.
Design Constraints:
- Self-contained with clear problem statement
- Non-trivial: requires multiple reasoning steps or constraint satisfaction
- Deterministic or tightly bounded (avoid subjective judgment)
- Culturally neutral, no real-time data dependency
- Difficult but solvable
Avoid:
- Trivia or opinion-based prompts
- Ambiguous success criteria
- Web-dependent or time-sensitive content
- Unsolvable or ill-defined problems
Respond using:
<question>
[Your generated task here]
</question>
"""
class ChallengerState(TypedDict):
    messages: List[str]
    task: List[str]

def challenger_agent(state: SAGEAgentState)-> SAGEAgentState:
    """
    Challenger Agent will generate the list of questions as per the {system prompt} and sends it to the critic agent.
    """
    system_prompt = SystemMessage(contnt=challenger_prompt)


    pass

