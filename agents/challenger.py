from typing import TypedDict, List

from langchain_core.messages import SystemMessage
from agents.agent_state import SAGEAgentState
from config.model_config import get_backbone

challenger_policy ="""
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

def challenger_agent(state: SAGEAgentState)-> SAGEAgentState:


    state['alpha'] = 0.7

    if state['score_quality']>=state['alpha']:
        state['reward_challenger'] = (state['score_quality'] + state['reward_diff'] + state['reward_format'])/3
    else:
        state['reward_challenger'] = (state['score_quality'] + state['reward_format'])/ 2

    response = get_backbone().invoke(prompt = state['input'], system_prompt = challenger_policy)
    state['messages'] = response.content
    return state
