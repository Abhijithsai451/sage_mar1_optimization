import os
import re

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
from dotenv import load_dotenv

from config.model_config import get_backbone, BackboneModel
from states.agent_state import SAGEAgentState
from states.rewards import RewardState
from states.scores import ScoreState
from states.tasks_state import TasksState

load_dotenv()
data_path = os.getenv("DATA_DIR")

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
<task>
[Your generated task here]
</task>
"""

def challenger(state: SAGEAgentState, model: BackboneModel)-> SAGEAgentState:
    user_content = f"Dataset Reference Examples: \n {state.input} \n\nPlease generate a set of 5 new tasks following the reference style."
    messages = [
        SystemMessage(content = challenger_policy),
        HumanMessage(content = user_content)
    ]
    response = model.invoke(messages)
    content = response.content
    content = re.findall(r"<task>(.*?)</task>", content, re.DOTALL)
    tasks = []
    for task in content:
        task_cleaned = task.strip()
        if task_cleaned:
            tasks.append(TasksState(
                    question=task_cleaned,
                    rewards=RewardState(), # Initialize with default RewardState
                    score=ScoreState(score_quality=0.0, score_planner=0.0, score_ground_truth=0.0), # Initialize with default ScoreState values
                    plan="",              # Now has a default in TasksState, but explicit is fine
                    solution=""           # Now has a default in TasksState, but explicit is fine
                )
            )
    state.tasks = state.tasks + tasks
    state.status = "challenged"
    return state