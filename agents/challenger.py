import os

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import tool
from dotenv import load_dotenv
from states.agent_state import SAGEAgentState
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
<question>
[Your generated task here]
</question>
"""




def create_tasks(state: SAGEAgentState, model):
    """
    This tool will create the tasks based on the given reference materials.
    """
    tasks = model.invoke([
        SystemMessage(content=challenger_policy),
        HumanMessage(content= f"Here is the question and answer {state['input']}, Create a task similar to this")
    ])
    state['tasks'] = tasks

    return {
        "tasks": tasks,
        "status": "challenged"
    }
