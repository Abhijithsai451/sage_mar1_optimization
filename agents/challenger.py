import os
import re

from langchain.agents import create_agent
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
from dotenv import load_dotenv

from config.model_config import get_backbone, BackboneModel
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

def challenger(state: SAGEAgentState)-> SAGEAgentState:
    model = get_backbone()
    user_content = f"Dataset Reference Examples: \n {state['input']} \n\nPlease generate a set of 5 new tasks following the reference style."
    messages = [
        SystemMessage(content = challenger_policy),
        HumanMessage(content = user_content)
    ]
    reponse = model.invoke(messages)
    content = reponse.content
    # Extracting the question from tags
    match = re.search(r'<question>(.*?)</question>', content, re.DOTALL)
    extracted_task = match.group(1).strip() if match else content

    return {
        "tasks": [extracted_task],
        "messages": [AIMessage(content=f"Challenger generated task: {extracted_task}")],
        "status": "challenged"
    }
