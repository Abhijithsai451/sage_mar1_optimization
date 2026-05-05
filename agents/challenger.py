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
    """
    First LLM Call to create tasks. .
    Each task string in the list MUST be wrapped in <question> and </question> tags.
    Example: ["<question>What is 2+2?</question>", "<question>Solve for x...</question>"]
    """
    model = BackboneModel().model
    messages_for_llm = [
        SystemMessage(content=challenger_policy),
        HumanMessage(
            content=f"Here is the question and answer {state['input']}, Create a task similar to this question")
    ]
    response = model.invoke(messages_for_llm)
    generated_text = response.content if isinstance(response, AIMessage) else str(response)
    question_pattern = r"<question>(.*?)</question>"
    generated_tasks = re.findall(question_pattern, generated_text, re.DOTALL)

    state['tasks'] = generated_tasks
    return state
