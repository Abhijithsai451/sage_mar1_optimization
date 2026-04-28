from agents.agent_state import SAGEAgentState
from config.model_config import llm
orchestrator_prompt="""
    You are a task Orchestrator Agent. Your task is to orchestrate the task generation process.
    1. User will provide a input prompt.
    2. Understand the prompt and generate the task.
    3. Initiate the task generation process by invoking the task generator agent.
    4. Capture the Analysis and output the task to the user.
"""
def orchestrator_agent(state: SAGEAgentState)-> SAGEAgentState:
    pass



