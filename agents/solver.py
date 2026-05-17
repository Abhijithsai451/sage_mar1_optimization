from states.agent_state import SAGEAgentState
from config.logger_config import  sars_logger as logger

solver_prompt="""
Role: Solver Agent
Description:
You will solve the problem by following the verified plan and prioritizing correct, well-reasoned content over formatting tricks.
Input:
- Problem: {question}
- Verified Plan: {plan}
Instructions:
- Explain the key reasoning steps clearly
- Follow the answer-format instruction in the problem statement exactly
- Do not introduce additional wrappers/tags unless explicitly required
"""

def solver(state: SAGEAgentState)-> SAGEAgentState:


    return state
