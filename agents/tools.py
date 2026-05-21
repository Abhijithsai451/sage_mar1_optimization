


"""
This File has all the tools which are used in the Process to reward the model as per their performance.
"""
from langchain_core.messages import tool


@tool
def reward_challenger(question):
    """Constructs the messages for the challenger reward phase."""
    return f"The question is {question}"

@tool
def reward_planner(question):
    """Constructs the messages for the planner reward phase."""
    return f"The question is {question}"

@tool
def reward_solver(question):
    """Constructs the messages for the solver reward phase."""

    return  f"The question is {question}"
