from langchain.agents import create_agent

from agents.critic import reward_challenger, evaluate_question_prompt
from states.agent_state import SAGEAgentState
from config.logger_config import  sars_logger as logger

def init_agents(model_name):
    logger.info("Initializing Agents...")


    planner = create_agent(
        model=model_name,
        response_format=SAGEAgentState
    )
    solver = create_agent(
        model=model_name,
        response_format=SAGEAgentState
    )
    return [planner, solver]
