import re
from langchain_core.messages import SystemMessage, HumanMessage
from config import prompts
from config.logger_config import sars_logger as logger
from config.model_config import BackboneModel
from config.database_utils import save_agent_state
from config.prompts import challenger_policy
from states.agent_state import SAGEAgentState
from states.rewards import RewardState
from states.scores import ScoreState
from states.tasks_state import TasksState


def challenger(state: SAGEAgentState, model: BackboneModel, lora_name: str) -> SAGEAgentState:
    logger.info("[Challenger]: Initiating the Challenger Agent")
    user_content = f"Dataset Reference Examples: \n {state.input} \n\nPlease generate a set of 5 new tasks following the reference style."
    messages = [
        SystemMessage(content=challenger_policy),
        HumanMessage(content=user_content)
    ]
    lora_model = model.with_config(configurable={"model": lora_name})
    response = lora_model.invoke(messages)
    content = response.content
    #print(content)
    logger.info("[Challenger]: Challenger Agent created the Response ")
    content = re.findall(r"<task>(.*?)</task>", content, re.DOTALL)
    tasks = []
    logger.info("[Challenger]: Extracting the tasks from the response and creating the state objects")
    for task in content:
        task_cleaned = task.strip()
        if task_cleaned:
            tasks.append(TasksState(
                question=task_cleaned,
                rewards=RewardState(),
                score=ScoreState(score_quality=0.0, score_planner=0.0, score_ground_truth=0.0),
                plan="",
                solution="")
            )
    #print("tasks: ", tasks)
    state.tasks = state.tasks + tasks
    #print("state.tasks: ", state.tasks)
    logger.info("[Challenger]: Updated the state with the new tasks")
    save_agent_state(state)
    return state

def challenger_test(state: SAGEAgentState, model: BackboneModel) -> SAGEAgentState:
    logger.info("[Challenger]: Initiating the Challenger Agent")
    user_content = f"Dataset Reference Examples: \n {state.input} \n\nPlease generate a set of 5 new tasks following the reference style."
    messages = [
        SystemMessage(content=challenger_policy),
        HumanMessage(content=user_content)
    ]
    response = model.invoke(messages)
    content = response.content
    #print(content)
    logger.info("[Challenger]: Challenger Agent created the Response ")
    content = re.findall(r"<task>(.*?)</task>", content, re.DOTALL)
    tasks = []
    logger.info("[Challenger]: Extracting the tasks from the response and creating the state objects")
    for task in content:
        task_cleaned = task.strip()
        if task_cleaned:
            tasks.append(TasksState(
                question=task_cleaned,
                rewards=RewardState(),
                score=ScoreState(score_quality=0.0, score_planner=0.0, score_ground_truth=0.0),
                plan="",
                solution="")
            )
    #print("tasks: ", tasks)
    state.tasks = state.tasks + tasks
    #print("state.tasks: ", state.tasks)
    logger.info("[Challenger]: Updated the state with the new tasks")
    save_agent_state(state)
    return state