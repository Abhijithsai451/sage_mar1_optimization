import re
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from config import prompts
from config.model_config import get_backbone, BackboneModel
from states.agent_state import SAGEAgentState
from states.rewards import RewardState
from states.scores import ScoreState
from states.tasks_state import TasksState
from config.logger_config import  sars_logger as logger

def challenger(state: SAGEAgentState, model: BackboneModel)-> SAGEAgentState:
    logger.info("[Challenger]: Initiating the Challenger Agent")
    user_content = f"Dataset Reference Examples: \n {state.input} \n\nPlease generate a set of 5 new tasks following the reference style."
    messages = [
        SystemMessage(content = prompts.challenger_policy),
        HumanMessage(content = user_content)
    ]
    response = model.invoke(messages)
    content = response.content
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
                    score=ScoreState(score_quality="", score_planner="", score_ground_truth=""),
                    plan="",
                    solution=""
                )
            )
    state.tasks = state.tasks + tasks
    logger.info("[Challenger]: Updated the state with the new tasks")
    state.status = "challenged"
    return state