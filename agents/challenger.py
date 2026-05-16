import re
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from config import prompts
from config.model_config import get_backbone, BackboneModel
from states.agent_state import SAGEAgentState
from states.rewards import RewardState
from states.scores import ScoreState
from states.tasks_state import TasksState

def challenger(state: SAGEAgentState, model: BackboneModel)-> SAGEAgentState:
    user_content = f"Dataset Reference Examples: \n {state.input} \n\nPlease generate a set of 5 new tasks following the reference style."
    messages = [
        SystemMessage(content = prompts.challenger_policy),
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
                    rewards=RewardState(),
                    score=ScoreState(score_quality="", score_planner="", score_ground_truth=""),
                    plan="",
                    solution=""
                )
            )
    state.tasks = state.tasks + tasks
    state.status = "challenged"
    return state