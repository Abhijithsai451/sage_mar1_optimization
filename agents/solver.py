import re

from langchain_core.messages import SystemMessage, HumanMessage

from config import prompts
from config.logger_config import sars_logger as logger
from config.model_config import BackboneModel
from config.database_utils import save_agent_state
from states.agent_state import SAGEAgentState


def solver(state: SAGEAgentState, model: BackboneModel) -> SAGEAgentState:
    logger.info("[Solver]: Initiating the Solver Agent")
    user_content = f"For every question and plan in the list. Please generate a detailed solution for the question."
    plans = "\n\n".join(
        [f"Plan {i + 1}:\n{task_item.question} + \n{task_item.plan}" for i, task_item in enumerate(state.tasks)])
    messages = [
        SystemMessage(content=prompts.solver_policy),
        HumanMessage(content=user_content + plans)
    ]
    response = model.invoke(messages)
    logger.info("[Solver]: Created the Solutions for every question ")
    task_blocks = re.findall(r"<task>(.*?)</task>", response.content, re.DOTALL)
    tasks = []
    logger.info("[Solver]: Extracting the tasks from the response and creating the state objects")
    for task_block in task_blocks:
        question_tags = re.search(r"<question>(.*?)</question>", task_block, re.DOTALL)
        question = question_tags.group(1).strip()
        solution_tags = re.search(r"<solution>(.*?)</solution>", task_block, re.DOTALL)
        solution = solution_tags.group(1).strip()
        tasks.append({"question": question, "solution": solution})
    logger.info("[Solver]: Extracted the tasks from the response")

    for i in range(len(tasks)):
        question = tasks[i].get("question")
        solution = tasks[i].get("solution")
        if question == state.tasks[i].question:
            state.tasks[i].solution = solution
    print(state.tasks)
    logger.info("[Solver]: Updated the state with the new solutions")
    state.status = "solved"
    logger.info(f"[Solver]: Updated the state with the new status {state.status}")
    save_agent_state(state)
    return state
