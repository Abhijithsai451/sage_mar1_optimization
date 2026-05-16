import re

from langchain_core.messages import SystemMessage, HumanMessage
from config.logger_config import  sars_logger as logger
from config import prompts
from config.model_config import BackboneModel
from states.agent_state import SAGEAgentState

def planner(state: SAGEAgentState, model:BackboneModel)-> SAGEAgentState:
    logger.info("[Planner]: Initiating the Planner Agent")
    user_content = f"For every question in the list. Please generate a concise plan for to solve the question."
    questions = "\n\n".join([f"Task {i + 1}:\n{task_item.question}" for i, task_item in enumerate(state.tasks)])
    messages = [
        SystemMessage(content=prompts.planner_policy),
        HumanMessage(content=user_content + questions)
    ]

    response = model.invoke(messages)
    logger.info("[Planner]: Created the Plans for every question ")
    task_blocks = re.findall(r"<task>(.*?)</task>", response.content, re.DOTALL)
    tasks = []
    extracted_tasks = []
    logger.info("[Planner]: Extracting the tasks from the response and creating the state objects")
    for task_block in task_blocks:
        question_tags = re.search(r"<question>(.*?)</question>", task_block, re.DOTALL)
        question = question_tags.group(1).strip()
        plan_tags = re.search(r"<plan>(.*?)</plan>", task_block, re.DOTALL)
        plan = plan_tags.group(1).strip()
        tasks.append({"question": question, "plan": plan})
    logger.info("[Planner]: Extracted the tasks from the response")

    for i in range(len(tasks)):
        print("updating the state")
        question = tasks[i].get("question")
        plan = tasks[i].get("plan")
        if question == state.tasks[i].question:
            print("updating the plan")
            state.tasks[i].plan = plan
    print(state.tasks)
    state.status = "planned"
    logger.info(f"[Planner]: Updated the state with the new status {state.status}")
    return state

