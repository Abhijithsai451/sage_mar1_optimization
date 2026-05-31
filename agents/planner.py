import re
from langchain_core.messages import SystemMessage, HumanMessage
from config import prompts
from config.logger_config import sars_logger as logger
from config.model_config import BackboneModel
from config.database_utils import save_agent_state
from states.agent_state import SAGEAgentState


def planner(state: SAGEAgentState, model: BackboneModel,  lora_name: str) -> SAGEAgentState:
    logger.info("[Planner]: Initiating the Planner Agent")
    print(state.tasks)
    user_content = f"For every question in the list. Please generate a concise plan for to solve the question."
    questions = "\n\n".join([f"question {i + 1}:\n{task_item.question}" for i, task_item in enumerate(state.tasks)])
    messages = [
        SystemMessage(content=prompts.planner_policy),
        HumanMessage(content=user_content + questions)
    ]
    lora_model = model.with_config(configurable={"model": lora_name})
    try:
        response = lora_model.invoke(messages)
        print(response.content)
        logger.info("[Planner]: Created the Plans for every question ")
    except Exception as e:
        logger.error(f"[Planner]: Error while creating the plans: {e}")
        raise
    plan_blocks = re.findall(r"<task>(.*?)</task>", response.content, re.DOTALL)
    plans = []
    print(plan_blocks)
    logger.info("[Planner]: Extracting the tasks from the response and creating the state objects")
    for plan_block in plan_blocks:
        question_tags = re.search(r"<question>(.*?)</question>", plan_block, re.DOTALL)
        question = question_tags.group(1).strip()
        plan_tags = re.search(r"<plan>(.*?)</plan>", plan_block, re.DOTALL)
        plan = plan_tags.group(1).strip()
        plans.append({"question": question, "plan": plan})
    logger.info("[Planner]: Extracted the tasks from the response")

    for i in range(len(plans)):
        question = plans[i].get("question")
        plan = plans[i].get("plan")
        if question == state.tasks[i].question:
            state.tasks[i].plan = plan
    state.status = "planned"
    logger.info(f"[Planner]: Updated the state with the new status {state.status}")
    save_agent_state(state)
    return state
