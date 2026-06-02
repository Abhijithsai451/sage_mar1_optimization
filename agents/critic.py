from langchain_core.messages import HumanMessage, SystemMessage
import re
from config.database_utils import save_agent_state
from config.logger_config import sars_logger as logger
from config.model_config import BackboneModel
from config.prompts import critic_prompt
from states.agent_state import SAGEAgentState

def critic(state: SAGEAgentState, model: BackboneModel, lora_name: str) -> SAGEAgentState:
    logger.info(f"[Critic]: Initiated the Critic Agent and Evaluating the Questions, Plan and Solutions")
    user_content = f"evaluate each task in the list and provide a scores and rewards between 1-10 for each task as per the criteria"
    tasks = state.tasks
    tasks = str(tasks)
    messages = [
        SystemMessage(content=critic_prompt),
        HumanMessage(content=user_content + tasks)
    ]
    lora_model = model.with_config(configurable={"model": lora_name})
    response = lora_model.invoke(messages)
    print(response.content)
    logger.info("[Critic_Planner]: Critic Agent created the Scores for the Plans ")
    scores_blocks = re.findall(r"<task>(.*?)</task>", response.content, re.DOTALL)
    scores = []
    rewards = []
    logger.info("[Critic_Planner]: Extracting the tasks from the response and creating the state objects")
    for scores_block in scores_blocks:

        question_tags = re.search(r"<question>(.*?)</question>", scores_block, re.DOTALL)
        question = question_tags.group(1).strip() if question_tags else ""
        score_question = re.search(r"<score_questions>(.*?)</score_questions>", scores_block, re.DOTALL)
        score_question = float(score_question.group(1).strip()) if score_question else 2.0

        score_plans = re.search(r"<score_plans>(.*?)</score_plans>", scores_block, re.DOTALL)
        score_plans = float(score_plans.group(1).strip()) if score_plans else 2.0

        score_solutions = re.search(r"<score_solutions>(.*?)</score_solutions>", scores_block, re.DOTALL)
        score_solutions = float(score_solutions.group(1).strip()) if score_solutions else 2.0

        # Extracting the reward values
        reward_diff = re.search(r"<reward_difficulty>(.*?)</reward_difficulty>", scores_block, re.DOTALL)
        reward_diff = float(reward_diff.group(1).strip()) if reward_diff else 2.0
        reward_format = re.search(r"<reward_format>(.*?)</reward_format>", scores_block, re.DOTALL)
        reward_format = float(reward_format.group(1).strip()) if reward_format else 2.0

        scores.append({"question": question, "score_question": score_question, "score_plans": score_plans,
                       "score_solutions": score_solutions, "reward_diff": reward_diff, "reward_format": reward_format})
    logger.info("[Critic_Planner]: Extracted the Plan Scores and creating the state objects")
    for i in range(len(scores)):
        question = scores[i].get("question")
        score_question = scores[i].get("score_question")
        score_plans = scores[i].get("score_plans")
        score_solutions = scores[i].get("score_solutions")
        reward_diff = scores[i].get("reward_diff")
        reward_format = scores[i].get("reward_format")
        if question == state.tasks[i].question:
            state.tasks[i].score.score_ground_truth = score_question
            state.tasks[i].score.score_planner = score_plans
            state.tasks[i].score.score_quality = score_solutions
            state.tasks[i].rewards.reward_diff = reward_diff
            state.tasks[i].rewards.reward_format = reward_format
    logger.info("[Critic_Planner]: Updated the state with the Planning Scores and Rewards")
    save_agent_state(state)
    return state