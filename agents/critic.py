import re

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool

from config import prompts
from config.logger_config import sars_logger as logger
from config.model_config import BackboneModel
from config.database_utils import save_agent_state
from states.agent_state import SAGEAgentState


@tool
def reward_challenger(question):
    """Constructs the messages for the challenger reward phase."""
    user_content = """
    Role: You are a rewarding agent.
    Review every question in {question} . Please provide a reward between 1-10 for each task as shown in the example. 
    example : 
    <rewards>
    <reward_challenger>10</reward_challenger> 
    </rewards>
    Please follow the basic equations below for scoring. 
    state['reward_challenger'] = (state['score_quality'] + state['reward_diff'] + state['reward_format']) / 3
    here
    state['reward_format'] is the reward for the format of the questions. (score it between 1 to 10)
    state['reward_diff'] is the difficulty of the questions from the tasks. (score it between 1 to 10)
    state['score_quality'] is the score of the quality of the questions. (score it between 1 to 10)
    Eventually the average of the three score will be the reward for the challenger. 
    example: 
    
    """
    messages = [
        SystemMessage(
            content=prompts.evaluate_question_prompt + "\nOutput in format: [8,9,3,4]"),
        HumanMessage(content=f"Evaluate these tasks:\n{question}")
    ]

    return messages


def question_critic(state: SAGEAgentState, model: BackboneModel) -> SAGEAgentState:
    current_status = state.status

    if current_status == "challenged":
        logger.info(f"[Critic_Challenger]: Initiated the Critic Agent and Evaluating the {current_status} phase")
        user_content = f"evaluate each question in the list and provide a score between 1-10 for each task."
        questions = "\n\n".join([f"Question {i + 1}:\n{task_item.question}" for i, task_item in enumerate(state.tasks)])
        messages = [
            SystemMessage(content=prompts.evaluate_question_prompt),
            HumanMessage(content=user_content + questions)
        ]
        response = model.invoke(messages)
        logger.info("[Critic_Challenger]: Critic Agent created the Response ")
        scores_blocks = re.findall(r"<task>(.*?)</task>", response.content, re.DOTALL)
        task_scores = []
        logger.info("[Critic_Challenger]: Extracting the tasks from the response and creating the state objects")
        for scores_block in scores_blocks:
            question_tags = re.search(r"<question>(.*?)</question>", scores_block, re.DOTALL)
            question = question_tags.group(1).strip()
            score_tags = re.search(r"<score_ground_truth>(.*?)</score_ground_truth>", scores_block, re.DOTALL)
            score = score_tags.group(1).strip()
            task_scores.append({"question": question, "score": score})
        print(task_scores)
        logger.info("[Critic_Challenger]: Extracted the tasks and creating the state objects")
        for i in range(len(task_scores)):
            question = task_scores[i].get("question")
            score = task_scores[i].get("score")
            if question == state.tasks[i].question:
                state.tasks[i].score.score_ground_truth = score
        logger.info("[Critic_Challenger]: Updated the state with the new tasks")
    save_agent_state(state)
    return state

def plan_critic(state: SAGEAgentState, model: BackboneModel) -> SAGEAgentState:
    current_status = state.status
    if current_status == "planned":
        logger.info(f"[Critic_Planner]: Initiated the Critic Agent and Evaluating the {current_status} phase")
        user_content = f"evaluate each plan in the list and provide a score between 1-10 for each task."
        plans = "\n\n".join([f"Plan {i + 1}:\n{task_item.question} + \n{task_item.plan}" for i, task_item in enumerate(state.tasks)])
        messages = [
                SystemMessage(content=prompts.evaluate_plan_prompt),
                HumanMessage(content=user_content + plans)
            ]
        response = model.invoke(messages)
        logger.info("[Critic_Planner]: Critic Agent created the Scores for the Plans ")
        scores_blocks = re.findall(r"<task>(.*?)</task>", response.content, re.DOTALL)
        plan_scores = []
        logger.info("[Critic_Planner]: Extracting the tasks from the response and creating the state objects")
        for scores_block in scores_blocks:
            question_tags = re.search(r"<question>(.*?)</question>", scores_block, re.DOTALL)
            question = question_tags.group(1).strip()
            score_tags = re.search(r"<score_planner>(.*?)</score_planner>", scores_block, re.DOTALL)
            score = score_tags.group(1).strip()
            plan_scores.append({"question": question, "score": score})
        print(plan_scores)
        logger.info("[Critic_Planner]: Extracted the Plan Scores and creating the state objects")
        for i in range(len(plan_scores)):
            question = plan_scores[i].get("question")
            score = plan_scores[i].get("score")
            if question == state.tasks[i].question:
                state.tasks[i].score.score_planner = score
                logger.info("[Critic_Planner]: Updated the state with the Planning Scores")
    save_agent_state(state)
    return state

def solution_critic(state: SAGEAgentState, model: BackboneModel) -> SAGEAgentState:
    current_status = state.status
    if current_status == "solved":
        logger.info(f"[Critic_Solver]: Initiated the Critic Agent and Evaluating the {current_status} phase")
        user_content = f"evaluate each Solution in the list and provide a score between 1-10 for each task."
        solutions = "\n\n".join(
            [f"question {i + 1}:\n{task_item.question} + \n{task_item.solution}" for i, task_item in enumerate(state.tasks)])
        messages = [
            SystemMessage(content=prompts.evaluate_solution_prompt),
            HumanMessage(content=user_content + solutions)
        ]
        response = model.invoke(messages)
        logger.info("[Critic_Solver]: Critic Agent created the Scores for the Solutions ")
        scores_blocks = re.findall(r"<task>(.*?)</task>", response.content, re.DOTALL)
        solution_scores = []
        logger.info("[Critic_Solver]: Extracting the tasks from the response and creating the state objects")
        for scores_block in scores_blocks:
            question_tags = re.search(r"<question>(.*?)</question>", scores_block, re.DOTALL)
            question = question_tags.group(1).strip()
            score_tags = re.search(r"<score_quality>(.*?)</score_quality>", scores_block, re.DOTALL)
            score = score_tags.group(1).strip()
            solution_scores.append({"question": question, "score": score})
        print(solution_scores)
        logger.info("[Critic_Solver]: Extracted the Plan Scores and creating the state objects")
        for i in range(len(solution_scores)):
            question = solution_scores[i].get("question")
            score = solution_scores[i].get("score")
            if question == state.tasks[i].question:
                state.tasks[i].score.score_quality = score
        logger.info("[Critic_Solver]: Updated the state with the Planning Scores")
    save_agent_state(state)
    return state