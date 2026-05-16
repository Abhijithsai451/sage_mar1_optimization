import json
import re

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool

from config import prompts
from states.agent_state import SAGEAgentState
from states.parameter_state import ParameterState
from config.model_config import get_backbone, BackboneModel
from states.rewards import RewardState
from states.scores import ScoreState
from states.tasks_state import TasksState

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

def critic(state: SAGEAgentState, model: BackboneModel)-> SAGEAgentState:
    current_status = state.status
    if current_status == "challenged":
        user_content = f"evaluate each question in the list and provide a score between 1-10 for each task."
        questions = "\n\n".join([f"Task {i+1}:\n{task_item.question}" for i, task_item in enumerate(state.tasks)])
        #messages = reward_challenger(questions)
        messages = [
            SystemMessage(content=prompts.evaluate_question_prompt),
            HumanMessage(content=user_content + questions)
        ]
        response = model.invoke(messages)
        task_blocks = re.findall(r"<task>(.*?)</task>", response.content, re.DOTALL)
        tasks = []
        extracted_tasks = []
        for task_block in task_blocks:
            question_tags = re.search(r"<question>(.*?)</question>", task_block, re.DOTALL)
            question = question_tags.group(1).strip()
            score_tags = re.search(r"<score_ground_truth>(.*?)</score_ground_truth>", task_block, re.DOTALL)
            score = score_tags.group(1).strip()
            tasks.append({"question": question, "score": score})
        for task in tasks:
            question = task.get("question")
            score = task.get("score")
            extracted_tasks.append(
                TasksState(
                    question=question,
                    rewards=RewardState(),
                    score=ScoreState(score_quality="", score_planner="", score_ground_truth=score),
                    plan="",
                    solution=""
                )
            )
        state.tasks = extracted_tasks
        print(state.tasks)
        state.status = "challenged"


    elif current_status == "planned":
        print("Evaluating the Plan")

    elif current_status == "solved":
        print("Evaluating the Solution")
    else:
        return state

    return state

