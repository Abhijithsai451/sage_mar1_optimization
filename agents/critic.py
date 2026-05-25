from langchain_core.messages import HumanMessage, SystemMessage
import re
from config.database_utils import save_agent_state
from config.logger_config import sars_logger as logger
from config.model_config import BackboneModel
from states.agent_state import SAGEAgentState

critic_prompt = """
Role: Critic Agent
Description:
You are an expert evaluator. Your task is to assess the quality of a generated questions, plans and solutions for reasoning benchmarks.
Input:
- Tasks - {Tasks} - 
Each task is a dictionary with the following keys:
- question: The question to be evaluated.
- plan: The proposed plan for solving the question.
- solution: The generated solution to the question.
Evaluate the question, Plan and Solution based on the following criteria:
<think>
Evaluation Criteria:
- Solvability: Is the question solvable with sufficient information? No internal contradictions?
- Logical Soundness: Is the question logically coherent and does not violate common sense?
- Clarity: Is the wording unambiguous with clear objectives and constraints?
- Appropriateness: Is it safe, relevant, and actually in the form of a question?
- Conciseness: Is it free from redundant repetition or unnecessary complexity?
Scoring Guidelines:
- 8-10: Excellent question - fully clear, logically sound, solvable, self-contained, and concise. Appropriate for evaluation purposes.
- 4-7: Acceptable question - has some ambiguity or missing details but no fatal flaws in solvability or logic.
- 1-3: Poor question - unsolvable, contradictory, violates common sense, unsafe, too open-ended, or not a valid question.
Critical Rule:
If any unsolvability or commonsense violation exists, score must be [1-3].
[Write your detailed analysis here, addressing each criterion.]
<\think>
Provide your final score as score_ground_truth.
Important: Strictly Format your response in as follow. 
<task>
<question>Question goes here</question>
<score_questions>Give your Score here </score_questions>
<score_plans>Give your Score here </score_plans>
<score_solutions>Give your Score here </score_solutions>
</task>
And do not include any other text.
"""

def critic(state: SAGEAgentState, model: BackboneModel) -> SAGEAgentState:
    logger.info(f"[Critic_Challenger]: Initiated the Critic Agent and Evaluating the Questions, Plan and Solutions")
    user_content = f"evaluate each task in the list and provide a score between 1-10 for each task as per the criteria"
    tasks = state.tasks
    tasks = str(tasks)
    messages = [
        SystemMessage(content=critic_prompt),
        HumanMessage(content=user_content + tasks)
    ]
    response = model.invoke(messages)
    print(response.content)
    logger.info("[Critic_Planner]: Critic Agent created the Scores for the Plans ")
    scores_blocks = re.findall(r"<task>(.*?)</task>", response.content, re.DOTALL)
    scores = []
    logger.info("[Critic_Planner]: Extracting the tasks from the response and creating the state objects")
    for scores_block in scores_blocks:
        question_tags = re.search(r"<question>(.*?)</question>", scores_block, re.DOTALL)
        question = question_tags.group(1).strip()
        score_question = re.search(r"<score_questions>(.*?)</score_questions>", scores_block, re.DOTALL)
        score_question = score_question.group(1).strip()
        score_plans = re.search(r"<score_plans>(.*?)</score_plans>", scores_block, re.DOTALL)
        score_plans = score_plans.group(1).strip()
        score_solutions = re.search(r"<score_solutions>(.*?)</score_solutions>", scores_block, re.DOTALL)
        score_solutions = score_solutions.group(1).strip()
        scores.append({"question": question, "score_question": score_question, "score_plans": score_plans, "score_solutions": score_solutions})
    print(scores)
    logger.info("[Critic_Planner]: Extracted the Plan Scores and creating the state objects")
    for i in range(len(scores)):
        question = scores[i].get("question")
        score_question = scores[i].get("score_question")
        score_plans = scores[i].get("score_plans")
        score_solutions = scores[i].get("score_solutions")
        if question == state.tasks[i].question:
            state.tasks[i].score.score_ground_truth = score_question
            state.tasks[i].score.score_planner = score_plans
            state.tasks[i].score.score_quality = score_solutions
    logger.info("[Critic_Planner]: Updated the state with the Planning Scores")
    save_agent_state(state)
    return state