import re

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool

from states.agent_state import SAGEAgentState
from states.critic_state import CriticState
from config.model_config import get_backbone
from states.tasks_state import TasksState

evaluate_question_prompt="""
Role: Question Quality Critic Agent
Description:
You are an expert evaluator. Your task is to assess the quality of a generated question for reasoning benchmarks.
Input:
- Question to evaluate:{question}
Evaluate the question based on the following criteria:
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
Provide your final score.
Important:
Output only one tag like <score>7</score> (replace 7 with your integer score 1-10).
"""
evaluate_plan_prompt = """
Role: Plan Critic Agent
Description:
You are an expert evaluator. Your task is to assess the quality of a proposed plan for solving a problem.
Input:
- Problem: {question}
- Proposed Plan: {plan}
Evaluate the plan based on the following criteria:
<think>
Evaluation Criteria:
- Clarity: Is the plan clear, structured, and easy to follow?
- Completeness: Does it cover all necessary steps to solve the problem?
- Correctness: Is the reasoning and approach logically sound?
- Feasibility: Can following this plan lead to a correct solution?
- Efficiency: Is the approach reasonably efficient, avoiding unnecessary steps?
Scoring Guidelines:
- 8-10: Excellent plan - clear, complete, logically sound, and feasible. Following it should lead to a correct solution.
- 4-7: Acceptable plan - has some gaps or minor issues but the general direction is correct.
- 1-3: Poor plan - unclear, incomplete, logically flawed, or unlikely to lead to a correct solution.
[Write your detailed analysis here, addressing each criterion.]
</think>
Provide your final score.
Important:
Output only one tag like <score>7</score> (replace 7 with your integer score 1-10).
"""
evaluate_solution_prompt = """
Role: Solution Quality Critic Agent
Description:
You are an expert evaluator. Your task is to assess the quality of a generated solution to a given question or problem.
Input:
- Question: {question}
- Generated Solution: {answer}
Evaluate the solution based on the following criteria:
<think>
Evaluation Criteria:
- Accuracy: Is the solution factually correct with no errors in reasoning, arithmetic, units, or assumptions?
- Completeness: Does it fully address the question with all necessary steps and derivations?
- Coherence: Is the reasoning logical and free from contradictions or hallucinations?
- Conciseness: Is the answer direct without meaningless repetition, rambling, or filler?
- Instruction Following: Does the solution follow any explicit formatting or structural requirements?
Scoring Guidelines:
- 8-10: Excellent solution - entirely correct, complete, logically sound, concise, and follows all instructions.
- 4-7: Acceptable solution - generally on-topic and partially correct, but has omissions or clarity issues (no factual errors).
- 1-3: Poor solution - contains any factual/logic/calculation error, hallucinated content, excessive repetition, or severe irrelevance.
Critical Rules:
- Any factual error (arithmetic, reasoning, common sense, units, invalid assumptions) → score must be [1-3]
- Hallucinated references, fabricated data, or unsupported claims → score must be [1-3]
- Meaningless repetition or excessive rambling → score must be [1-3]
[Write your detailed analysis here, addressing each criterion. If any critical issue exists, note that the score must be [1-3].]
<\think>
Provide your final score.
Important:
Output only one tag like <score>7</score> (replace 7 with your integer score 1-10).
"""

def reward_challenger(tasks):
    """Constructs the messages for the challenger reward phase."""
    user_content = """
    Review every task in  {state['tasks']} . Please provide a reward between 1-10 for each task. 
    Please follow the basic equations below for scoring. 
    state['reward_challenger'] = (state['score_quality'] + state['reward_diff'] + state['reward_format']) / 3
    here
    state['reward_format'] is the reward for the format of the questions. (score it between 1 to 10)
    state['reward_diff'] is the difficulty of the questions from the tasks. (score it between 1 to 10)
    state['score_quality'] is the score of the quality of the questions. (score it between 1 to 10)
    
    Eventually the average of the three score will be the reward for the challenger. Assign it for each question with a tag like 
    <reward_challenger>7</reward_challenger> (replace 7 with your integer score 1-10).
    """
    messages = [
        SystemMessage(
            content=evaluate_question_prompt + "\nOutput in format: <tasks><task><question>...</question><reward>...</reward></task></tasks>"),
        HumanMessage(content=f"Evaluate these tasks:\n{tasks}")
    ]

    return messages

def critic(state: SAGEAgentState):
    model = get_backbone()
    current_status = state["status"]
    # 1. Determine which prompt to use based on the loop status
    if current_status == "challenged":
        tasks = "\n".join([f"- {t}" for t in state["tasks"]])
        messages = reward_challenger(tasks)
        response = model.invoke(messages)
        print(f"Response from the critic is: {response.content}")
        content = response.content

    elif current_status == "planned":
        print("Evaluating the Plan")

    elif current_status == "solved":
        print("Evaluating the Solution")
    else:
        return state

    return state

