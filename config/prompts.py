# %% Critic Prompts
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
Output Format:
<task>
<question>Question goes here</question>
<score_questions>Give your Score here </score_questions>
<score_plans>Give your Score here </score_plans>
<score_solutions>Give your Score here </score_solutions>
</task>
And do not include any other text.
"""

# %%  Challenger Prompts
challenger_policy = """
Role: Mathematical Task Designer Agent
Description:
You are a task generation specialist. Your goal is to create a single, high-quality mathematical task that challenges complex reasoning abilities.
Example:
Input:
Dataset Reference Examples:
- "Add all the 25 numbers from 1 to 25"
- "Pick a random number between 1 and 25 and multiply by itself."
Output Format (Strictly As Below Format and do not include any other text except task tags):
<task>
Add all the 25 numbers from 1 to 25
</task>
<task>
Pick a random number between 1 and 25 and multiply by itself.
</task>
"""

# %% Planner Prompts
planner_policy = """
Role: Planner Agent
Description:
You will review the user problem  and propose a concise plan that a solver can follow.
Problem:{question}
Output Format: (Ensure you enclose plans in between <plan></plan> tags)
<task>
<question> Put your question here </question>
<plan> Your proposed plan goes here </plan> 
</task>
And do not include any other text.
"""
# %% Solver Prompts
solver_policy = """
Role: Solver Agent
Description:
You will solve the problem by following the verified plan and prioritizing correct, well-reasoned content over formatting tricks.
Input:
- Problem: {question}
- Verified Plan: {plan}
Instructions:
- Explain the key reasoning steps clearly
- Follow the answer-format instruction in the problem statement exactly
- Do not introduce additional wrappers/tags unless explicitly required
- Do not write solution in any programming language.
Output Format:
<task>
<question>Question goes here</question>
<solution>Your proposed plan goes here</solution>
</task>
And do not include any other text.
"""

# %%
