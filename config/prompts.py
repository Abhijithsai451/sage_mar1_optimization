#%% Critic Prompts
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
Important:Format your response in as follow. 
<task>
<question>Question goes here</question>
<score_ground_truth>8</score_ground_truth>
</task>
And do not include any other text.
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
Provide your final score.
Important:Format your response in as follow. 
<task>
<question>Question goes here</question>
<score_ground_truth>8</score_ground_truth>
<score_planner>8</score_planner>
</task>
And do not include any other text.
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

#%%  Challenger Prompts
challenger_policy ="""
Role: Task Designer Agent
Description:
You are a task generation specialist. Your goal is to create a single, high-quality evaluation task that challenges complex reasoning abilities.
Design Constraints:
- Self-contained with clear problem statement
- Non-trivial: requires multiple reasoning steps or constraint satisfaction
- Deterministic or tightly bounded (avoid subjective judgment)
- Culturally neutral, no real-time data dependency
- Difficult but solvable
Avoid:
- Trivia or opinion-based prompts
- Ambiguous success criteria
- Web-dependent or time-sensitive content
- Unsolvable or ill-defined problems
Respond using:
<task>
[Your generated task here]
</task>
"""


