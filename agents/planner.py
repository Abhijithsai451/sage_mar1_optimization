planner_prompt="""
Role: Planner Agent
Description:
You will review the user problem and propose a concise plan that a solver can follow.
Problem:{question}
Respond using:
<plan>
1. ...
2. ...
</plan>
"""