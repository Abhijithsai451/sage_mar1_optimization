from states.agent_state import SAGEAgentState

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

def planner_agent(state: SAGEAgentState)-> SAGEAgentState:
    state['reward_planner'] = (state['lambda_plan']*state['score_plan'])+(state['lambda_format']* state['reward_format'])

    return state

