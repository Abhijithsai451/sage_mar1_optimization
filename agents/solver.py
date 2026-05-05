from states.agent_state import SAGEAgentState

solver_prompt="""
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
"""

def solver_agent(state: SAGEAgentState)-> SAGEAgentState:

    if state['score_planner'] >= state['beta']:
        state['reward_solver'] = (state['w_p']*state['score_planner'])+(state['w_c']*state['score_ground_truth'])+(state['w_f']*state['reward_format'])
    else:
        state['reward_solver'] = (state['w_c']*state['reward_challenger'])+(state['w_f']*state['reward_format'])
    return state
