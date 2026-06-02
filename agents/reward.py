from states.agent_state import SAGEAgentState
from config.logger_config import sars_logger as logger

def reward_agent(state: SAGEAgentState) -> SAGEAgentState:
    logger.info(f"[Reward]: Initiated the tool Rewards to Evaluating the Questions, Plan and Solutions")
    tasks = state.tasks
    parameters = state.parameter_state
    for task in tasks:
        if task.score.score_quality <= parameters.alpha:
            task.rewards.reward_challenger = (task.score.score_ground_truth + task.rewards.reward_format) / 2
        else:
            task.rewards.reward_challenger = (task.score.score_ground_truth + task.rewards.reward_diff +
                                              task.rewards.reward_format) / 3
        task.rewards.reward_planner = parameters.lambda_plan * task.score.score_planner + parameters.lambda_format * task.rewards.reward_format

        task.rewards.reward_solver = (parameters.w_p * task.score.score_planner + parameters.w_c * task.score.score_ground_truth
                                        + parameters.w_f * task.rewards.reward_format)


    print(state.tasks.rewards)
    return state