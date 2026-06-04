import random

import numpy as np
from langchain_core.messages import HumanMessage, SystemMessage

from config.losses_optimizers import compute_advantage, optimize_adapters
from config.prompts import challenger_policy, critic_prompt, solver_policy, planner_policy
from states.agent_state import SAGEAgentState
from states.parameter_state import ParameterState
from config.logger_config import sars_logger as logger


def sage_trainer(backbone, graph, train_data, batch_size= 4, optimization_epochs =4):
    """
    Drives multi-agent sampel collection, normalization advantage loops, and training.
    """
    history = {"challenger": [], "planner": [], "solver": [], "critic": []}
    dataset_samples = list(train_data)
    random.shuffle(dataset_samples)

    current_index = 0
    step = 0

    default_params = ParameterState(
        id = "train_param",
        alpha = 0.7,
        beta = 0.3,
        lambda_plan = 0.5,
        lambda_format = 0.0,
        w_p = 0.2,
        w_c = 0.6,
        w_f = 0.2,
    )
    while current_index < len(dataset_samples):
        step +=1
        logger.info(f"====================== EVOLUTION STEP {step} ======================")
        batch_trajectories = {"challenger": [], "planner": [], "solver": [], "critic": []}
        end_index = min(current_index + batch_size, len(dataset_samples))
        current_batch = dataset_samples[current_index:end_index]
        current_index = end_index
        logger.info(f"Collected {len(current_batch)} samples for training")

        for sample in current_batch:
            question = sample["question"]  if isinstance(sample, dict) else sample.question

            initial_state = SAGEAgentState(
                            messages =[] ,
                            input=[question],
                            parameter_state=default_params,
                            tasks=[], )

            try:
                final_state = graph.invoke(initial_state)
            except Exception as e:
                logger.error(f"Error while invoking the graph: {e}")
                continue
            if not final_state or not hasattr(final_state, "tasks") or not final_state.tasks:
                continue

            task_adv_c, task_adv_p, task_adv_s, task_adv_cr = [], [], [], []
            for task in final_state.tasks:
                r_c = task.rewards.reward_challenger
                r_p = task.rewards.reward_planner
                r_s = task.rewards.reward_solver
                r_cr = task.rewards.reward_format + task.rewards.reward_diff

                history["challenger"].append(r_c)
                history["planner"].append(r_p)
                history["solver"].append(r_s)
                history["critic"].append(r_cr)

                task_adv_c = compute_advantage(r_c, history['challenger'])
                task_adv_p = compute_advantage(r_p, history['planner'])
                task_adv_s = compute_advantage(r_s, history['solver'])
                task_adv_cr = compute_advantage(r_c, history['critic'])

                # Aggregate task rewards into unified sequence advantages matching the unified node calls
                mean_adv_c = float(np.mean(task_adv_c))
                mean_adv_p = float(np.mean(task_adv_p))
                mean_adv_s = float(np.mean(task_adv_s))
                mean_adv_cr = float(np.mean(task_adv_cr))

                batch_trajectories['challenger'].append({
                    "prompt": challenger_policy,
                    "generation": f"<task>{task.question}</task>",
                    "advantage": mean_adv_c
                })
                # Planner Memory Pool
                planner_prompt_str = backbone.get_formatted_prompt_string([
                    SystemMessage(content=planner_policy),
                    HumanMessage(
                        content="For every question in the list. Please generate a concise plan for to solve the question.\n\nquestion 1:\n" + task.question)
                ])
                batch_trajectories["planner"].append({
                    "prompt": planner_prompt_str,
                    "generation": f"<task><question>{task.question}</question><plan>{task.plan}</plan></task>",
                    "advantage": mean_adv_p
                })

                # Solver Memory Pool
                solver_prompt_str = backbone.get_formatted_prompt_string([
                    SystemMessage(content=solver_policy),
                    HumanMessage(
                        content=f"For every question and plan in the list. Please generate a detailed solution for the question.\n\nPlan 1:\n{task.question} + \n{task.plan}")
                ])
                batch_trajectories["solver"].append({
                    "prompt": solver_prompt_str,
                    "generation": f"<task><question>{task.question}</question><solution>{task.solution}</solution></task>",
                    "advantage": mean_adv_s
                })

                # Critic Memory Pool
                critic_prompt_str = backbone.get_formatted_prompt_string([
                    SystemMessage(content=critic_prompt),
                    HumanMessage(
                        content=f"evaluate each task in the list and provide a scores and rewards between 1-10 for each task as per the criteria" + str(
                            [task]))
                ])

                critic_gen = (
                    f"<task>\n<question>{task.question}</question>\n"
                    f"<score_questions>{task.score.score_ground_truth}</score_questions>\n"
                    f"<score_plans>{task.score.score_planner}</score_plans>\n"
                    f"<score_solutions>{task.score.score_quality}</score_solutions>\n"
                    f"<reward_difficulty>{task.rewards.reward_diff}</reward_difficulty>\n"
                    f"<reward_format>{task.rewards.reward_format}</reward_format>\n</task>"
                )
                batch_trajectories["critic"].append({
                    "prompt": critic_prompt_str,
                    "generation": critic_gen,
                    "advantage": mean_adv_cr
                })

                # --- 4. OPTIMIZATION UPDATES ON SEPARATE PEFT FLIPS ---
            logger.info("Executing token-level REINFORCE++ updates over the multiplexed memory pool adapters...")
            optimize_adapters(backbone, "challenger", batch_trajectories["challenger"], epochs=optimization_epochs)
            optimize_adapters(backbone, "planner", batch_trajectories["planner"], epochs=optimization_epochs)
            optimize_adapters(backbone, "solver", batch_trajectories["solver"], epochs=optimization_epochs)
            optimize_adapters(backbone, "critic", batch_trajectories["critic"], epochs=optimization_epochs)

        logger.info("Multi-agent self-evolution training cycle completed successfully.")