from functools import partial
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from agents.challenger import challenger as challenger_node
from agents.critic import critic as critic_node
from agents.planner import planner as planner_node
from agents.solver import solver as solver_node
from config.logger_config import sars_logger as logger
from config.model_config import get_backbone
from states.agent_state import SAGEAgentState

def save_graph_image(app, filename="agent_workflow.png"):
    try:
        graph_png = app.get_graph().draw_mermaid_png()
        with open(filename, "wb") as f:
            f.write(graph_png)
        logger.info(f"Workflow visualization saved to {filename}")
    except Exception as e:
        logger.info(f"Could not generate graph image: {e}")
        # Fallback: Print the Mermaid string which you can paste into mermaid.live
        logger.info("\nMermaid Diagram String:")
        logger.info(app.get_graph().draw_mermaid())
"""
def create_graph(llm_instance):
    graph = StateGraph(SAGEAgentState)
    challenger = partial(challenger_node, model=llm_instance)
    question_critic = partial(question_critic_node, model=llm_instance)
    planner = partial(planner_node, model=llm_instance)
    plan_critic = partial(plan_critic_node, model=llm_instance)
    solver = partial(solver_node, model=llm_instance)
    solution_critic = partial(solution_critic_node, model=llm_instance)

    graph.add_node("challenger", challenger)
    graph.add_node("question_critic", question_critic)
    graph.add_node("planner", planner)
    graph.add_node("plan_critic", plan_critic)
    graph.add_node("solver", solver)
    graph.add_node("solution_critic", solution_critic)

    graph.add_edge(START, "challenger")
    graph.add_edge("challenger", "question_critic")
    graph.add_edge("question_critic", "planner")
    graph.add_edge("planner","plan_critic")
    graph.add_edge("plan_critic","solver")
    graph.add_edge("solver","solution_critic")
    graph.add_edge("solution_critic",END)
    graph = graph.compile()
    save_graph_image(graph, filename="agent_workflow.png")
    return graph
"""

def create_graph():
    llm_instance = get_backbone()
    graph = StateGraph(SAGEAgentState)
    challenger = partial(challenger_node, model=llm_instance, lora_name="challenger")
    planner = partial(planner_node, model=llm_instance, lora_name="planner")
    solver = partial(solver_node, model=llm_instance, lora_name="solver")
    critic = partial(critic_node, model=llm_instance, lora_name="critic")

    graph.add_node("challenger", challenger)
    graph.add_node("planner", planner)
    graph.add_node("solver", solver)
    graph.add_node("critic", critic)

    graph.add_edge(START, "challenger")
    graph.add_edge("challenger", "planner")
    graph.add_edge("planner","solver")
    graph.add_edge("solver","critic")
    graph.add_edge("critic",END)
    graph = graph.compile()
    save_graph_image(graph, filename="agent_workflow.png")
    return graph