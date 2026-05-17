from functools import partial

from langgraph.constants import START, END
from langgraph.graph import StateGraph
from agents.critic import critic as critic_node
from config.logger_config import  sars_logger as logger
from states.agent_state import SAGEAgentState
from agents.challenger import challenger as challenger_node
from agents.planner import planner as planner_node
from agents.solver import solver as solver_node

def save_graph_image(app, filename="agent_workflow.png"):
    try:
        # get_graph() generates the internal representation
        # draw_mermaid_png() renders it via a web service (no local dependencies needed)
        graph_png = app.get_graph().draw_mermaid_png()

        with open(filename, "wb") as f:
            f.write(graph_png)
        logger.info(f"Workflow visualization saved to {filename}")
    except Exception as e:
        logger.info(f"Could not generate graph image: {e}")
        # Fallback: Print the Mermaid string which you can paste into mermaid.live
        logger.info("\nMermaid Diagram String:")
        logger.info(app.get_graph().draw_mermaid())


def create_graph(agents, llm_instance):
    planner, solver = agents

    graph = StateGraph(SAGEAgentState)
    challenger = partial(challenger_node, model=llm_instance)
    critic = partial(critic_node, model=llm_instance)
    planner = partial(planner_node, model=llm_instance)
    solver = partial(solver_node, model=llm_instance)


    graph.add_node("challenger",challenger)
    graph.add_node("critic",critic)
    graph.add_node("planner",planner)
    graph.add_node("solver",solver)

    graph.add_edge(START,"challenger")
    graph.add_edge("challenger","critic")
    graph.add_edge("critic","planner")
    """
    graph.add_edge("planner","critic")
    graph.add_edge("critic","solver")
    graph.add_edge("solver","critic")
    graph.add_edge("critic",END)
    """

    graph.add_edge("planner","critic")
    graph.add_edge("critic","solver")
    graph.add_edge("solver",END)
    graph = graph.compile()
    save_graph_image(graph, filename="agent_workflow.png")
    return graph

