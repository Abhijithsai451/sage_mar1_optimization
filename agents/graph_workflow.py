from langgraph.constants import START, END
from langgraph.graph import StateGraph

from agents.agent_state import SAGEAgentState
from agents.challenger import challenger_agent

graph = StateGraph(SAGEAgentState)

graph.add_node("challenger", challenger_agent)
graph.set_entry_point("challenger")
graph.add_edge("challenger", END)

app = graph.compile()
message = app.invoke("hey")
print(message)

