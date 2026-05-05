import os
from IPython.display import Image
from langchain_core.messages import HumanMessage, AIMessage

from agents.graph_workflow import create_graph
from agents.init_agents import init_agents
from config.model_config import get_backbone
from dotenv import load_dotenv

from config import model_config
from config.data_config import get_data
from config.logger_config import  sars_logger as logger
from states.agent_state import SAGEAgentState

load_dotenv()
data_dir = os.getenv("DATA_DIR")
greeting = os.getenv("GREETING")
def main():
    logger.info("[SAGE: Multi Agent Self Evolution for LLM Reasoning]")
    logger.info("Importing the dataset: [GSM8K]")
    dataset =get_data(data_dir)
    logger.info("Importing the dataset: [GSM8K] Completed")

    # Importing the Backbone Model
    logger.info("Creating the backbone model: [llama3:8b]")
    llama = get_backbone()
    response = llama.test_model(greeting)
    logger.info("Backbone model Successfully created: [llama3:8b]:")
    print(f"Model-> {response}")

    # Creating the Agents and the Workflow
    agents = init_agents(llama)
    logger.info("Agents Successfully created")
    graph = create_graph(agents)
    logger.info("Graph Successfully created and workflow visualization saved to agent_workflow.png")

    query = ["Add all the 25 numbers from 1 to 25","Pick a random number between 1 and 25 and multiply by itself."]
    # Training Loop comes here.
    messages = [HumanMessage(content = f"Generate 3 different mathematical tasks similar to the {query}")]
    initial_state: SAGEAgentState = {
        "messages": messages,
        "input": query,
        "tasks": []
    }

    response = graph.invoke(initial_state)
    print(response)

    if 'tasks' in response and response['tasks']:
        print("\nGenerated Tasks from Challenger:")
        for task in response['tasks']:
            print(task)
    else:
        print("\nNo tasks generated or found in the final graph state from the challenger.")

if __name__ == "__main__":
    main()