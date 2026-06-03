import os

import numpy as np
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from torch.optim import Adam

from agents.graph_workflow import create_graph
from config.data_config import get_data
from config.logger_config import sars_logger as logger
from config.model_config import get_backbone
from config.database_utils import init_database, clear_db
from config.trainer import sage_trainer
from states.agent_state import SAGEAgentState

load_dotenv()
data_dir = os.getenv("DATA_DIR")
greeting = os.getenv("GREETING")

def test():
    # Reset and prepare tracking state records
    clear_db()
    init_database()

    logger.info("[SAGE: Multi-Agent Self-Evolution for LLM Reasoning]")
    logger.info("Importing the dataset: [GSM8K]")
    train_data = get_data(split="train")
    logger.info("Importing the dataset: [GSM8K] Completed")

    # Access singleton backbone infrastructure
    logger.info("Loading unified backbone instance...")
    model = get_backbone()

    # Run simple setup greeting response trace
    response = model.test_model(greeting)
    logger.info(f"Backbone connection alive. Verification response: -> {response}")

    # Compile routing graph topology using dynamic config overrides
    logger.info("Compiling LangGraph Multi-Agent Adaptive System Topology...")
    graph = create_graph()
    logger.info("Graph compiled successfully. Visualization layout saved.")

    # Begin self-evolutionary training cycle
    logger.info("Commencing RL optimization phase via on-policy policy gradient steps...")
    sage_trainer(
        backbone=model,
        graph=graph,
        train_data=train_data,
        batch_size=4,
        optimization_epochs=4
    )

if __name__ == "__main__":
    test()
