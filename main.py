import os
from config.model_config import get_backbone
from dotenv import load_dotenv

from config import model_config
from config.data_config import get_data
from config.logger_config import  sars_logger as logger
load_dotenv()
data_dir = os.getenv("DATA_DIR")
greeting = os.getenv("GREETING")
def main():
    logger.info("[SAGE: Multi Agent Self Evolution for LLM Reasoning]")
    logger.info("Importing the dataset: [GSM8K]")
    dataset =get_data(data_dir)
    logger.info("Importing the dataset: [GSM8K] Completed")
    logger.info("Creating the backbone model: [llama3:8b]")
    llama = get_backbone()
    response = llama.test_model(greeting)
    logger.info("Backbone model Successfully created: [llama3:8b]:")
    print(f"Model-> {response}")

if __name__ == "__main__":
    main()