import os

from dotenv import load_dotenv

from config import model_config
from config.data_config import get_data
from config.logger_config import  sars_logger as logger
load_dotenv()
data_dir = os.getenv("DATA_DIR")
def main():
    logger.info("[SAGE: Multi Agent Self Evolution for LLM Reasoning]")
    logger.info("Importing the dataset: [GSM8K]")
    dataset =get_data(data_dir)


if __name__ == "__main__":
    main()