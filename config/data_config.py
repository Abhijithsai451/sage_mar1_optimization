from datasets import load_dataset
from config.logger_config import  sars_logger as logger
import os
def get_data(dir):
    if not os.path.exists(dir):
        logger.info("Data Directory does not exist. Creating it...")
        os.makedirs(dir)
        logger.info("Data Directory Created")
    logger.info(f"Loading the dataset to the path {dir} ")
    dataset = load_dataset(path ="openai/gsm8k",
                  name = "main",
                  cache_dir = dir
                  )
    logger.info(f"Dataset Loaded Successfully")
    return dataset
