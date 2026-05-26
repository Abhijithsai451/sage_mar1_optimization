from datasets import load_dataset
from config.logger_config import  sars_logger as logger
import os
def get_data(dir):
    if not os.path.exists(dir):
        logger.info("Data Directory does not exist. Creating it...")
        os.makedirs(dir)
        logger.info("Data Directory Created")
    logger.info(f"Loading the dataset to the path {dir} ")
    train_data = load_dataset(path ="openai/gsm8k",
                              name = "main",
                              cache_dir = dir,
                              split = "train[:70%]"
                              )
    test_data = load_dataset(path ="openai/gsm8k",
                             name = "main",
                             cache_dir = dir,
                             split = "train[70%:]"
                             )
    return train_data, test_data
