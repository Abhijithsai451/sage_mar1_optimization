from datasets import load_dataset
from dotenv import load_dotenv
from config.logger_config import  sars_logger as logger
import os
load_dotenv()
data_dir = os.getenv("DATA_DIR")

def get_data(split):
    if not os.path.exists(data_dir):
        logger.info("Data Directory does not exist. Creating it...")
        os.makedirs(data_dir)
        logger.info("Data Directory Created")
    logger.info(f"Loading the dataset to the path {dir} ")
    train_data = load_dataset(path ="openai/gsm8k",
                              name = "main",
                              cache_dir = data_dir,
                              split = "train[:70%]"
                              )
    test_data = load_dataset(path ="openai/gsm8k",
                             name = "main",
                             cache_dir = data_dir,
                             split = "train[70%:]"
                             )
    if split == "train":
        return train_data
    elif split == "test":
        return test_data
    return "Data Import Failed"