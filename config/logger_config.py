import logging
import os
from datetime import datetime

def setup_logger(name="SAGE_RL", log_level = logging.INFO):
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # 1. Console Handler (Pretty colors or simple text)
    c_handler = logging.StreamHandler()
    c_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    c_handler.setFormatter(c_format)
    logger.addHandler(c_handler)

    # 2. File Handler (Detailed for debugging)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    f_handler = logging.FileHandler(f"{log_dir}/session_{timestamp}.log")
    f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    f_handler.setFormatter(f_format)
    logger.addHandler(f_handler)


    return logger

# Initialize a global instance
sars_logger = setup_logger()