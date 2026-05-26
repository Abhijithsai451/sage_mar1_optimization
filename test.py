from config.database_utils import clear_db, init_database
from config.model_config import config_lora

if __name__ == "__main__":
    init_database()
    clear_db()
    model, tokenizer = config_lora()
    print(model)