import os

from dotenv import load_dotenv
from transformers import TrainingArguments
from trl import SFTTrainer

from config.data_config import get_data
from config.model_config import get_backbone, config_lora

load_dotenv()
save_dir = os.getenv("SAVED_DIR")
training_args = TrainingArguments(
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    warmup_steps=2,
    max_steps=10,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=1,
    output_dir="lora_output",
    optim="paged_adamw_8bit",
    report_to="none",
    num_train_epochs=4,
    remove_unused_columns=False,
    )
def train_model():
    trainer = SFTTrainer(
        model = get_backbone().model,
        train_dataset = get_data("train"),
        peft_config = config_lora(),
        dataset_text_field = "text",
        tokenizer = get_backbone().tokenizer,
        args = training_args
    )
    trainer.train()
    trainer.model.save_pretrained("lora_output")

    return "Model Trained Successfully"

