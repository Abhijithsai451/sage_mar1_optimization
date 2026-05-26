import transformers

training_args = transformers.TrainingArguments(
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
    num_train_epochs=4
    )


class SAGETrainer:
    def __init__(self, peft_model, train_dataset, eval_dataset, agent_graph ,args=training_args):
        self.trainer = transformers.Trainer(
            model=peft_model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            args=args
        )
        self.model = peft_model

        self.graph = agent_graph



