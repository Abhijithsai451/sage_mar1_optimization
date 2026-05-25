import os
from huggingface_hub import hf_hub_download
import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model

load_dotenv()
model_name = os.getenv("MODEL_NAME")

"""
class BackboneModel:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            logger.info("Initializing the singleton Backbone Model: llama3:8b")
            cls._instance = super(BackboneModel, cls).__new__(cls)
            cls._instance.model = ChatOllama(
                model="llama3.1:8b",
                temperature=0,
                base_url="http://localhost:11434"
            )
        return cls._instance

    def invoke(self, prompt):
        #Helper to call the underlying LLM's invoke method
        return self.model.invoke(prompt)

    def bind_tools(self, tools, **kwargs):
        return self.model.bind_tools(tools, **kwargs)

    def test_model(self, prompt):
        response = self.invoke(prompt)
        return response.content

    def structured_output(self):
        self.model = self.model.with_structured_output(SAGEAgentState)
        return self.model


def get_backbone():
    return BackboneModel()
"""

def build_model():
    hf_hub_download(repo_id=model_name, filename="config.json")
    # huggingface-cli download meta-llama/Llama-3.1-8B-Instruct --local-dir ./model_source --local-dir-use-symlinks False
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_4bit=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    return model, tokenizer

def config_lora():
    lora_config = LoraConfig(
        r = 8,
        lora_alpha = 16,
        target_modules = ["q_proj","k_project","v_proj","o_proj"],
        lora_dropout = 0.05,
        bias = "none",
        task_type = "CAUSAL_LM",
    )
    model, tokenizer = build_model()
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer

if __name__ == "__main__":
    model, tokenizer = config_lora()
    print(model)

