import os
from typing import List, Dict

from huggingface_hub import hf_hub_download
import torch
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from config.logger_config import sars_logger as logger
from states.agent_state import SAGEAgentState

load_dotenv()
model_name = os.getenv("MODEL_NAME")
model_dir = os.getenv("MODEL_DIR")

def download_model():
    logger.info("Downloading the model from HuggingFace Hub")
    hf_hub_download(repo_id=model_name, filename="config.json")
    # hf download TinyLlama/TinyLlama-1.1B-Chat-v1.0 --exclude "originals/*"--local-dir ./base_model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_4bit=True,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    model.requires_grad_(False)
    #return model, tokenizer

def load_model():
    if not os.path.exists(model_dir) or not os.listdir(model_dir):
        logger.info("Model Directory does not Exists")
        download_model()
    else:
        logger.info("Model Directory Exists")
    logger.info("Loading the model from the local directory")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir,
                                                 load_in_4bit=True,
                                                 dtype = torch.bfloat16,
                                                 device_map="auto")
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
    return lora_config

class BackboneModel:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            logger.info("Initializing the singleton Backbone Model: llama3:8b")
            cls._instance = super(BackboneModel, cls).__new__(cls)
            cls._instance.model,cls._instance.tokenizer = load_model()
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



"""class BackboneModel:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            logger.info("Initializing the singleton Backbone Model: llama3:8b")
            cls._instance = super(BackboneModel, cls).__new__(cls)
            cls._instance.model,cls._instance.ref_model, cls._instance.tokenizer = config_lora()
        return cls._instance

    def _invoke(self, prompt):
        #Helper to call the underlying LLM's invoke method
        return self.model.invoke(prompt)

    def bind_tools(self, tools, **kwargs):
        return self.model.bind_tools(tools, **kwargs)

    def _test(self, prompt):
        response = self.invoke(prompt)
        return response.content

    def structured_output(self):
        self.model = self.model.with_structured_output(SAGEAgentState)
        return self.model


def get_backbone():
    return BackboneModel()


class BackBone:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            logger.info("Initializing the singleton Backbone Model: llama3:8b")
            cls._instance = super(BackBone, cls).__new__(cls)
            cls._instance.model, cls._instance.tokenizer = load_model()
            cls._instance.device = torch.device("mps" if torch.cuda.is_available() else "cpu")
            cls._instance.model.to(cls._instance.device)

        if cls._instance.tokenizer.pad_token is None:
            cls._instance.tokenizer.pad_token = cls._instance.tokenizer.eos_token
        return cls._instance

    def invoke(self, messages: List[Dict[str, str]])-> AIMessage:
        # Helper to call the underlying LLM's invoke method
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        inputs = self.tokenizer(prompt, return_tensors = "pt", padding=True, truncation = True).to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=50000,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id
            )
        response_tokens = outputs[0, inputs["input_ids"].shape[1]:]
        response = self.tokenizer.decode(response_tokens, skip_special_tokens=True)
        return AIMessage(content=response)
    def bind_tools(self, tools, **kwargs):
        return self.model.bind_tools(tools, **kwargs)

    def test_model(self, prompt: str)->str:
        response = self.invoke([{
                "role": "user",
                "content": prompt}])
        return response.content

    def structured_output(self):
        self.model = self.model.with_structured_output(SAGEAgentState)
        return self.model


def get_basemodel():
 return BackBone()
"""