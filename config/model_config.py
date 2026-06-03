import os
from huggingface_hub import hf_hub_download
import torch
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from config.logger_config import sars_logger as logger
from states.agent_state import SAGEAgentState
from langchain_openai import ChatOpenAI
load_dotenv()
model_name = os.getenv("MODEL_NAME")
model_dir = os.getenv("MODEL_DIR")
model_url = os.getenv("MODEL_URL")
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

def download_model():
    logger.info("Downloading the model from HuggingFace Hub")
    hf_hub_download(repo_id=model_name, filename="config.json")
    # hf download meta-llama/Llama-3.2-3B-Instruct --exclude "originals/*" --local-dir ./base_model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    logger.info(f"Loading base model onto device: [{DEVICE}] using float16 precision")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map=None,
    ).to(DEVICE)
    return model, tokenizer

def load_model():
    if not os.path.exists(model_dir) or not os.listdir(model_dir):
        logger.info("Model Directory does not Exists")
        return download_model()
    else:
        logger.info("Model Directory Exists")
    logger.info("Loading the model from the local directory")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    logger.info(f"Loading local base model onto device: [{DEVICE}] using float16 precision")
    model = AutoModelForCausalLM.from_pretrained(model_dir,
                                                 torch_dtype = torch.bfloat16,
                                                 device_map=None).to(DEVICE)
    return model, tokenizer

def config_lora():
    lora_config = LoraConfig(
        r = 8,
        lora_alpha = 16,
        target_modules = ["q_proj","k_proj","v_proj","o_proj"],
        lora_dropout = 0.05,
        bias = "none",
        task_type = "CAUSAL_LM",
    )
    return lora_config

class LocalConfigurableModelWrapper:
    """
    Acts as a proxy for .with_config() calls inside LangGraph nodes.
    Swaps teh active adapter before running inference.
    """
    def __init__(self, backbone_instance, lora_name:str):
        self.backbone = backbone_instance
        self.lora_name = lora_name
    def invoke(self,messages: list)-> AIMessage:
        self.backbone.set_active_adapter(self.lora_name)
        return self.backbone._local_invoke(messages)

class BackboneModel:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            logger.info("Initializing the singleton Backbone Model: llama3:8b")
            cls._instance = super(BackboneModel, cls).__new__(cls)

            base_model, cls._instance.tokenizer = load_model()
            if cls._instance.tokenizer.pad_token is None:
                cls._instance.tokenizer.pad_token = cls._instance.tokenizer.eos_token
            lora_config = config_lora()

            adapter_paths = {
                "challenger": "./adapters/challenger_lora",
                "planner": "./adapters/planner_lora",
                "solver": "./adapters/solver_lora",
                "critic": "./adapters/critic_lora",
            }
            if os.path.exists(adapter_paths["challenger"]) and os.listdir(adapter_paths["challenger"]) :
                logger.info("Loading existing challenger adapters weights from disk....")
                cls._instance.model = get_peft_model(base_model, lora_config, adapter_paths["challenger"])
                cls._instance.model.load_adapter(adapter_paths["challenger"], adapter_name="challenger")
            else:
                logger.info("Initializing fresh challenger adapter from config....")
                cls._instance.model = get_peft_model(base_model, lora_config, adapter_name = "challenger")

            for name in ["planner", "solver", "critic"]:
                path = adapter_paths[name]
                if os.path.exists(path) and os.listdir(path):
                    logger.info(f"Loading existing {name} adapters weights from disk....")
                    cls._instance.model.load_adapter(path, adapter_name=name)
                else:
                    logger.info(f"Adding empty {name} adapters structure  to backbone....")
                    cls._instance.model.add_adapter(name, lora_config)
            logger.info("All PEFT Adapters initialized successfully")
        return cls._instance

    def set_active_adapter(self, lora_name: str):
        """Swaps active attention weights instantly in GPU memory."""
        if lora_name in self.model.peft_config:
            self.model.set_adapter(lora_name)
        else:
            logger.warning(f"Adapter [{lora_name}] not registered. Sticking with current active configuration.")

    def _messages_to_dict(self, messages: list) -> list:
        hf_chat = []
        for msg in messages:
            if isinstance(msg, SystemMessage) or (hasattr(msg, 'type') and msg.type == "system"):
                role = "system"
            elif isinstance(msg, HumanMessage) or (hasattr(msg, 'type') and msg.type == "human"):
                role = "user"
            elif hasattr(msg, 'type') and msg.type == "ai":
                role = "assistant"
            else:
                role = "user"
            hf_chat.append({"role": role, "content": msg.content})
        return hf_chat

    def format_prompt(self, messages: list) -> str:
        chat_format = self._messages_to_dict(messages)
        return self.tokenizer.apply_chat_template(chat_format, tokenize=False, add_generation_prompt=True)

    def _local_invoke(self, messages: list) -> AIMessage:
        prompt_text = self.format_prompt(messages)
        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id
            )

        prompt_len = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][prompt_len:]
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        return AIMessage(content=generated_text)

    def with_config(self, configurable: dict):
        """Intercepts and routes via the proxy wrapper to keep agent functions intact."""
        config_inner = configurable.get("configurable", configurable)
        lora_name = config_inner.get("model", "challenger")
        return LocalConfigurableModelWrapper(self, lora_name)

    def invoke(self, prompt):
        if isinstance(prompt, str):
            prompt = [HumanMessage(content=prompt)]
        return self._local_invoke(prompt)

    def bind_tools(self, tools, **kwargs):
        raise NotImplementedError("Tool binding is not natively supported in this local model configuration.")

    def parameters(self):
        return self.model.parameters()

    def test_model(self, prompt):
        response = self.invoke(prompt)
        return response.content

    def structured_output(self):
        return self

def get_backbone():
    return BackboneModel()



