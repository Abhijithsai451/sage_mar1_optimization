from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from config.logger_config import sars_logger as logger
load_dotenv()

class BackboneModel:
    _instance = None
    _model = None
    # Here we define the model and ensure we only create one instance of the model so as to be shared among the agents    def __new__(cls):
    def __new__(cls):
        if cls._instance is None:
            logger.info("Initializing the singleton Backbone Model: llama3:8b")
            cls._instance = super(BackboneModel, cls).__new__(cls)

            cls.model = ChatOllama(
                model = "llama3:8b",
                temperature = 0,
                base_url = "http://localhost:11434",
                num_predict = 256
            )
        return cls._instance
    def invoke(self, prompt):
        """Helper to call the underlying LLM's invoke method"""
        return self.model.invoke(prompt)

    def test_model(self, prompt):
        """Your custom test method"""
        response = self.invoke(prompt)
        return response.content

def get_backbone():
    return BackboneModel()

