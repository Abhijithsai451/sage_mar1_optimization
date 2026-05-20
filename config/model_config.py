from dotenv import load_dotenv
from langchain_ollama import ChatOllama

from config.logger_config import sars_logger as logger
from states.agent_state import SAGEAgentState

load_dotenv()


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
        """Helper to call the underlying LLM's invoke method"""
        return self.model.invoke(prompt)

    def bind_tools(self, tools, **kwargs):
        """Delegate bind_tools to the underlying LangChain model"""
        return self.model.bind_tools(tools, **kwargs)

    def test_model(self, prompt):
        """Your custom test method"""
        response = self.invoke(prompt)
        return response.content

    def structured_output(self):
        """returns the model with structured output """
        self.model = self.model.with_structured_output(SAGEAgentState)
        return self.model


def get_backbone():
    return BackboneModel()
