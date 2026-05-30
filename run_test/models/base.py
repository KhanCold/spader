from abc import ABC, abstractmethod
import time
import logging

logger = logging.getLogger(__name__)

class BaseModel(ABC):
    def __init__(self, model_name: str, api_base: str = None, api_key: str = None, system_prompt: str = None):
        self.model_name = model_name
        self.api_base = api_base
        self.api_key = api_key
        self.system_prompt = system_prompt

    @abstractmethod
    def generate(self, messages: list[dict], tools: list[dict] = None, stop: list[str] = None, max_tokens: int = 5096, temperature: float = 1.0) -> str:
        """
        Generate text based on messages.
        """
        pass

    def get_system_prompt(self) -> str:
        return self.system_prompt

    def retry_generate(self, messages: list[dict], k: int = 3, **kwargs) -> str:
        """
        Retry generation k times upon failure.
        """
        for i in range(k):
            try:
                return self.generate(messages, **kwargs)
            except Exception as e:
                logger.warning(f"Generation attempt {i+1} failed: {e}")
                if i == k - 1:
                    raise e
                time.sleep(1)
