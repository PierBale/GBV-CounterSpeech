from abc import ABC, abstractmethod
from unito_amazon.llm.schema import LLMGenerationConfig, LLMResponse, BatchLLMResponse


class BaseLLM(ABC):
    @abstractmethod
    def generate(
        self,
        prompt: str,
        config: LLMGenerationConfig | None = None,
    ) -> LLMResponse:
        raise NotImplementedError

    @abstractmethod
    def generate_batch(
        self,
        prompts: list[str],
        config: LLMGenerationConfig | None = None,
    ) -> BatchLLMResponse:
        raise NotImplementedError

    def destroy(self) -> None:
        pass