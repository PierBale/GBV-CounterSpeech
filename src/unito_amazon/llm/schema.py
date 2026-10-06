from typing import Any

from pydantic import BaseModel, Field


class LLMGenerationConfig(BaseModel):
    max_new_tokens: int = 512

    do_sample: bool = False
    temperature: float | None = None
    top_p: float | None = None
    repetition_penalty: float = 1.0

    assistant_prefix: str = ""
    system_prompt: str = ""


class LLMResponse(BaseModel):
    text: str
    model_name: str
    provider: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class BatchLLMResponse(BaseModel):
    texts: list[str]
    model_name: str
    provider: str
    metadata: dict[str, Any] = Field(default_factory=dict)