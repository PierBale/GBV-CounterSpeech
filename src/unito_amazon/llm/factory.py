from typing import Any

import torch

from unito_amazon.llm.base import BaseLLM
from unito_amazon.llm.huggingface import HuggingFaceLLM


class LLMFactory:
    @staticmethod
    def create(config: dict[str, Any]) -> BaseLLM:
        provider = config.get("provider")

        if provider == "huggingface":
            dtype = LLMFactory._parse_dtype(config.get("dtype"))

            return HuggingFaceLLM(
                model_name=config["model_name"],
                device=config.get("device"),
                dtype=dtype,
                trust_remote_code=config.get("trust_remote_code", True),
                model_kwargs=config.get("model_kwargs"),
                tokenizer_kwargs=config.get("tokenizer_kwargs"),
            )

        raise ValueError(f"Provider LLM non supportato: {provider}")

    @staticmethod
    def _parse_dtype(dtype: str | None):
        if dtype is None:
            return None

        mapping = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }

        if dtype not in mapping:
            raise ValueError(f"dtype non supportato: {dtype}")

        return mapping[dtype]