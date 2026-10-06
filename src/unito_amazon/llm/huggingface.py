from typing import Any

import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

from unito_amazon.llm.base import BaseLLM
from unito_amazon.llm.schema import LLMGenerationConfig, LLMResponse, BatchLLMResponse


def _is_gemma4(model_name: str) -> bool:
    n = model_name.lower()
    return "gemma-4" in n or "gemma4" in n


def _is_mistral3(model_name: str) -> bool:
    n = model_name.lower()
    return "ministral" in n or "mistral-3" in n or "mistral3" in n


def _is_qwen35(model_name: str) -> bool:
    n = model_name.lower()
    return "qwen3.5" in n or "qwen-3.5" in n or "qwen3_5" in n


def _uses_multimodal_processor(model_name: str) -> bool:
    return _is_qwen35(model_name) or _is_mistral3(model_name) or _is_gemma4(model_name)


class HuggingFaceLLM(BaseLLM):
    def __init__(
        self,
        model_name: str,
        device: str | None = None,
        dtype: torch.dtype | None = None,
        trust_remote_code: bool = True,
        model_kwargs: dict | None = None,
        tokenizer_kwargs: dict | None = None,
    ):
        self.model_name = model_name
        self.provider = "huggingface"
        self.trust_remote_code = trust_remote_code

        load_dotenv()

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype or self._get_default_dtype()

        self.model_kwargs = model_kwargs or {}
        self.tokenizer_kwargs = tokenizer_kwargs or {}

        print(f"[HuggingFaceLLM] Caricamento modello: {self.model_name}")
        print(f"[HuggingFaceLLM] CUDA disponibile: {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            print(f"[HuggingFaceLLM] GPU: {torch.cuda.get_device_name(0)}")

        print(f"[HuggingFaceLLM] dtype: {self.dtype} | device: {self.device}")

        self.tokenizer = self._load_tokenizer()
        self.model = self._load_model()

        print("[HuggingFaceLLM] Pronto.\n")

    def _get_default_dtype(self) -> torch.dtype:
        if torch.cuda.is_available():
            return torch.float16

        return torch.float32

    def generate(
        self,
        prompt: str,
        config: LLMGenerationConfig | None = None,
    ) -> LLMResponse:
        config = config or LLMGenerationConfig()

        batch_response = self.generate_batch(
            prompts=[prompt],
            config=config,
        )

        return LLMResponse(
            text=batch_response.texts[0],
            model_name=self.model_name,
            provider=self.provider,
            metadata=batch_response.metadata,
        )

    def generate_batch(
        self,
        prompts: list[str],
        config: LLMGenerationConfig | None = None,
    ) -> BatchLLMResponse:
        config = config or LLMGenerationConfig()

        formatted_prompts = [
            self._apply_chat_template(
                prompt=prompt,
                assistant_prefix=config.assistant_prefix,
                system_prompt=config.system_prompt,
            )
            for prompt in prompts
        ]

        inputs = self.tokenizer(
            text=formatted_prompts,
            return_tensors="pt",
            add_special_tokens=False,
            padding=True,
            truncation=True,
        ).to(self.model.device)

        generation_kwargs = self._build_generation_kwargs(config)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                **generation_kwargs,
            )

        input_length = inputs["input_ids"].shape[1]

        text_tokenizer = self._text_tokenizer()
        texts = [
            text_tokenizer.decode(
                outputs[i][input_length:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            ).strip()
            for i in range(len(prompts))
        ]

        return BatchLLMResponse(
            texts=texts,
            model_name=self.model_name,
            provider=self.provider,
            metadata={
                "generation_config": config.model_dump(),
                "formatted_prompts": formatted_prompts,
            },
        )

    def destroy(self) -> None:
        print(f"[HuggingFaceLLM] Liberazione modello {self.model_name}...")

        if hasattr(self, "model"):
            del self.model

        if hasattr(self, "tokenizer"):
            del self.tokenizer

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("[HuggingFaceLLM] Fatto.")

    def _load_tokenizer(self):
        if _uses_multimodal_processor(self.model_name):
            try:
                from transformers import AutoProcessor

                tokenizer = AutoProcessor.from_pretrained(
                    self.model_name,
                    trust_remote_code=self.trust_remote_code,
                    **self.tokenizer_kwargs,
                )
                print("[HuggingFaceLLM] Tokenizer: AutoProcessor")
            except (ValueError, OSError, ImportError):
                print("[HuggingFaceLLM] AutoProcessor non supportato, fallback su AutoTokenizer.")
                tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    trust_remote_code=self.trust_remote_code,
                    fix_mistral_regex=True,
                    **self.tokenizer_kwargs,
                )
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=self.trust_remote_code,
                **self.tokenizer_kwargs,
            )

        text_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)

        if getattr(text_tokenizer, "pad_token", None) is None:
            text_tokenizer.pad_token = text_tokenizer.eos_token
            text_tokenizer.pad_token_id = text_tokenizer.eos_token_id

        text_tokenizer.padding_side = "left"

        return tokenizer

    def _load_model(self):
        common_kwargs = {
            "dtype": self.dtype,
            "trust_remote_code": self.trust_remote_code,
            **self.model_kwargs,
        }

        if torch.cuda.is_available():
            common_kwargs["device_map"] = "auto"

        if _is_qwen35(self.model_name):
            try:
                from transformers import AutoModelForMultimodalLM
            except ImportError as error:
                raise RuntimeError(
                    "Qwen3.5 requires a Transformers version that provides "
                    "AutoModelForMultimodalLM."
                ) from error

            model = AutoModelForMultimodalLM.from_pretrained(
                self.model_name,
                **common_kwargs,
            )
            print("[HuggingFaceLLM] Model: AutoModelForMultimodalLM")
            return self._finalize_model(model)

        if _is_mistral3(self.model_name):
            try:
                from transformers import Mistral3ForConditionalGeneration, FineGrainedFP8Config

                model = Mistral3ForConditionalGeneration.from_pretrained(
                    self.model_name,
                    quantization_config=FineGrainedFP8Config(dequantize=True),
                    **common_kwargs,
                )
                print("[HuggingFaceLLM] Model: Mistral3ForConditionalGeneration")
                return self._finalize_model(model)
            except ImportError as error:
                raise RuntimeError(
                    "Ministral 3 requires Mistral3ForConditionalGeneration and "
                    "FineGrainedFP8Config in Transformers."
                ) from error

        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **common_kwargs,
        )

        print("[HuggingFaceLLM] Model: AutoModelForCausalLM")
        return self._finalize_model(model)

    def _finalize_model(self, model):
        if not torch.cuda.is_available():
            model = model.to(self.device)
        model.eval()
        return model

    def _text_tokenizer(self):
        return getattr(self.tokenizer, "tokenizer", self.tokenizer)

    def _apply_chat_template(
        self,
        prompt: str,
        assistant_prefix: str = "",
        system_prompt: str = "",
    ) -> str:
        messages = []

        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt,
            })

        messages.append({
            "role": "user",
            "content": self._message_content(prompt),
        })

        kwargs = {
            "tokenize": False,
            "add_generation_prompt": not bool(assistant_prefix),
            "continue_final_message": bool(assistant_prefix),
        }

        if assistant_prefix:
            messages.append({
                "role": "assistant",
                "content": self._message_content(assistant_prefix),
            })

        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                return self.tokenizer.apply_chat_template(
                    messages,
                    **kwargs,
                    enable_thinking=False,
                )
            except TypeError:
                return self.tokenizer.apply_chat_template(
                    messages,
                    **kwargs,
                )

        return self._fallback_prompt_template(
            prompt=prompt,
            system_prompt=system_prompt,
            assistant_prefix=assistant_prefix,
        )

    def _message_content(self, text: str):
        if _uses_multimodal_processor(self.model_name):
            return [{"type": "text", "text": text}]
        return text

    def _fallback_prompt_template(
        self,
        prompt: str,
        system_prompt: str = "",
        assistant_prefix: str = "",
    ) -> str:
        parts = []

        if system_prompt:
            parts.append(f"System: {system_prompt}")

        parts.append(f"User: {prompt}")
        parts.append(f"Assistant: {assistant_prefix}")

        return "\n".join(parts)

    def _build_generation_kwargs(self, config: LLMGenerationConfig) -> dict[str, Any]:
        text_tokenizer = self._text_tokenizer()
        kwargs = {
            "max_new_tokens": config.max_new_tokens,
            "pad_token_id": text_tokenizer.pad_token_id,
            "eos_token_id": text_tokenizer.eos_token_id,
            "do_sample": config.do_sample,
            "repetition_penalty": config.repetition_penalty,
        }

        if config.do_sample:
            if config.temperature is not None:
                kwargs["temperature"] = config.temperature

            if config.top_p is not None:
                kwargs["top_p"] = config.top_p

        return kwargs
