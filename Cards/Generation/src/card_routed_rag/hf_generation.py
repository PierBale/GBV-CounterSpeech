from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class HuggingFaceModelSpec:
    alias: str
    repo_id: str
    backend: str
    disable_thinking: bool = False
    load_in_4bit: bool = True


class HuggingFaceChatGenerator:
    """Local Hugging Face chat generation with sequential model loading."""

    def __init__(
        self,
        spec: HuggingFaceModelSpec,
        *,
        quantization: dict[str, Any] | None = None,
        max_memory: dict[Any, str] | None = None,
    ) -> None:
        import torch
        from transformers import BitsAndBytesConfig

        self.spec = spec
        self.torch = torch
        self.backend = spec.backend
        self.tokenizer: Any = None
        self.model: Any = None

        model_kwargs: dict[str, Any] = {
            "device_map": "auto",
            "low_cpu_mem_usage": True,
        }
        if max_memory:
            model_kwargs["max_memory"] = max_memory
        if spec.load_in_4bit:
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "4-bit bitsandbytes loading requires a CUDA GPU. "
                    "Disable load_in_4bit only on a machine with enough RAM/VRAM."
                )
            qcfg = quantization or {}
            compute_dtype_name = str(
                qcfg.get("bnb_4bit_compute_dtype", "bfloat16")
            )
            compute_dtype = getattr(torch, compute_dtype_name)
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=str(
                    qcfg.get("bnb_4bit_quant_type", "nf4")
                ),
                bnb_4bit_use_double_quant=bool(
                    qcfg.get("bnb_4bit_use_double_quant", True)
                ),
                bnb_4bit_compute_dtype=compute_dtype,
            )
        else:
            model_kwargs["torch_dtype"] = "auto"

        if self.backend == "causal_lm":
            self._load_causal_lm(model_kwargs)
        elif self.backend == "multimodal_auto":
            self._load_multimodal_auto(model_kwargs)
        elif self.backend == "mistral3":
            self._load_mistral3(model_kwargs)
        else:
            raise ValueError(
                f"Unsupported backend {self.backend!r} for {spec.alias}."
            )

    def _load_causal_lm(self, model_kwargs: dict[str, Any]) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(self.spec.repo_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.spec.repo_id,
            **model_kwargs,
        )

    def _load_multimodal_auto(self, model_kwargs: dict[str, Any]) -> None:
        from transformers import AutoModelForMultimodalLM, AutoProcessor

        self.tokenizer = AutoProcessor.from_pretrained(self.spec.repo_id)
        self.model = AutoModelForMultimodalLM.from_pretrained(
            self.spec.repo_id,
            **model_kwargs,
        )

    def _load_mistral3(self, model_kwargs: dict[str, Any]) -> None:
        from transformers import AutoTokenizer, Mistral3ForConditionalGeneration

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.spec.repo_id,
            fix_mistral_regex=True,
        )
        self.model = Mistral3ForConditionalGeneration.from_pretrained(
            self.spec.repo_id,
            **model_kwargs,
        )

    @property
    def input_device(self) -> Any:
        try:
            return self.model.device
        except Exception:
            return self.torch.device("cuda:0")

    def _messages(self, system_prompt: str, user_prompt: str) -> list[dict[str, Any]]:
        if self.backend == "multimodal_auto":
            return [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": system_prompt}],
                },
                {
                    "role": "user",
                    "content": [{"type": "text", "text": user_prompt}],
                },
            ]
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def generate(
        self,
        user_prompt: str,
        *,
        system_prompt: str,
        max_new_tokens: int,
        temperature: float,
    ) -> str:
        messages = self._messages(system_prompt, user_prompt)
        template_kwargs: dict[str, Any] = {
            "add_generation_prompt": True,
            "tokenize": True,
            "return_dict": True,
            "return_tensors": "pt",
        }
        if self.spec.disable_thinking:
            template_kwargs["enable_thinking"] = False
        inputs = self.tokenizer.apply_chat_template(messages, **template_kwargs)
        inputs = {
            key: value.to(self.input_device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }
        input_length = int(inputs["input_ids"].shape[-1])
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
        }
        if temperature > 0:
            generation_kwargs["temperature"] = temperature

        with self.torch.inference_mode():
            output = self.model.generate(**inputs, **generation_kwargs)[0]
        generated = output[input_length:]
        return self.tokenizer.decode(
            generated,
            skip_special_tokens=True,
        ).strip()

    def close(self) -> None:
        self.model = None
        self.tokenizer = None
        gc.collect()
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()

    def __enter__(self) -> "HuggingFaceChatGenerator":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()
