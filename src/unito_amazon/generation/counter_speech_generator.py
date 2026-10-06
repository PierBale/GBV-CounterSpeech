from unito_amazon.llm.base import BaseLLM
from unito_amazon.llm.schema import LLMGenerationConfig

from .prompts import CounterSpeechPrompt
from .schema import GenerationResult, parse_counter_narrative


class CounterSpeechGenerator:
    def __init__(
        self,
        llm: BaseLLM,
        prompt: CounterSpeechPrompt | None = None,
        generation_config: LLMGenerationConfig | None = None,
    ):
        self.llm = llm
        self.prompt = prompt or CounterSpeechPrompt()
        base_config = generation_config or LLMGenerationConfig(
            max_new_tokens=128,
            do_sample=False,
            repetition_penalty=1.0,
        )
        self.generation_config = base_config.model_copy(
            update={"system_prompt": self.prompt.system_prompt}
        )

    def generate(self, hateful_message: str) -> GenerationResult:
        return self.generate_batch([hateful_message])[0]

    def generate_batch(
        self,
        hateful_messages: list[str],
    ) -> list[GenerationResult]:
        prompts = [self.prompt.build(message) for message in hateful_messages]
        response = self.llm.generate_batch(
            prompts=prompts,
            config=self.generation_config,
        )

        results = []
        for message, prompt, raw_response in zip(
            hateful_messages,
            prompts,
            response.texts,
            strict=True,
        ):
            counter_speech, response_format_valid = parse_counter_narrative(
                raw_response
            )
            results.append(
                GenerationResult(
                    hateful_message=message,
                    counter_speech=counter_speech,
                    system_prompt=self.prompt.system_prompt,
                    prompt=prompt,
                    metadata={
                        "llm_model_name": response.model_name,
                        "llm_provider": response.provider,
                        "generation_strategy": "without_rag",
                        "raw_response": raw_response,
                        "response_format_valid": response_format_valid,
                    },
                )
            )
        return results
