from unito_amazon.llm.base import BaseLLM
from unito_amazon.llm.schema import LLMGenerationConfig
from unito_amazon.retrieval.base import BaseRetriever
from unito_amazon.retrieval.schema import RetrievalResult

from .prompts import RAGCounterSpeechPrompt
from .schema import GenerationResult, parse_counter_narrative


class RAGGenerator:
    def __init__(
        self,
        llm: BaseLLM,
        retriever: BaseRetriever,
        prompt: RAGCounterSpeechPrompt | None = None,
        generation_config: LLMGenerationConfig | None = None,
        top_k: int = 3,
    ):
        if top_k != RAGCounterSpeechPrompt.required_evidence_count:
            raise ValueError("The RAG prompt requires top_k=3.")

        self.llm = llm
        self.retriever = retriever
        self.prompt = prompt or RAGCounterSpeechPrompt()
        base_config = generation_config or LLMGenerationConfig(
            max_new_tokens=128,
            do_sample=False,
            repetition_penalty=1.0,
        )
        self.generation_config = base_config.model_copy(
            update={"system_prompt": self.prompt.system_prompt}
        )
        self.top_k = top_k

    def generate(self, hateful_message: str) -> GenerationResult:
        return self.generate_batch([hateful_message])[0]

    def generate_batch(
        self,
        hateful_messages: list[str],
    ) -> list[GenerationResult]:
        retrieval_batches = [
            self.retriever.retrieve(message, top_k=self.top_k)
            for message in hateful_messages
        ]
        prompts = [
            self.prompt.build(message, retrieval_results)
            for message, retrieval_results in zip(
                hateful_messages,
                retrieval_batches,
                strict=True,
            )
        ]
        response = self.llm.generate_batch(
            prompts=prompts,
            config=self.generation_config,
        )

        return [
            self._build_result(
                hateful_message=message,
                prompt=prompt,
                raw_response=counter_speech,
                retrieval_results=retrieval_results,
                model_name=response.model_name,
                provider=response.provider,
            )
            for message, prompt, counter_speech, retrieval_results in zip(
                hateful_messages,
                prompts,
                response.texts,
                retrieval_batches,
                strict=True,
            )
        ]

    def build_prompt(
        self,
        hateful_message: str,
        retrieval_results: list[RetrievalResult],
    ) -> str:
        return self.prompt.build(hateful_message, retrieval_results)

    def _build_result(
        self,
        hateful_message: str,
        prompt: str,
        raw_response: str,
        retrieval_results: list[RetrievalResult],
        model_name: str,
        provider: str,
    ) -> GenerationResult:
        counter_speech, response_format_valid = parse_counter_narrative(
            raw_response
        )
        return GenerationResult(
            hateful_message=hateful_message,
            counter_speech=counter_speech,
            system_prompt=self.prompt.system_prompt,
            prompt=prompt,
            context=[result.document.text for result in retrieval_results],
            retrieval_results=retrieval_results,
            metadata={
                "llm_model_name": model_name,
                "llm_provider": provider,
                "generation_strategy": "rag",
                "retrieval_method": retrieval_results[0].retriever_name,
                "raw_response": raw_response,
                "response_format_valid": response_format_valid,
            },
        )
