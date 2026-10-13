import math

from unito_amazon.llm.base import BaseLLM
from unito_amazon.llm.schema import LLMGenerationConfig

from .prompts import CounterSpeechPairwisePrompt
from .schema import PairwiseJudgeResult, PairwiseWinner


class LLMJudge:
    """Pairwise JudgeLM evaluation of two anonymous system responses."""

    def __init__(
        self,
        llm: BaseLLM,
        prompt: CounterSpeechPairwisePrompt | None = None,
        max_retries: int = 5,
        retry_temperature: float = 0.7,
    ):
        if max_retries < 0:
            raise ValueError("max_retries cannot be negative.")
        if retry_temperature <= 0:
            raise ValueError("retry_temperature must be positive.")

        self.llm = llm
        self.prompt = prompt or CounterSpeechPairwisePrompt()
        self.max_retries = max_retries
        self.generation_config = LLMGenerationConfig(
            max_new_tokens=512,
            do_sample=False,
            repetition_penalty=1.0,
            system_prompt=self.prompt.system_prompt,
        )
        self.retry_generation_config = self.generation_config.model_copy(
            update={"do_sample": True, "temperature": retry_temperature}
        )

    def judge(self, comparison: dict) -> PairwiseJudgeResult:
        return self.judge_batch([comparison])[0]

    def judge_batch(self, comparisons: list[dict]) -> list[PairwiseJudgeResult]:
        prompts = [self._build_prompt(record) for record in comparisons]
        response = self.llm.generate_batch(
            prompts=prompts,
            config=self.generation_config,
        )
        return [
            self._parse_with_retries(comparison, raw_response)
            for comparison, raw_response in zip(
                comparisons,
                response.texts,
                strict=True,
            )
        ]

    def _build_prompt(self, comparison: dict) -> str:
        required = ("hateful_message", "response_1", "response_2")
        missing = [field for field in required if not comparison.get(field)]
        if missing:
            raise ValueError(
                "Pairwise judge input is missing fields: " + ", ".join(missing)
            )
        return self.prompt.build(
            hateful_message=comparison["hateful_message"],
            response_1=comparison["response_1"],
            response_2=comparison["response_2"],
        )

    def _parse_with_retries(
        self,
        comparison: dict,
        raw_response: str,
    ) -> PairwiseJudgeResult:
        attempts = 1
        current = raw_response
        while True:
            parsed = self.parse_judgment(current)
            if parsed is not None:
                winner, score_1, score_2 = parsed
                return PairwiseJudgeResult(
                    winner=winner,
                    response_1_score=score_1,
                    response_2_score=score_2,
                    reasoning=current.strip().partition("\n")[2].strip() or None,
                    raw_response=current,
                    attempts=attempts,
                )

            if attempts > self.max_retries:
                return PairwiseJudgeResult(
                    raw_response=current,
                    parse_error=(
                        "Expected exactly two numeric scores between 1 and 10 "
                        "on the first line of the JudgeLM response."
                    ),
                    attempts=attempts,
                )

            retry_response = self.llm.generate(
                prompt=self._build_prompt(comparison),
                config=self.retry_generation_config,
            )
            current = retry_response.text
            attempts += 1

    @classmethod
    def parse_judgment(
        cls,
        raw_response: str,
    ) -> tuple[PairwiseWinner, float | None, float | None] | None:
        text = raw_response.strip()
        if not text:
            return None
        return cls._parse_score_pair(text)

    @classmethod
    def _parse_judgment(
        cls,
        raw_response: str,
    ) -> tuple[PairwiseWinner, float | None, float | None] | None:
        """Backward-compatible alias for callers using the old private API."""
        return cls.parse_judgment(raw_response)

    @staticmethod
    def _parse_score_pair(
        text: str,
    ) -> tuple[PairwiseWinner, float | None, float | None] | None:
        """Follow JudgeLM parse_score: only the first line defines the scores."""
        first_line = text.split("\n", 1)[0].strip()
        values = first_line.replace(",", " ").split()
        if len(values) != 2:
            return None
        try:
            score_1, score_2 = (float(value) for value in values)
        except ValueError:
            return None
        if (
            not math.isfinite(score_1)
            or not math.isfinite(score_2)
            or not 1 <= score_1 <= 10
            or not 1 <= score_2 <= 10
        ):
            return None

        if score_1 > score_2:
            expected_winner: PairwiseWinner = "response_1"
        elif score_2 > score_1:
            expected_winner = "response_2"
        else:
            expected_winner = "tie"
        return expected_winner, score_1, score_2
