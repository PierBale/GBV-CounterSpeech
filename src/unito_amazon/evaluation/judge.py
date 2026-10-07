import json
import re

from unito_amazon.llm.base import BaseLLM
from unito_amazon.llm.schema import LLMGenerationConfig

from .prompts import CounterSpeechPairwisePrompt
from .schema import PairwiseJudgeResult, PairwiseWinner


class LLMJudge:
    """Pairwise JudgeLM evaluation for RAG against No-RAG."""

    def __init__(
        self,
        llm: BaseLLM,
        prompt: CounterSpeechPairwisePrompt | None = None,
        max_retries: int = 1,
    ):
        if max_retries < 0:
            raise ValueError("max_retries cannot be negative.")

        self.llm = llm
        self.prompt = prompt or CounterSpeechPairwisePrompt()
        self.max_retries = max_retries
        self.generation_config = LLMGenerationConfig(
            max_new_tokens=512,
            do_sample=False,
            repetition_penalty=1.0,
            system_prompt=self.prompt.system_prompt,
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
            parsed = self._parse_judgment(current)
            if parsed is not None:
                winner, score_1, score_2 = parsed
                return PairwiseJudgeResult(
                    winner=winner,
                    response_1_score=score_1,
                    response_2_score=score_2,
                    raw_response=current,
                    attempts=attempts,
                )

            if attempts > self.max_retries:
                return PairwiseJudgeResult(
                    raw_response=current,
                    parse_error=(
                        "Could not identify a winner or a pair of scores in "
                        "the JudgeLM response."
                    ),
                    attempts=attempts,
                )

            repaired = self.llm.generate(
                prompt=self._build_prompt(comparison),
                config=self.generation_config,
            )
            current = repaired.text
            attempts += 1

    @classmethod
    def _parse_judgment(
        cls,
        raw_response: str,
    ) -> tuple[PairwiseWinner, float | None, float | None] | None:
        text = raw_response.strip()
        if not text:
            return None

        json_result = cls._parse_json(text)
        if json_result is not None:
            return json_result

        score_result = cls._parse_score_pair(text)
        if score_result is not None:
            return score_result

        normalized = re.sub(r"\s+", " ", text.casefold())
        tie_patterns = (
            r"\[\[c\]\]",
            r"\b(?:tie|draw)\b",
            r"\b(?:equally good|equal in quality|responses? (?:are|is) equal)\b",
        )
        if any(re.search(pattern, normalized) for pattern in tie_patterns):
            return "tie", None, None

        response_1_patterns = (
            r"\[\[a\]\]",
            r"\bresponse\s*(?:1|one|a)\s+is\s+(?:the\s+)?better\b",
            r"\bprefer\s+(?:counter-?speech\s+)?response\s*(?:1|one|a)\b",
            r"\b(?:winner|choice|verdict)\s*[:=-]?\s*(?:response\s*)?(?:1|one|a)\b",
        )
        response_2_patterns = (
            r"\[\[b\]\]",
            r"\bresponse\s*(?:2|two|b)\s+is\s+(?:the\s+)?better\b",
            r"\bprefer\s+(?:counter-?speech\s+)?response\s*(?:2|two|b)\b",
            r"\b(?:winner|choice|verdict)\s*[:=-]?\s*(?:response\s*)?(?:2|two|b)\b",
        )
        has_1 = any(re.search(pattern, normalized) for pattern in response_1_patterns)
        has_2 = any(re.search(pattern, normalized) for pattern in response_2_patterns)
        if has_1 != has_2:
            return ("response_1" if has_1 else "response_2"), None, None
        return None

    @staticmethod
    def _parse_json(
        text: str,
    ) -> tuple[PairwiseWinner, float | None, float | None] | None:
        decoder = json.JSONDecoder()
        for index, character in enumerate(text):
            if character != "{":
                continue
            try:
                payload, _ = decoder.raw_decode(text[index:])
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict):
                continue
            winner = str(payload.get("winner") or "").strip().casefold()
            mapping = {
                "1": "response_1",
                "a": "response_1",
                "response_1": "response_1",
                "response 1": "response_1",
                "2": "response_2",
                "b": "response_2",
                "response_2": "response_2",
                "response 2": "response_2",
                "tie": "tie",
                "draw": "tie",
            }
            if winner in mapping:
                return mapping[winner], None, None
        return None

    @staticmethod
    def _parse_score_pair(
        text: str,
    ) -> tuple[PairwiseWinner, float, float] | None:
        patterns = (
            r"^\s*\[*\(?\s*(\d+(?:\.\d+)?)\s*[,\s]+(\d+(?:\.\d+)?)\s*\)?\]*",
            r"score\s+of\s+(?:response|assistant)\s*1\s*:\s*(\d+(?:\.\d+)?).*?score\s+of\s+(?:response|assistant)\s*2\s*:\s*(\d+(?:\.\d+)?)",
        )
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
            if not match:
                continue
            score_1, score_2 = (float(value) for value in match.groups())
            if score_1 > score_2:
                winner: PairwiseWinner = "response_1"
            elif score_2 > score_1:
                winner = "response_2"
            else:
                winner = "tie"
            return winner, score_1, score_2
        return None
