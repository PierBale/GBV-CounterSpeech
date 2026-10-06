import json
from json import JSONDecodeError

from pydantic import ValidationError

from unito_amazon.llm.base import BaseLLM
from unito_amazon.llm.schema import LLMGenerationConfig

from .prompts import CounterSpeechJudgePrompt
from .schema import JudgeAssessment, JudgeResult


class LLMJudge:
    def __init__(
        self,
        llm: BaseLLM,
        prompt: CounterSpeechJudgePrompt | None = None,
        max_retries: int = 1,
    ):
        if max_retries < 0:
            raise ValueError("max_retries cannot be negative.")

        self.llm = llm
        self.prompt = prompt or CounterSpeechJudgePrompt()
        self.max_retries = max_retries
        self.generation_config = LLMGenerationConfig(
            max_new_tokens=512,
            do_sample=False,
            repetition_penalty=1.0,
            system_prompt=self.prompt.system_prompt,
        )

    def judge(self, record: dict) -> JudgeResult:
        return self.judge_batch([record])[0]

    def judge_batch(self, records: list[dict]) -> list[JudgeResult]:
        prompts = [self._build_prompt(record) for record in records]
        response = self.llm.generate_batch(
            prompts=prompts,
            config=self.generation_config,
        )

        return [
            self._parse_with_retries(record, raw_response)
            for record, raw_response in zip(
                records,
                response.texts,
                strict=True,
            )
        ]

    def _build_prompt(self, record: dict) -> str:
        required_fields = ("hateful_message", "generated_counter_speech")
        missing = [field for field in required_fields if not record.get(field)]
        if missing:
            raise ValueError(
                f"Judge input is missing required fields: {', '.join(missing)}."
            )

        return self.prompt.build(
            hateful_message=record["hateful_message"],
            counter_speech=record["generated_counter_speech"],
            evidence=record.get("evidence") or [],
        )

    def _parse_with_retries(
        self,
        record: dict,
        raw_response: str,
    ) -> JudgeResult:
        attempts = 1
        current_response = raw_response

        while True:
            try:
                assessment = self._parse_assessment(current_response)
                self._validate_grounding_mode(record, assessment)
                return JudgeResult(
                    assessment=assessment,
                    passed=self._is_passing(record, assessment),
                    raw_response=current_response,
                    attempts=attempts,
                )
            except (JSONDecodeError, ValidationError, ValueError) as error:
                if attempts > self.max_retries:
                    return JudgeResult(
                        raw_response=current_response,
                        parse_error=str(error),
                        attempts=attempts,
                    )

                repair_prompt = self.prompt.build_repair_prompt(
                    raw_response=current_response,
                    error=str(error),
                )
                repaired = self.llm.generate(
                    prompt=repair_prompt,
                    config=self.generation_config,
                )
                current_response = repaired.text
                attempts += 1

    @staticmethod
    def _parse_assessment(raw_response: str) -> JudgeAssessment:
        decoder = json.JSONDecoder()
        for index, character in enumerate(raw_response):
            if character != "{":
                continue
            try:
                payload, _ = decoder.raw_decode(raw_response[index:])
                return JudgeAssessment.model_validate(payload)
            except JSONDecodeError:
                continue

        raise JSONDecodeError(
            "No valid JSON object found in judge response",
            raw_response,
            0,
        )

    @staticmethod
    def _validate_grounding_mode(
        record: dict,
        assessment: JudgeAssessment,
    ) -> None:
        has_evidence = bool(record.get("evidence"))
        if has_evidence and assessment.evidence_grounding is None:
            raise ValueError("evidence_grounding must be scored for RAG output.")
        if not has_evidence and assessment.evidence_grounding is not None:
            raise ValueError(
                "evidence_grounding must be null for generation without RAG."
            )

    @staticmethod
    def _is_passing(record: dict, assessment: JudgeAssessment) -> bool:
        required_scores = [
            assessment.relevance,
            assessment.respectfulness,
            assessment.persuasiveness,
            assessment.self_contained,
            assessment.conciseness,
        ]
        if record.get("evidence"):
            required_scores.append(assessment.evidence_grounding or 1)

        return assessment.overall >= 4 and min(required_scores) >= 3
