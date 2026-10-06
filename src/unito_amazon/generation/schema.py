from typing import Any

import json
import re

from pydantic import BaseModel, ConfigDict, Field

from unito_amazon.retrieval.schema import RetrievalResult


class GenerationResult(BaseModel):
    hateful_message: str
    counter_speech: str
    system_prompt: str = ""
    prompt: str
    context: list[str] = Field(default_factory=list)
    retrieval_results: list[RetrievalResult] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class CounterNarrativeResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    counter_narrative: str = Field(min_length=1)


def parse_counter_narrative(raw_response: str) -> tuple[str, bool]:
    raw_response = raw_response.strip()
    candidates = [raw_response]

    fenced_match = re.search(
        r"```(?:json)?\s*(\{.*?\})\s*```",
        raw_response,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if fenced_match:
        candidates.append(fenced_match.group(1))

    first_brace = raw_response.find("{")
    last_brace = raw_response.rfind("}")
    if first_brace >= 0 and last_brace > first_brace:
        candidates.append(raw_response[first_brace : last_brace + 1])

    for candidate in candidates:
        try:
            payload = json.loads(candidate)
            parsed = CounterNarrativeResponse.model_validate(payload)
            return parsed.counter_narrative.strip(), True
        except (json.JSONDecodeError, ValueError):
            continue

    return raw_response, False
