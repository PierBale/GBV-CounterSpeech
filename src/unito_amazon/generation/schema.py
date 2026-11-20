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


class CounterSpeechResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    counter_speech: str = Field(min_length=1)


class CounterNarrativeResponse(BaseModel):
    """Key used by the earlier prompts (data/counterspeech_v2)."""
    model_config = ConfigDict(extra="forbid")

    counter_narrative: str = Field(min_length=1)


def parse_counter_narrative(raw_response: str) -> tuple[str, bool]:
    """Text of a {"counter_speech": ...} JSON response ({"counter_narrative": ...} for the
    earlier prompts); (raw response, False) when it cannot be parsed."""
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
        except json.JSONDecodeError:
            continue
        for model, key in ((CounterSpeechResponse, "counter_speech"), (CounterNarrativeResponse, "counter_narrative")):
            try:
                return getattr(model.model_validate(payload), key).strip(), True
            except ValueError:
                continue

    return raw_response, False
