from __future__ import annotations

import json
from typing import Any

try:
    from json_repair import repair_json
except Exception:  # optional dependency fallback
    def repair_json(text: str) -> str:
        return text


def extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        snippet = text[start:end + 1]
        try:
            return json.loads(repair_json(snippet))
        except Exception as exc:
            raise ValueError(f"Could not parse JSON object: {exc}\n{text[:500]}") from exc
    raise ValueError(f"No JSON object found in model output: {text[:500]}")


def ollama_json(prompt: str, schema: dict[str, Any], model: str, temperature: float = 0.0) -> dict[str, Any]:
    import ollama

    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": "You extract quote-grounded JSON evidence cards. Return only JSON."},
            {"role": "user", "content": prompt},
        ],
        format=schema,
        options={"temperature": temperature},
    )
    content = response["message"]["content"]
    return extract_json_object(content)
