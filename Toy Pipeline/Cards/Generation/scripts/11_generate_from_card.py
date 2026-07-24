#!/usr/bin/env python3
"""
11_generate_from_card.py

Generate a counter-narrative for one hate-speech instance using one validated
quote-first EDOS card as grounding.

This module is intentionally small and local-first:
- It calls Ollama through the local HTTP API.
- It uses one card at a time.
- It does not do retrieval. Use script 08 to select cards first.
- It outputs a JSON file with the generated response and provenance.

Example from repository root:

    python scripts/11_generate_from_card.py \
      --hate-speech "Women are too emotional to make rational decisions." \
      --cards data/cards/validated/validated_cards.jsonl \
      --card-id EPRS_2021_EDOS_3_2_001 \
      --model llama3.1:8b \
      --output data/generated/example_generation.json

You can also pass the card as a standalone JSON file:

    python scripts/11_generate_from_card.py \
      --hate-speech "..." \
      --card-json examples/example_card.json \
      --output data/generated/example_generation.json

Optional:
    --strategy "Fact-Checking"
    --language English
    --max-sentences 2
    --temperature 0.2
    --show-prompt

Expected card schema:
    card_id
    source
    source_quote
    claim
    primary_edos_label
    secondary_edos_labels
    edos_alignment
    retrieval_keywords
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
    return rows


def load_card_from_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        if len(data) != 1:
            raise ValueError("card-json contains a list. Provide a single-card JSON file or use --cards + --card-id.")
        return data[0]
    if not isinstance(data, dict):
        raise ValueError("card-json must contain a JSON object.")
    return data


def load_card_from_jsonl(path: Path, card_id: str) -> Dict[str, Any]:
    cards = read_jsonl(path)
    for card in cards:
        if card.get("card_id") == card_id:
            return card
    raise ValueError(f"Could not find card_id={card_id!r} in {path}")


def source_to_compact_string(source: Any) -> str:
    if not isinstance(source, dict):
        return str(source or "")

    title = source.get("title") or ""
    publisher = source.get("publisher") or ""
    year = source.get("year") or ""
    page = source.get("page")
    section = source.get("section") or ""
    url = source.get("url") or ""

    parts = []
    if title:
        parts.append(title)
    if publisher:
        parts.append(publisher)
    if year:
        parts.append(str(year))
    if page not in [None, ""]:
        parts.append(f"p. {page}")
    if section:
        parts.append(f"section: {section}")
    if url:
        parts.append(url)
    return " | ".join(parts)


def validate_minimal_card(card: Dict[str, Any]) -> None:
    required = [
        "card_id",
        "source_quote",
        "claim",
        "primary_edos_label",
        "edos_alignment",
    ]
    missing = [field for field in required if not card.get(field)]
    if missing:
        raise ValueError(f"Card is missing required fields: {missing}")


def build_generation_prompt(
    hate_speech: str,
    card: Dict[str, Any],
    strategy: Optional[str],
    language: str,
    max_sentences: int,
    cite_source: bool,
) -> List[Dict[str, str]]:
    source_string = source_to_compact_string(card.get("source"))

    system = f"""You generate concise counter-narratives against gender-based hate speech.

Rules:
- Answer in {language}.
- Use the evidence card as grounding.
- Do not invent facts beyond the card.
- Do not claim personal lived experience.
- Do not insult or dehumanize the author of the hate speech.
- Do not amplify the hateful claim.
- Directly counter the harmful idea.
- Keep the response to at most {max_sentences} sentence(s).
- Do not mention that you are an AI.
- {'You may briefly mention the source if useful.' if cite_source else 'Do not explicitly cite or mention the source unless necessary.'}

Return valid JSON only, with this shape:
{{
  "counter_narrative": "string",
  "used_claim": "string",
  "grounding_note": "string"
}}"""

    strategy_text = strategy if strategy else "not specified"

    user = f"""HATE_SPEECH:
{hate_speech}

EDOS LABEL:
{card.get("primary_edos_label", "")}

OPTIONAL COUNTERSPEECH STRATEGY:
{strategy_text}

EVIDENCE CARD:
card_id: {card.get("card_id", "")}
source: {source_string}
source_quote: {card.get("source_quote", "")}
claim: {card.get("claim", "")}
edos_alignment: {card.get("edos_alignment", "")}
retrieval_keywords: {", ".join(card.get("retrieval_keywords", []) or [])}

Task:
Generate one counter-narrative grounded in this card.
The response should be suitable as a counterspeech reply to the hate speech.
"""

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def ollama_chat(
    messages: List[Dict[str, str]],
    model: str,
    base_url: str,
    temperature: float,
    timeout: int,
    retries: int = 2,
) -> str:
    endpoint = base_url.rstrip("/") + "/api/chat"
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": temperature,
        },
    }

    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    last_error: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = response.read().decode("utf-8")
            parsed = json.loads(raw)
            return parsed["message"]["content"]
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, KeyError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(1.0 + attempt)
            else:
                break

    raise RuntimeError(
        f"Failed to call Ollama at {endpoint}. "
        f"Is Ollama running? Try: ollama serve . Last error: {last_error}"
    )


def extract_json_object(text: str) -> Dict[str, Any]:
    text = text.strip()

    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start : end + 1]
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass

    return {
        "counter_narrative": text,
        "used_claim": "",
        "grounding_note": "Model did not return valid JSON; raw text was stored.",
    }


def normalize_generation_output(obj: Dict[str, Any]) -> Dict[str, str]:
    return {
        "counter_narrative": str(obj.get("counter_narrative", "")).strip(),
        "used_claim": str(obj.get("used_claim", "")).strip(),
        "grounding_note": str(obj.get("grounding_note", "")).strip(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a counter-narrative from one hate speech and one EDOS evidence card using Ollama."
    )

    parser.add_argument("--hate-speech", required=True, help="Input hate speech text.")
    parser.add_argument("--cards", help="Path to JSONL card library.")
    parser.add_argument("--card-id", help="Card ID to use from --cards.")
    parser.add_argument("--card-json", help="Path to a standalone card JSON file.")

    parser.add_argument("--model", default="llama3.1:8b", help="Ollama model name.")
    parser.add_argument("--ollama-url", default="http://localhost:11434", help="Ollama base URL.")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--timeout", type=int, default=120)

    parser.add_argument("--strategy", default=None, help="Optional counterspeech strategy, e.g. Fact-Checking.")
    parser.add_argument("--language", default="English", help="Output language.")
    parser.add_argument("--max-sentences", type=int, default=2)
    parser.add_argument("--cite-source", action="store_true", help="Allow explicit mention of the source.")
    parser.add_argument("--show-prompt", action="store_true", help="Print prompt messages before calling Ollama.")

    parser.add_argument("--output", default="data/generated/generated_from_card.json", help="Output JSON path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.card_json:
        card = load_card_from_json(Path(args.card_json))
    else:
        if not args.cards or not args.card_id:
            raise SystemExit("Provide either --card-json or both --cards and --card-id.")
        card = load_card_from_jsonl(Path(args.cards), args.card_id)

    validate_minimal_card(card)

    messages = build_generation_prompt(
        hate_speech=args.hate_speech,
        card=card,
        strategy=args.strategy,
        language=args.language,
        max_sentences=args.max_sentences,
        cite_source=args.cite_source,
    )

    if args.show_prompt:
        print("\n--- SYSTEM ---\n")
        print(messages[0]["content"])
        print("\n--- USER ---\n")
        print(messages[1]["content"])
        print("\n-------------\n")

    raw_output = ollama_chat(
        messages=messages,
        model=args.model,
        base_url=args.ollama_url,
        temperature=args.temperature,
        timeout=args.timeout,
    )

    parsed = normalize_generation_output(extract_json_object(raw_output))

    result = {
        "input": {
            "hate_speech": args.hate_speech,
            "strategy": args.strategy,
            "language": args.language,
            "max_sentences": args.max_sentences,
        },
        "card": {
            "card_id": card.get("card_id"),
            "primary_edos_label": card.get("primary_edos_label"),
            "claim": card.get("claim"),
            "source_quote": card.get("source_quote"),
            "source": card.get("source"),
        },
        "generation": parsed,
        "model": {
            "provider": "ollama",
            "model": args.model,
            "temperature": args.temperature,
            "ollama_url": args.ollama_url,
        },
        "raw_model_output": raw_output,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"\nSaved generation to: {output_path}")


if __name__ == "__main__":
    main()
