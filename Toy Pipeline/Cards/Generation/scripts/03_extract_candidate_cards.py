#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from card_routed_rag.io_utils import read_jsonl, write_jsonl, read_yaml
from card_routed_rag.card_validation import load_schema, schema_errors
from card_routed_rag.ollama_client import ollama_json
from card_routed_rag.text_utils import keyword_overlap_score, quote_in_passage, normalize_space


def select_passages_for_label(chunks: list[dict[str, Any]], terms: list[str], max_passages: int) -> list[dict[str, Any]]:
    scored = []
    for c in chunks:
        text = c.get("text", "")
        score = keyword_overlap_score(text, terms)
        if score > 0:
            scored.append((score, c))
    scored.sort(key=lambda x: x[0], reverse=True)
    if not scored:
        return chunks[:max_passages]
    return [c for _, c in scored[:max_passages]]


def make_prompt(passage: dict[str, Any], edos_label: str, definition: str, min_quote: int, max_quote: int) -> str:
    source = passage["source"]
    return f"""
You are extracting quote-grounded candidate evidence cards for a card-routed RAG system for counterspeech against gender-based hate.

Target EDOS label:
{edos_label}

EDOS label definition:
{definition}

Task:
Extract exactly ONE candidate card from the source passage, only if the passage contains a short ORIGINAL quote that directly supports a useful claim for this EDOS label.

Rules:
- Do NOT generate counterspeech.
- Do NOT make a generic card that could fit every EDOS label.
- The source_quote MUST be copied verbatim from the passage.
- The source_quote should be between {min_quote} and {max_quote} characters when possible.
- The claim must be atomic: one reusable idea only.
- The claim must be directly supported by the source_quote.
- The edos_alignment must explain why the card is useful for this specific EDOS label.
- secondary_edos_labels can be empty.
- Use status="candidate" and validation.status="not_validated" with null scores.

Source metadata:
{json.dumps(source, ensure_ascii=False, indent=2)}

Source passage:
{passage['text']}
""".strip()


def make_mock_card(passage: dict[str, Any], edos_label: str, idx: int, definition: str) -> dict[str, Any]:
    text = normalize_space(passage.get("text", ""))
    quote = " ".join(text.split()[:28])
    source = passage["source"]
    safe_label = edos_label.replace(" ", "_").replace("/", "_").replace("&", "and").replace(".", "_")
    return {
        "card_id": f"MOCK_{safe_label}_{idx:03d}",
        "status": "candidate",
        "source": source,
        "source_quote": quote,
        "claim": f"This source passage provides a quote-grounded candidate claim relevant to {edos_label}.",
        "primary_edos_label": edos_label,
        "secondary_edos_labels": [],
        "edos_alignment": f"This mock card is aligned to {edos_label}: {definition}",
        "retrieval_keywords": list({w.lower() for w in edos_label.replace("&", " ").replace(".", " ").split() if len(w) > 2})[:8],
        "validation": {
            "status": "not_validated",
            "faithfulness": None,
            "edos_alignment": None,
            "usefulness": None,
            "notes": None,
        },
    }


def ensure_card_id(card: dict[str, Any], edos_label: str, n: int) -> dict[str, Any]:
    if not card.get("card_id"):
        label_slug = edos_label.split()[0].replace(".", "_")
        src = (card.get("source") or {}).get("source_id", "SRC")
        card["card_id"] = f"{src}_{label_slug}_{n:03d}"
    return card


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract EDOS-specific, quote-first candidate evidence cards using Ollama or mock mode.")
    ap.add_argument("--chunks", default="data/processed/document_chunks.jsonl")
    ap.add_argument("--output", default="data/cards/candidates/candidate_cards.jsonl")
    ap.add_argument("--profiles", default="configs/edos_label_profiles.yaml")
    ap.add_argument("--schema", default="configs/card_schema.json")
    ap.add_argument("--config", default="configs/extraction_config.yaml")
    ap.add_argument("--model", default=None)
    ap.add_argument("--cards-per-label", type=int, default=None)
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--no-enforce-quote-match", action="store_true")
    args = ap.parse_args()

    chunks = read_jsonl(args.chunks)
    if not chunks:
        raise SystemExit(f"No chunks found at {args.chunks}. Run 02_parse_sources.py first or use examples/example_chunks.jsonl.")

    profiles = read_yaml(args.profiles)["labels"]
    cfg = read_yaml(args.config)
    schema = load_schema(args.schema)
    cards_per_label = args.cards_per_label or int(cfg.get("cards_per_label", 5))
    max_passages = int(cfg.get("max_candidate_passages_per_label", 40))
    model = args.model or cfg.get("ollama_model", "llama3.1:8b")
    temperature = float(cfg.get("temperature", 0))
    enforce_quote = bool(cfg.get("enforce_quote_match", True)) and not args.no_enforce_quote_match
    min_quote = int(cfg.get("min_quote_chars", 20))
    max_quote = int(cfg.get("max_quote_chars", 280))

    all_cards: list[dict[str, Any]] = []
    for edos_label, profile in profiles.items():
        print(f"[label] {edos_label}")
        passages = select_passages_for_label(chunks, profile.get("search_terms", []), max_passages)
        label_cards: list[dict[str, Any]] = []
        for p_idx, passage in enumerate(passages, start=1):
            if len(label_cards) >= cards_per_label:
                break
            try:
                if args.mock:
                    card = make_mock_card(passage, edos_label, len(label_cards) + 1, profile.get("definition", ""))
                else:
                    prompt = make_prompt(passage, edos_label, profile.get("definition", ""), min_quote, max_quote)
                    card = ollama_json(prompt, schema=schema, model=model, temperature=temperature)
                    # Force source and target label from the pipeline, not from model hallucination.
                    card["source"] = passage["source"]
                    card["primary_edos_label"] = edos_label
                    card.setdefault("secondary_edos_labels", [])
                    card = ensure_card_id(card, edos_label, len(label_cards) + 1)

                if enforce_quote and not quote_in_passage(card.get("source_quote", ""), passage.get("text", "")):
                    print(f"  [skip] quote not found in passage for {card.get('card_id')}")
                    continue
                errors = schema_errors(card, schema)
                if errors:
                    print(f"  [skip] schema errors for {card.get('card_id')}: {errors[:2]}")
                    continue
                label_cards.append(card)
                print(f"  [ok] {card['card_id']} ({len(label_cards)}/{cards_per_label})")
            except Exception as exc:
                print(f"  [warn] extraction failed on passage {p_idx}: {exc}")
        if len(label_cards) < cards_per_label:
            print(f"  [warn] only {len(label_cards)}/{cards_per_label} cards extracted for {edos_label}")
        all_cards.extend(label_cards)

    write_jsonl(all_cards, args.output)
    print(f"[done] wrote {len(all_cards)} candidate cards to {args.output}")


if __name__ == "__main__":
    main()
