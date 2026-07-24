#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from card_routed_rag.card_validation import load_schema, schema_errors
from card_routed_rag.hf_generation import (
    HuggingFaceChatGenerator,
    HuggingFaceModelSpec,
)
from card_routed_rag.io_utils import read_jsonl, read_yaml, write_json
from card_routed_rag.ollama_client import extract_json_object
from card_routed_rag.text_utils import quote_in_passage


SYSTEM_PROMPT = (
    "You extract quote-grounded JSON evidence cards. "
    "Return exactly one JSON object and no prose or Markdown."
)

CARD_FIELDS = {
    "card_id",
    "status",
    "source",
    "source_quote",
    "claim",
    "primary_edos_label",
    "secondary_edos_labels",
    "edos_alignment",
    "retrieval_keywords",
    "validation",
}


def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def append_jsonl(path: Path, item: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def slug(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_")
    return cleaned or "item"


def make_card_id(
    model_alias: str,
    label: str,
    passage: dict[str, Any],
) -> str:
    source_id = str((passage.get("source") or {}).get("source_id", "SRC"))
    chunk_id = str(passage.get("chunk_id", "CHUNK"))
    return "_".join(
        [
            slug(model_alias).upper(),
            slug(source_id).upper(),
            slug(label.split()[0]).upper(),
            slug(chunk_id.split("_")[-1]).upper(),
        ]
    )


def make_prompt(
    passage: dict[str, Any],
    edos_label: str,
    definition: str,
    schema: dict[str, Any],
    min_quote: int,
    max_quote: int,
) -> str:
    return f"""
You are extracting one quote-grounded candidate evidence card for a
card-routed RAG system for counterspeech against gender-based hate.

Target EDOS Task C label:
{edos_label}

EDOS label definition:
{definition}

Rules:
- Extract exactly ONE candidate card from the source passage.
- Do NOT generate counterspeech.
- Do NOT invent information not present in the passage.
- The source_quote MUST be copied verbatim from the source passage.
- Prefer a source_quote between {min_quote} and {max_quote} characters.
- The claim must be atomic and directly supported by source_quote.
- The card must be specific to the target EDOS label.
- edos_alignment must explain that label-specific relevance.
- retrieval_keywords must contain concise strings useful for retrieval.
- Use status="candidate".
- Use validation.status="not_validated" and null for every other
  validation value.
- Return only the JSON object, without a code fence.

Required JSON schema:
{json.dumps(schema, ensure_ascii=False)}

Source metadata:
{json.dumps(passage.get("source", {}), ensure_ascii=False, indent=2)}

Source chunk ID:
{passage.get("chunk_id")}

Source passage:
{passage.get("text", "")}
""".strip()


def normalize_generated_card(
    raw_card: dict[str, Any],
    *,
    model_alias: str,
    label: str,
    passage: dict[str, Any],
) -> dict[str, Any]:
    card = {key: raw_card.get(key) for key in CARD_FIELDS}
    card["card_id"] = make_card_id(model_alias, label, passage)
    card["status"] = "candidate"
    card["source"] = passage.get("source", {})
    card["primary_edos_label"] = label
    secondary = card.get("secondary_edos_labels")
    card["secondary_edos_labels"] = secondary if isinstance(secondary, list) else []
    keywords = card.get("retrieval_keywords")
    card["retrieval_keywords"] = keywords if isinstance(keywords, list) else []
    card["validation"] = {
        "status": "not_validated",
        "faithfulness": None,
        "edos_alignment": None,
        "usefulness": None,
        "notes": None,
    }
    return card


def load_attempted_chunks(path: Path) -> set[str]:
    attempted: set[str] = set()
    for item in read_jsonl(path):
        label = str(item.get("label", ""))
        chunk_id = str(item.get("chunk_id", ""))
        if label and chunk_id:
            attempted.add(f"{label}\0{chunk_id}")
    return attempted


def select_model_specs(
    cfg: dict[str, Any],
    requested: list[str] | None,
) -> list[HuggingFaceModelSpec]:
    available = cfg.get("models", {}) or {}
    aliases = requested or [
        alias
        for alias, model_cfg in available.items()
        if bool((model_cfg or {}).get("enabled", True))
    ]
    unknown = [alias for alias in aliases if alias not in available]
    if unknown:
        raise SystemExit(
            f"Unknown model aliases: {unknown}. Available: {list(available)}"
        )
    return [
        HuggingFaceModelSpec(
            alias=alias,
            repo_id=str(available[alias]["repo_id"]),
            backend=str(available[alias]["backend"]),
            disable_thinking=bool(
                available[alias].get("disable_thinking", False)
            ),
            load_in_4bit=bool(available[alias].get("load_in_4bit", True)),
        )
        for alias in aliases
    ]


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Generate quote-grounded EDOS candidate cards from retrieved chunks "
            "with multiple local Hugging Face models."
        )
    )
    ap.add_argument(
        "--retrieved-chunks",
        default="data/retrieval_outputs/edos_label_chunks.json",
    )
    ap.add_argument("--profiles", default="configs/edos_label_profiles.yaml")
    ap.add_argument("--schema", default="configs/card_schema.json")
    ap.add_argument(
        "--models-config",
        default="configs/generation_models_hf.yaml",
    )
    ap.add_argument(
        "--output-dir",
        default="data/cards/candidates/huggingface",
    )
    ap.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Model aliases to run. By default all enabled models are run.",
    )
    ap.add_argument("--cards-per-label", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=None)
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--labels", nargs="+", default=None)
    ap.add_argument(
        "--resume",
        action="store_true",
        help="Skip label/chunk attempts already present in each attempt log.",
    )
    args = ap.parse_args()

    retrieved = read_json(args.retrieved_chunks)
    retrieved_labels = retrieved.get("labels", {}) or {}
    if not retrieved_labels:
        raise SystemExit(f"No retrieved EDOS chunks found at {args.retrieved_chunks}.")

    profiles = read_yaml(args.profiles).get("labels", {})
    schema = load_schema(args.schema)
    model_cfg = read_yaml(args.models_config)
    generation_cfg = model_cfg.get("generation", {}) or {}
    specs = select_model_specs(model_cfg, args.models)

    cards_per_label = (
        args.cards_per_label
        if args.cards_per_label is not None
        else int(generation_cfg.get("cards_per_label", 10))
    )
    max_new_tokens = (
        args.max_new_tokens
        if args.max_new_tokens is not None
        else int(generation_cfg.get("max_new_tokens", 900))
    )
    temperature = (
        args.temperature
        if args.temperature is not None
        else float(generation_cfg.get("temperature", 0.0))
    )
    min_quote = int(generation_cfg.get("min_quote_chars", 20))
    max_quote = int(generation_cfg.get("max_quote_chars", 280))
    enforce_quote = bool(generation_cfg.get("enforce_quote_match", True))
    quantization = generation_cfg.get("quantization", {}) or {}
    max_memory = generation_cfg.get("max_memory", {}) or {}
    selected_labels = args.labels or list(retrieved_labels)

    missing_labels = [
        label
        for label in selected_labels
        if label not in retrieved_labels or label not in profiles
    ]
    if missing_labels:
        raise SystemExit(f"Unknown or missing EDOS labels: {missing_labels}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_summary: dict[str, Any] = {
        "input": args.retrieved_chunks,
        "models": {},
    }

    for spec in specs:
        cards_path = output_dir / f"{spec.alias}_candidate_cards.jsonl"
        attempts_path = output_dir / f"{spec.alias}_attempts.jsonl"
        if (cards_path.exists() or attempts_path.exists()) and not args.resume:
            raise SystemExit(
                f"Output already exists for {spec.alias}. Use --resume or choose "
                f"a different --output-dir."
            )
        attempted = load_attempted_chunks(attempts_path) if args.resume else set()
        model_stats = {"accepted": 0, "rejected": 0, "errors": 0, "skipped": 0}
        print(f"[model] loading {spec.alias}: {spec.repo_id}")

        with HuggingFaceChatGenerator(
            spec,
            quantization=quantization,
            max_memory=max_memory,
        ) as generator:
            for label in selected_labels:
                definition = str(profiles[label]["definition"])
                passages = retrieved_labels[label].get("chunks", [])[:cards_per_label]
                print(f"[label] {label}: {len(passages)} chunks")
                for passage in passages:
                    chunk_id = str(passage.get("chunk_id", ""))
                    attempt_key = f"{label}\0{chunk_id}"
                    if attempt_key in attempted:
                        model_stats["skipped"] += 1
                        continue
                    raw_output: str | None = None
                    try:
                        prompt = make_prompt(
                            passage,
                            label,
                            definition,
                            schema,
                            min_quote,
                            max_quote,
                        )
                        raw_output = generator.generate(
                            prompt,
                            system_prompt=SYSTEM_PROMPT,
                            max_new_tokens=max_new_tokens,
                            temperature=temperature,
                        )
                        raw_card = extract_json_object(raw_output)
                        card = normalize_generated_card(
                            raw_card,
                            model_alias=spec.alias,
                            label=label,
                            passage=passage,
                        )
                        if enforce_quote and not quote_in_passage(
                            str(card.get("source_quote", "")),
                            str(passage.get("text", "")),
                        ):
                            raise ValueError("source_quote not found in source passage")
                        errors = schema_errors(card, schema)
                        if errors:
                            raise ValueError(f"schema errors: {errors[:3]}")

                        append_jsonl(cards_path, card)
                        append_jsonl(
                            attempts_path,
                            {
                                "status": "accepted",
                                "model_alias": spec.alias,
                                "model_id": spec.repo_id,
                                "label": label,
                                "chunk_id": chunk_id,
                                "card_id": card["card_id"],
                            },
                        )
                        model_stats["accepted"] += 1
                        print(f"  [ok] {card['card_id']}")
                    except ValueError as exc:
                        append_jsonl(
                            attempts_path,
                            {
                                "status": "rejected",
                                "model_alias": spec.alias,
                                "model_id": spec.repo_id,
                                "label": label,
                                "chunk_id": chunk_id,
                                "error": str(exc),
                                "raw_output": raw_output,
                            },
                        )
                        model_stats["rejected"] += 1
                        print(f"  [reject] {chunk_id}: {exc}")
                    except Exception as exc:
                        append_jsonl(
                            attempts_path,
                            {
                                "status": "error",
                                "model_alias": spec.alias,
                                "model_id": spec.repo_id,
                                "label": label,
                                "chunk_id": chunk_id,
                                "error": f"{type(exc).__name__}: {exc}",
                            },
                        )
                        model_stats["errors"] += 1
                        print(f"  [error] {chunk_id}: {exc}")

        run_summary["models"][spec.alias] = {
            "model_id": spec.repo_id,
            "cards_file": str(cards_path),
            "attempts_file": str(attempts_path),
            **model_stats,
        }
        write_json(run_summary, output_dir / "generation_summary.json")
        print(f"[unload] {spec.alias}")

    print(json.dumps(run_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

