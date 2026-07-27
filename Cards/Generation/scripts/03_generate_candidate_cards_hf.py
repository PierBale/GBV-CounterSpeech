#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
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


SYSTEM_PROMPT = (
    "You develop evidence-grounded counter-arguments to gender-based hate speech. "
    "Return exactly one JSON object and no prose or Markdown."
)

GENERATED_FIELDS = {"reasoning", "argument"}


def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def append_jsonl(path: Path, item: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(item, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def checkpoint_model(
    run_summary: dict[str, Any],
    *,
    spec: HuggingFaceModelSpec,
    cards_path: Path,
    attempts_path: Path,
    model_stats: dict[str, int],
    output_dir: Path,
) -> None:
    run_summary["models"][spec.alias] = {
        "model_id": spec.repo_id,
        "cards_file": str(cards_path),
        "attempts_file": str(attempts_path),
        **model_stats,
    }
    write_json(run_summary, output_dir / "generation_summary.json")


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
) -> str:
    return f"""
You are extracting one quote-grounded candidate evidence card for a
card-routed RAG system for counterspeech against gender-based hate.

Target EDOS Task C label:
{edos_label}

EDOS label definition:
{definition}

Task:
1. Write a concise reasoning that connects evidence in the source passage to
   the harmful idea represented by the target EDOS label. Explain why that
   evidence supports a response to this specific type of hate speech.
2. Based on that reasoning, write one self-contained argument against the
   indicated type of hate speech. The argument must challenge the harmful idea,
   remain factual, and avoid insulting or dehumanizing anyone.
Rules:
- Use only information present in the source passage.
- Do not invent facts, statistics, laws, or citations.
- Keep reasoning concise (2-4 sentences); do not include hidden deliberation,
  alternative drafts, or meta-commentary.
- The argument must be specific to the target EDOS label, not generic advice.
- Return only this intermediate JSON object, in this field order:
{{
  "reasoning": "concise evidence-based reasoning",
  "argument": "standalone counter-argument"
}}
- Do not quote or copy the source passage into a separate field.
- Do not return card metadata, the source chunk, labels, validation fields, or
  a code fence; those fields are filled automatically by the application.

Source metadata:
{json.dumps(passage.get("source", {}), ensure_ascii=False, indent=2)}

Source chunk ID:
{passage.get("chunk_id")}

Source passage:
{passage.get("text", "")}
""".strip()


def normalize_generated_card(
    generated: dict[str, Any],
    *,
    model_alias: str,
    label: str,
    definition: str,
    passage: dict[str, Any],
) -> dict[str, Any]:
    missing = GENERATED_FIELDS - generated.keys()
    if missing:
        raise ValueError(f"generated output is missing fields: {sorted(missing)}")
    values = {key: generated.get(key) for key in GENERATED_FIELDS}
    invalid = [key for key, value in values.items() if not isinstance(value, str) or not value.strip()]
    if invalid:
        raise ValueError(f"generated fields must be non-empty strings: {invalid}")

    argument = values["argument"].strip()
    label_terms = re.findall(r"[A-Za-z][A-Za-z'-]{2,}", label.lower())
    argument_terms = re.findall(r"[A-Za-z][A-Za-z'-]{4,}", argument.lower())
    keywords = list(dict.fromkeys([label, *label_terms, *argument_terms]))[:10]

    return {
        "card_id": make_card_id(model_alias, label, passage),
        "status": "candidate",
        "source": passage.get("source", {}),
        "chunk": str(passage.get("text", "")),
        "reasoning": values["reasoning"].strip(),
        "argument": argument,
        "primary_edos_label": label,
        "secondary_edos_labels": [],
        "edos_alignment": (
            f"This argument addresses {label}. EDOS definition: {definition}"
        ),
        "retrieval_keywords": keywords,
        "validation": {
            "status": "not_validated",
            "faithfulness": None,
            "edos_alignment": None,
            "usefulness": None,
            "notes": None,
        },
    }


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
        expected_attempts = {
            f"{label}\0{passage.get('chunk_id', '')}"
            for label in selected_labels
            for passage in retrieved_labels[label].get("chunks", [])[:cards_per_label]
        }
        if args.resume and expected_attempts and expected_attempts <= attempted:
            model_stats["skipped"] = len(expected_attempts)
            checkpoint_model(
                run_summary,
                spec=spec,
                cards_path=cards_path,
                attempts_path=attempts_path,
                model_stats=model_stats,
                output_dir=output_dir,
            )
            print(
                f"[skip] {spec.alias}: all {len(expected_attempts)} "
                "label/chunk attempts already exist"
            )
            continue
        checkpoint_model(
            run_summary,
            spec=spec,
            cards_path=cards_path,
            attempts_path=attempts_path,
            model_stats=model_stats,
            output_dir=output_dir,
        )
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
                        checkpoint_model(
                            run_summary,
                            spec=spec,
                            cards_path=cards_path,
                            attempts_path=attempts_path,
                            model_stats=model_stats,
                            output_dir=output_dir,
                        )
                        continue
                    raw_output: str | None = None
                    try:
                        prompt = make_prompt(
                            passage,
                            label,
                            definition,
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
                            definition=definition,
                            passage=passage,
                        )
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
                    finally:
                        checkpoint_model(
                            run_summary,
                            spec=spec,
                            cards_path=cards_path,
                            attempts_path=attempts_path,
                            model_stats=model_stats,
                            output_dir=output_dir,
                        )

        print(f"[unload] {spec.alias}")

    print(json.dumps(run_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
