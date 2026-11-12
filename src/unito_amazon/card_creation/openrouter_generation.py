"""Evidence-card generation through OpenRouter (Batch API by default), with a full
reproducibility record.

Reuses the existing prompt (`SYSTEM_PROMPT`, `build_card_prompt`), output
validation (`GeneratedContent`) and card construction (`normalize_card`). For each
EDOS label it takes the top `--top-k` chunks of one embedder+reranker
configuration (default: the best one in the annotation, Qwen3-Embedding-8B +
Qwen3-Reranker-8B) and asks the model for one card per chunk.

Modes
- batch (default): POST /api/v1/batches, poll GET /api/v1/batches/{id} until a final
  status, read the inlined results (~50% of the standard price, up to 24 h).
  The provider is pinned at batch level with {"only": [...], "allow_fallbacks": false};
  the Batch API does not support "order" or "require_parameters", and drops unknown
  body parameters silently, so check the recorded responses after the first run.
  The batch id is saved before polling: rerunning resumes the same batch.
- sync: POST /api/v1/chat/completions, one request at a time (quick tests).

Recorded for reproducibility (output folder `generation/<name>/`):
- manifest.json: git commit and dirty files, sha256 of the code and of the ranking
  file, request parameters (model, provider, seed, reasoning effort, max tokens,
  JSON schema), system prompt, Python and library versions, planned cards;
- batches/<batch_id>.json: the full final batch object (status, provider, usage,
  request_counts, every result);
- attempts.jsonl: per card, the full request body (never the API key), the full raw
  response and, when available, GET /api/v1/generation statistics (provider_name,
  native token counts, total_cost);
- cards.jsonl: validated EvidenceCard records. Accepted cards are skipped on rerun.

Usage, from the src directory (API key OPENROUTER_API in the project .env or in the environment):
    python -m unito_amazon.card_creation.openrouter_generation \\
        --config unito_amazon/card_creation/config.edos-v2-words-clean-v4.json --top-k 1 --dry-run
    (then the same without --dry-run)
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import requests

from .card_schema import EvidenceCard, GeneratedContent
from .config import PROJECT_ROOT, load_config, model_slug
from .generation import normalize_card
from .io import append_jsonl, file_sha256, fingerprint, read_jsonl, write_json
from .prompts import SYSTEM_PROMPT, build_card_prompt
from .schema import RankedPassage


BASE_URL = "https://openrouter.ai/api/v1"
API_KEY_ENV = "OPENROUTER_API"
FINAL_STATUSES = ("completed", "failed", "cancelled", "expired")
CODE_FILES = ("openrouter_generation.py", "generation.py", "prompts.py", "card_schema.py", "passage_text.py", "schema.py")
# Explicit schema for strict structured outputs (no keywords strict mode may reject);
# the response is then validated with GeneratedContent (non-empty strings).
RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {"reasoning": {"type": "string"}, "evidence": {"type": "string"}},
    "required": ["reasoning", "evidence"],
    "additionalProperties": False,
}


def load_api_key() -> str:
    """API key from the environment or from the project's .env (as in __main__.py)."""
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
    key = os.environ.get(API_KEY_ENV)
    if not key:
        sys.exit(f"Set {API_KEY_ENV} in {PROJECT_ROOT / '.env'} or in the environment.")
    return key


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_state(root: Path) -> dict:
    def run(*args):
        return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True).stdout.strip()
    return {"commit": run("rev-parse", "HEAD"), "dirty_files": run("status", "--porcelain").splitlines()}


def select_passages(config, embedding_model: str, reranker_model: str, top_k: int):
    """Top-k reranked passages per label, checked against the frozen run archive."""
    current = config.model_copy(update={
        "embedding_model": embedding_model, "reranker_model": reranker_model, "model_scoped_artifacts": True,
    })
    ranked = [RankedPassage.model_validate(row) for row in read_jsonl(current.ranked_path)]
    selected = sorted((p for p in ranked if p.reranker_rank <= top_k), key=lambda p: (p.label, p.reranker_rank))
    archive = config.output_dir / "runs" / config.run_id / "reranking" / f"{current.pipeline_id}.jsonl"
    if archive.exists():
        frozen = {(row["label"], row["reranker_rank"]): row["chunk"]["chunk_id"] for row in read_jsonl(archive)}
        for passage in selected:
            expected = frozen.get((passage.label, passage.reranker_rank))
            if expected is not None and expected != passage.chunk.chunk_id:
                raise ValueError(f"Work ranking differs from run archive {config.run_id}: "
                                 f"{passage.label} rank {passage.reranker_rank}")
    labels = {p.label for p in ranked}
    if len(selected) != len(labels) * top_k:
        raise ValueError(f"Expected {top_k} passages for each of {len(labels)} labels, found {len(selected)}.")
    return selected, current.ranked_path


def chat_body(args, prompt: str) -> dict:
    """Body of one chat completion. Model and provider are added per mode."""
    body = {
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
        "seed": args.seed,
        "max_completion_tokens": args.max_completion_tokens,
        "response_format": {"type": "json_schema",
                            "json_schema": {"name": "evidence_card", "strict": True, "schema": RESPONSE_SCHEMA}},
    }
    if args.reasoning_effort:
        body["reasoning"] = {"effort": args.reasoning_effort}
    if args.temperature is not None:
        body["temperature"] = args.temperature
    return body


def batch_provider(args) -> dict:
    return {"only": [args.provider], "allow_fallbacks": False}


def sync_provider(args) -> dict:
    return {"order": [args.provider], "allow_fallbacks": False, "require_parameters": True}


def headers(api_key: str) -> dict:
    return {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}


def generation_stats(api_key: str, generation_id: str, attempts: int = 4) -> dict:
    """GET /api/v1/generation?id=...: provider, native token counts and cost.
    Documented for sync responses; for batch results it is tried and the outcome recorded."""
    last_error = None
    for attempt in range(attempts):
        try:
            response = requests.get(f"{BASE_URL}/generation", params={"id": generation_id},
                                    headers=headers(api_key), timeout=30)
            if response.status_code == 200:
                return response.json().get("data", {})
            last_error = f"HTTP {response.status_code}: {response.text[:200]}"
        except requests.RequestException as error:
            last_error = repr(error)
        time.sleep(2 * (attempt + 1))
    return {"error": last_error}


def key_usage(api_key: str) -> dict:
    """GET /api/v1/key: spending of this API key in USD (total, UTC day/week/month) and limit."""
    try:
        response = requests.get(f"{BASE_URL}/key", headers=headers(api_key), timeout=30)
        if response.status_code != 200:
            return {"error": f"HTTP {response.status_code}: {response.text[:200]}"}
        data = response.json().get("data", {})
    except requests.RequestException as error:
        return {"error": repr(error)}
    fields = ("usage", "usage_daily", "usage_weekly", "usage_monthly", "limit", "limit_remaining", "limit_reset")
    return {"checked_at": now(), **{field: data.get(field) for field in fields}}


def print_key_usage(title: str, usage: dict) -> None:
    if "error" in usage:
        print(f"{title}: usage not available ({usage['error']})")
        return
    limit = "no limit" if usage.get("limit") is None else f"limit {usage['limit']} $, remaining {usage.get('limit_remaining')} $"
    print(f"{title}: total {usage.get('usage')} $ · today {usage.get('usage_daily')} $ · "
          f"this month {usage.get('usage_monthly')} $ · {limit}")


def attempt_cost(record: dict) -> float | None:
    """Cost of one call: usage.cost of the response body, else total_cost of the generation stats."""
    body = (record.get("result") or {}).get("response", {}).get("body") or record.get("response") or {}
    cost = (body.get("usage") or {}).get("cost")
    if cost is None:
        cost = (record.get("generation_stats") or {}).get("total_cost")
    return cost


def attempt_tokens(record: dict) -> dict:
    """Input, output and reasoning tokens of one call, from the response usage
    (reasoning tokens are part of the output tokens)."""
    body = (record.get("result") or {}).get("response", {}).get("body") or record.get("response") or {}
    usage = body.get("usage") or {}
    return {
        "input_tokens": usage.get("prompt_tokens"),
        "output_tokens": usage.get("completion_tokens"),
        "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
        "cached_input_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
    }


def cost_report(output_dir: Path, usage_before: dict, usage_after: dict) -> dict:
    """Spending and tokens of this generation folder from the saved attempts, plus the key's usage."""
    attempts = read_jsonl(output_dir / "attempts.jsonl") if (output_dir / "attempts.jsonl").exists() else []
    per_call = [{"card_id": a["card_id"], "label": a["label"], "status": a["status"],
                 "batch_id": a.get("batch_id"), "cost_usd": attempt_cost(a), **attempt_tokens(a)}
                for a in attempts]
    batches = {}
    for path in sorted((output_dir / "batches").glob("*.json")):
        batch = json.loads(path.read_text(encoding="utf-8"))
        usage = batch.get("usage") or {}
        batches[batch.get("id")] = {"cost_usd": usage.get("cost"), "input_tokens": usage.get("prompt_tokens"),
                                    "output_tokens": usage.get("completion_tokens")}
    known = [c["cost_usd"] for c in per_call if c["cost_usd"] is not None]
    totals = {field: sum(c[field] for c in per_call if c[field] is not None)
              for field in ("input_tokens", "output_tokens", "reasoning_tokens", "cached_input_tokens")}
    report = {
        "generated_at": now(),
        "calls": len(per_call), "calls_with_cost": len(known),
        "total_cost_usd_from_calls": round(sum(known), 6),
        "total_tokens_from_calls": totals,
        "batches": batches,
        "key_usage_before_run": usage_before, "key_usage_after_run": usage_after,
        "per_call": per_call,
    }
    if "error" not in usage_before and "error" not in usage_after:
        report["key_usage_increase_during_run_usd"] = round(usage_after["usage"] - usage_before["usage"], 6)
    write_json(output_dir / "costs.json", report)
    return report


def accept(output_dir, job, raw_body: dict, record: dict, pipeline_config) -> tuple[bool, str]:
    """Validate one chat completion body; write the card and the attempt record."""
    key, passage, prompt, _ = job
    try:
        choice = raw_body["choices"][0]
        if choice.get("finish_reason") != "stop":
            raise ValueError(f"finish_reason={choice.get('finish_reason')}")
        content = GeneratedContent.model_validate_json(choice["message"]["content"] or "")
        card = normalize_card(content, passage, pipeline_config, key, prompt)
    except (ValueError, KeyError, IndexError, TypeError) as error:
        append_jsonl(output_dir / "attempts.jsonl", {**record, "status": "invalid", "error": str(error)})
        return False, str(error)
    append_jsonl(output_dir / "cards.jsonl", card.model_dump(mode="json"))
    append_jsonl(output_dir / "attempts.jsonl", {**record, "status": "accepted"})
    return True, ""


def run_batch(args, api_key, jobs, output_dir, pipeline_config):
    state_path = output_dir / "batch_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else None
    by_id = {"card_" + job[0][:24]: job for job in jobs}

    if state and state.get("processed") is False and not set(state["card_ids"]) <= set(by_id):
        # The saved batch was submitted with different requests (prompt or parameters changed):
        # its results must not be turned into cards for the current requests.
        old_id = state["batch_id"]
        batch = requests.get(f"{BASE_URL}/batches/{old_id}", headers=headers(api_key), timeout=60).json()
        if batch.get("status") not in FINAL_STATUSES:
            sys.exit(f"Batch {old_id} was submitted with different requests (prompt or parameters changed) "
                     f"and is still {batch.get('status')}. Rerun when it has finished (it will be archived "
                     f"as is), or use --name to start the new requests in another folder.")
        write_json(output_dir / "batches" / f"{old_id}.json", batch)
        write_json(output_dir / f"batch_state_superseded_{old_id}.json",
                   {**state, "processed": "superseded", "note": "requests changed; batch archived, no cards",
                    "final_status": batch.get("status"), "archived_at": now()})
        print(f"Batch {old_id} ({batch.get('status')}) used different requests: archived in batches/, no cards made.")
        state = None

    if state and state.get("processed") is False:
        batch_id = state["batch_id"]
        print(f"Resuming batch {batch_id} (submitted {state['submitted_at']})")
    else:
        pending = [card_id for card_id in by_id if card_id not in done_cards(output_dir)]
        if not pending:
            print("All cards already generated.")
            return 0
        # Field order matters: endpoint, model, provider, completion_window before requests.
        payload = {
            "endpoint": "/v1/chat/completions",
            "model": args.model,
            "provider": batch_provider(args),
            "completion_window": "24h",
            "requests": [{"custom_id": card_id, "body": by_id[card_id][3]} for card_id in pending],
        }
        response = requests.post(f"{BASE_URL}/batches", headers=headers(api_key), json=payload, timeout=120)
        if response.status_code >= 400:
            sys.exit(f"Batch submission failed: HTTP {response.status_code} {response.text[:500]}")
        created = response.json()
        batch_id = created["id"]
        state = {"batch_id": batch_id, "submitted_at": now(), "card_ids": pending, "processed": False}
        write_json(state_path, state)
        print(f"Batch {batch_id} submitted: {len(pending)} requests; status {created.get('status')}")

    while True:
        batch = requests.get(f"{BASE_URL}/batches/{batch_id}", headers=headers(api_key), timeout=60).json()
        print(f"{now()} status={batch.get('status')} counts={batch.get('request_counts')}")
        if batch.get("status") in FINAL_STATUSES:
            break
        time.sleep(args.poll_seconds)

    write_json(output_dir / "batches" / f"{batch_id}.json", batch)
    results = {item["custom_id"]: item for item in batch.get("results") or []}
    failed = 0
    for card_id in state["card_ids"]:
        job = by_id[card_id]
        _, passage, _, body = job
        record = {"card_id": card_id, "label": passage.label, "chunk_id": passage.chunk.chunk_id,
                  "mode": "batch", "batch_id": batch_id, "batch_status": batch.get("status"),
                  "batch_provider": batch.get("provider"), "recorded_at": now(),
                  "request": {"model": args.model, "provider": batch_provider(args), **body}}
        item = results.get(card_id)
        if item is None or item.get("error") or not item.get("response"):
            append_jsonl(output_dir / "attempts.jsonl", {**record, "status": "request_failed",
                                                        "result": item})
            failed += 1
            continue
        raw_body = item["response"]["body"]
        record["result"] = item
        record["generation_stats"] = generation_stats(api_key, raw_body.get("id", ""))
        ok, error = accept(output_dir, job, raw_body, record, pipeline_config)
        print(f"{passage.label}: {'card saved' if ok else 'invalid output: ' + error}")
        failed += not ok
    write_json(state_path, {**state, "processed": True, "final_status": batch.get("status"),
                            "processed_at": now()})
    return failed


def run_sync(args, api_key, jobs, output_dir, pipeline_config):
    failed = 0
    done = done_cards(output_dir)
    for key, passage, prompt, body in jobs:
        card_id = "card_" + key[:24]
        if card_id in done:
            continue
        request = {"model": args.model, "provider": sync_provider(args), **body}
        for retry in range(args.retries + 1):
            record = {"card_id": card_id, "label": passage.label, "chunk_id": passage.chunk.chunk_id,
                      "mode": "sync", "retry": retry, "requested_at": now(), "request": request}
            started = time.perf_counter()
            response = requests.post(f"{BASE_URL}/chat/completions", headers=headers(api_key),
                                     json=request, timeout=300)
            record["latency_s"] = round(time.perf_counter() - started, 3)
            if response.status_code >= 400:
                append_jsonl(output_dir / "attempts.jsonl", {**record, "status": "api_error",
                                                            "error": f"HTTP {response.status_code} {response.text[:500]}"})
                sys.exit(f"API error on {passage.label}: HTTP {response.status_code}")
            raw_body = response.json()
            record["response"] = raw_body
            record["generation_stats"] = generation_stats(api_key, raw_body.get("id", ""))
            ok, error = accept(output_dir, (key, passage, prompt, body), raw_body, record, pipeline_config)
            print(f"{passage.label}: {'card saved' if ok else 'invalid output: ' + error}")
            if ok:
                break
            if retry == args.retries:
                failed += 1
    return failed


def done_cards(output_dir: Path) -> set:
    path = output_dir / "cards.jsonl"
    return {card["card_id"] for card in read_jsonl(path)} if path.exists() else set()


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", choices=["batch", "sync"], default="batch")
    parser.add_argument("--embedding-model", default="Qwen/Qwen3-Embedding-8B")
    parser.add_argument("--reranker-model", default="Qwen/Qwen3-Reranker-8B")
    parser.add_argument("--top-k", type=int, default=1, help="chunks per EDOS label (1 = one card per label)")
    parser.add_argument("--model", default=None,
                        help="default: openrouter_model in the config (must be listed in openrouter_models)")
    parser.add_argument("--provider", default=None,
                        help="OpenRouter provider slug, pinned without fallback; default: the model's config entry")
    parser.add_argument("--seed", type=int, default=20261112)
    parser.add_argument("--reasoning-effort", default="low",
                        choices=["none", "minimal", "low", "medium", "high", "xhigh", "max", ""])
    parser.add_argument("--temperature", type=float, default=None,
                        help="not sent by default: GPT-6 Luna does not advertise temperature")
    parser.add_argument("--max-completion-tokens", type=int, default=4000)
    parser.add_argument("--retries", type=int, default=2, help="sync mode only")
    parser.add_argument("--poll-seconds", type=int, default=60, help="batch mode: polling interval")
    parser.add_argument("--name", default=None, help="output folder name (default: model, pipeline, top-k)")
    parser.add_argument("--dry-run", action="store_true", help="write prompts and requests, send nothing")
    parser.add_argument("--show-usage", action="store_true",
                        help="only print the current spending of the API key, then exit")
    args = parser.parse_args()

    if args.show_usage:
        print_key_usage("OpenRouter key", key_usage(load_api_key()))
        return

    config = load_config(args.config)
    entries = {entry.model: entry for entry in config.openrouter_models}
    args.model = args.model or config.openrouter_model
    if not args.model:
        sys.exit("No model: set openrouter_model in the config or pass --model.")
    if args.model not in entries:
        sys.exit(f"{args.model} is not listed in openrouter_models in the config.")
    args.provider = args.provider or entries[args.model].provider
    if not args.provider:
        sys.exit(f"No provider for {args.model}: set its \"provider\" in the config or pass --provider.")
    passages, ranking_path = select_passages(config, args.embedding_model, args.reranker_model, args.top_k)
    pipeline_config = config.model_copy(update={
        "embedding_model": args.embedding_model, "reranker_model": args.reranker_model,
        "generator_model": args.model, "model_scoped_artifacts": True,
    })
    name = args.name or f"{model_slug(args.model)}__{pipeline_config.pipeline_id}__top{args.top_k}"
    output_dir = config.output_dir / "generation" / name
    output_dir.mkdir(parents=True, exist_ok=True)

    jobs = []
    for passage in passages:
        prompt = build_card_prompt(passage, config.include_section_titles)
        body = chat_body(args, prompt)
        key = fingerprint({"model": args.model, "provider": args.provider, "body": body,
                           "retrieval": passage.model_dump(mode="json"),
                           "card_schema": EvidenceCard.model_json_schema()})
        jobs.append((key, passage, prompt, body))

    package = Path(__file__).parent
    parameters = {k: v for k, v in chat_body(args, "<prompt>").items() if k != "messages"}
    manifest = {
        "created_at": now(), "name": name, "mode": args.mode, "dry_run": args.dry_run,
        "run_id": config.run_id, "pipeline_id": pipeline_config.pipeline_id,
        "embedding_model": args.embedding_model, "reranker_model": args.reranker_model,
        "top_k_per_label": args.top_k, "n_cards": len(jobs),
        "ranking_file": str(ranking_path), "ranking_file_sha256": file_sha256(ranking_path),
        "chunks_sha256": file_sha256(config.chunks_path),
        "model": args.model,
        "provider_routing": batch_provider(args) if args.mode == "batch" else sync_provider(args),
        "endpoint": f"{BASE_URL}/batches (endpoint /v1/chat/completions, completion_window 24h)"
                    if args.mode == "batch" else f"{BASE_URL}/chat/completions",
        "body_parameters": parameters,
        "system_prompt": SYSTEM_PROMPT,
        "user_prompt_builder": f"prompts.build_card_prompt(include_section_titles={config.include_section_titles})",
        "git": git_state(package.parents[2]),
        "code_sha256": {file: file_sha256(package / file) for file in CODE_FILES},
        "python": platform.python_version(),
        "libraries": {lib: version(lib) for lib in ("requests", "pydantic")},
        "jobs": [{"card_id": "card_" + key[:24], "label": p.label, "chunk_id": p.chunk.chunk_id,
                  "reranker_rank": p.reranker_rank} for key, p, _, _ in jobs],
    }
    write_json(output_dir / ("manifest_dry_run.json" if args.dry_run else "manifest.json"), manifest)
    print(f"Model: {args.model} · provider: {args.provider} (no fallback) · mode: {args.mode} · "
          f"pipeline: {pipeline_config.pipeline_id} · {len(jobs)} cards ({args.top_k} per label)\n"
          f"Output: {output_dir}")

    if args.dry_run:
        write_json(output_dir / "dry_run_requests.json",
                   [{"custom_id": "card_" + key[:24], "body": body} for key, _, _, body in jobs])
        print(f"Dry run ({args.mode}): {len(jobs)} requests written to {output_dir}; nothing was sent.")
        return

    api_key = load_api_key()
    usage_before = key_usage(api_key)
    print_key_usage("Key spending before the run", usage_before)
    runner = run_batch if args.mode == "batch" else run_sync
    failed = runner(args, api_key, jobs, output_dir, pipeline_config)
    usage_after = key_usage(api_key)
    print_key_usage("Key spending after the run", usage_after)
    costs = cost_report(output_dir, usage_before, usage_after)
    tokens = costs["total_tokens_from_calls"]
    print(f"Cost of this folder's calls: {costs['total_cost_usd_from_calls']} $ "
          f"({costs['calls_with_cost']}/{costs['calls']} calls with a reported cost)\n"
          f"Tokens: input {tokens['input_tokens']} (cached {tokens['cached_input_tokens']}) · "
          f"output {tokens['output_tokens']} (of which reasoning {tokens['reasoning_tokens']})\n"
          f"Batches: {costs['batches']}; details per card in {output_dir / 'costs.json'}")
    print(f"Done: {len(done_cards(output_dir))}/{len(jobs)} cards in {output_dir / 'cards.jsonl'}; "
          f"{failed} failed (details in attempts.jsonl).")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
