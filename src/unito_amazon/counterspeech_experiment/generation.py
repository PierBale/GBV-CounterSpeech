"""Counter-speech generation for the four strategies, resumable record by record."""

import gc
import platform
import subprocess
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version

from unito_amazon.card_creation.io import append_jsonl, read_jsonl, write_json
from unito_amazon.generation.context_stats import measure_context
from unito_amazon.generation.prompts import (
    CategoryRAGCounterSpeechPrompt,
    CounterSpeechPrompt,
    RAGCounterSpeechPrompt,
)
from unito_amazon.generation.schema import parse_counter_narrative
from unito_amazon.llm.schema import LLMGenerationConfig

from .config import PROJECT_ROOT, ExperimentConfig
from .data import attach_predicted_labels, label_code, load_or_create_subset
from .retrieval import ExperimentRetrievers


def build_prompts(config: ExperimentConfig) -> dict:
    options = {"language": config.language, "max_sentences": config.max_sentences, "cite_source": False}
    return {
        "zero_shot": CounterSpeechPrompt(**options),
        "rag": RAGCounterSpeechPrompt(**options, evidence_count=config.top_k),
        "card_rag": RAGCounterSpeechPrompt(**options, evidence_count=config.top_k),
        "card_rag_label": CategoryRAGCounterSpeechPrompt(**options, evidence_count=config.top_k),
    }


def plan_requests(config: ExperimentConfig) -> tuple[list[dict], dict]:
    """Retrieve evidence for every message and build every prompt (embedding model only)."""
    items = attach_predicted_labels(load_or_create_subset(config), config)
    prompts = build_prompts(config)
    retrievers = ExperimentRetrievers(config)
    try:
        retrievers.load()
        requests = []
        for position, item in enumerate(items, start=1):
            retrieved = retrievers.retrieve(item["hateful_message"], item["predicted_edos_label"])
            category = retrievers.definitions[label_code(item["predicted_edos_label"])]
            for strategy in config.strategies:
                results = retrieved.get(strategy, [])
                prompt = prompts[strategy]
                if strategy == "zero_shot":
                    user_prompt = prompt.build(item["hateful_message"])
                elif strategy == "card_rag_label":
                    user_prompt = prompt.build(item["hateful_message"], results, category)
                else:
                    user_prompt = prompt.build(item["hateful_message"], results)
                requests.append({
                    "item": item, "strategy": strategy, "system_prompt": prompt.system_prompt,
                    "prompt": user_prompt, "results": results,
                    "category_in_prompt": list(category) if strategy == "card_rag_label" else None,
                })
            if position % 20 == 0 or position == len(items):
                print(f"Retrieval: {position}/{len(items)} messages")
        return requests, retrievers.metadata()
    finally:
        retrievers.close()


def evidence_records(results) -> list[dict]:
    return [{"rank": result.rank, "score": result.score, "id": result.document.id,
             "text": result.document.text, "metadata": result.document.metadata}
            for result in results]


def git_state() -> dict:
    def run(*args):
        return subprocess.run(["git", "-C", str(PROJECT_ROOT), *args], capture_output=True, text=True).stdout.strip()
    return {"commit": run("rev-parse", "HEAD"), "dirty_files": run("status", "--porcelain").splitlines()}


def package_versions(*names: str) -> dict:
    versions = {}
    for name in names:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    return versions


def run_generation(config: ExperimentConfig) -> None:
    requests, retrieval_metadata = plan_requests(config)
    done = {}
    if config.generations_path.exists():
        for record in read_jsonl(config.generations_path):
            done[(record["id"], record["generation_strategy"])] = record
    pending = []
    for request in requests:
        previous = done.get((request["item"]["id"], request["strategy"]))
        if previous is None:
            pending.append(request)
        elif (previous["prompt"], previous["system_prompt"]) != (request["prompt"], request["system_prompt"]):
            raise ValueError(f"{config.generations_path} was generated with different prompts "
                             f"({request['item']['id']}, {request['strategy']}); use another output_dir.")
    generation_config = {"max_new_tokens": config.max_new_tokens, "do_sample": False, "repetition_penalty": 1.0}
    write_json(config.output_dir / "manifest_generation.json", {
        "created_at": datetime.now(timezone.utc).isoformat(), "config": config.as_dict(),
        "retrieval": retrieval_metadata, "generation_config": generation_config,
        "system_prompts": {strategy: prompt.system_prompt for strategy, prompt in build_prompts(config).items()},
        "requests": len(requests), "already_done": len(requests) - len(pending),
        "git": git_state(), "python": platform.python_version(),
        "libraries": package_versions("torch", "transformers", "sentence-transformers", "faiss-cpu", "faiss-gpu"),
    })
    print(f"{len(requests)} generations planned, {len(requests) - len(pending)} already done.")
    if not pending:
        return

    from unito_amazon.llm.factory import LLMFactory

    llm = LLMFactory.create({"provider": "huggingface", "model_name": config.generator_model,
                             "device": config.device, "dtype": config.generator_dtype})
    tokenizer = getattr(llm.tokenizer, "tokenizer", llm.tokenizer)
    try:
        for strategy in config.strategies:
            todo = [request for request in pending if request["strategy"] == strategy]
            for start in range(0, len(todo), config.generation_batch_size):
                batch = todo[start:start + config.generation_batch_size]
                llm_config = LLMGenerationConfig(**generation_config, system_prompt=batch[0]["system_prompt"])
                response = llm.generate_batch([request["prompt"] for request in batch], config=llm_config)
                for request, raw in zip(batch, response.texts, strict=True):
                    text, valid = parse_counter_narrative(raw)
                    item = request["item"]
                    append_jsonl(config.generations_path, {
                        "id": item["id"], "source": item["source"], "split": item["split"],
                        "target": item["target"], "edos_label": item["gold_edos_label"],
                        "predicted_edos_label": item["predicted_edos_label"], "classifier": item["classifier"],
                        "hateful_message": item["hateful_message"],
                        "reference_counter_speech": item["reference_counter_speech"],
                        "generation_strategy": strategy, "generation_model": config.generator_model,
                        "generated_counter_speech": text, "response_format_valid": valid, "raw_response": raw,
                        "system_prompt": request["system_prompt"], "prompt": request["prompt"],
                        "category_in_prompt": request["category_in_prompt"],
                        "evidence": evidence_records(request["results"]),
                        "context_length": measure_context(request["prompt"], len(request["results"]),
                                                          tokenizer, config.generator_model),
                        "generation_config": generation_config,
                        "generated_at": datetime.now(timezone.utc).isoformat(),
                    })
                print(f"[{strategy}] {min(start + len(batch), len(todo))}/{len(todo)} generated")
    finally:
        llm.destroy()
        gc.collect()
    print(f"Generations written to {config.generations_path}")
