"""Run archives, independent of mutable work files and LLM generation."""

from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from uuid import uuid4

from .config import CardCreationConfig, model_slug
from .io import append_jsonl, file_sha256, fingerprint, read_jsonl, write_json, write_jsonl
from .schema import DensePassage, RetrievalQuery
from .editorial import EDITORIAL_RULES_VERSION
from .passage_text import format_chunk_text


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


class RunArtifacts:
    def __init__(
        self, config: CardCreationConfig, queries: list[RetrievalQuery],
        combinations: list[tuple[str, str]], include_evidence: bool = False,
    ):
        self.run_id = config.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_") + uuid4().hex[:12]
        self.root = config.output_dir / "runs" / self.run_id
        self.manifest_path = self.root / "manifest.json"
        self._last_error = None
        self.include_evidence = include_evidence and config.generate_evidence
        self.combinations = combinations
        self.queries = queries
        configuration = config.model_dump(mode="json")
        # Operational settings may change on resume without altering the experiment.
        semantic = {key: value for key, value in configuration.items() if key not in {
            "run_id", "device", "embedding_batch_size", "reranker_batch_size",
            "generation_retries", "long_chunk_warning_tokens",
        }}
        inputs = {
            "corpus_sha256": file_sha256(config.chunks_path) if config.chunks_path.exists() else None,
            "editorial_audit_sha256": file_sha256(config.editorial_audit_path) if config.editorial_audit_path.exists() else None,
            "queries": [query.model_dump(mode="json") for query in queries],
            "combinations": combinations, "configuration": semantic,
            "include_evidence": self.include_evidence,
            "orchestration_code": {
                name: file_sha256(Path(__file__).with_name(name))
                for name in ("artifacts.py", "experiment.py", "config.py", "cache.py")
            },
        }
        from .cache import StageCache
        from .vector_store import CHUNK_INSTRUCTION
        from .reranking import RERANK_INSTRUCTION

        inputs["stage_settings"] = StageCache(config).settings
        signature = fingerprint(inputs)
        if self.manifest_path.exists():
            import json

            self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            if self.manifest.get("signature") != signature:
                raise ValueError(f"Run {self.run_id} has different inputs/settings; choose a new run_id.")
        else:
            self.manifest = {
                "run_id": self.run_id, "signature": signature, "created_at": timestamp(),
                "inputs": inputs, "retrieval": {}, "reranking": {}, "evidence": {},
            }
        self.manifest.update(status="running", configuration=configuration)
        libraries = {}
        for package in ("torch", "transformers", "sentence-transformers", "faiss-cpu", "numpy", "pydantic"):
            try:
                libraries[package] = version(package)
            except PackageNotFoundError:
                libraries[package] = None
        self.manifest["libraries"] = libraries
        self.manifest["encoding"] = {
            "retrieval": "dense cosine (unit-normalized vectors, FAISS IndexFlatIP)",
            "qwen_query_instruction": CHUNK_INSTRUCTION,
            "qwen_rerank_instruction": RERANK_INSTRUCTION,
            "bge_m3_query_instruction": None,
            "document_instruction": None, "truncation": False,
            "chunk_tokenizer": config.chunk_tokenizer,
            "include_section_titles": config.include_section_titles,
            "passage_format": "Section: <heading 1> > <heading 2>\n\n<body>" if config.include_section_titles else "body",
        }
        if config.editorial_audit_path.exists():
            audit_path = self.root / "chunking" / "editorial_exclusions.jsonl"
            decisions = read_jsonl(config.editorial_audit_path)
            write_jsonl(audit_path, decisions)
            self.manifest["chunking"] = {
                "editorial_filter_enabled": config.editorial_filter_enabled,
                "rules_version": EDITORIAL_RULES_VERSION,
                "audit_path": str(audit_path.relative_to(self.root)),
                "audit_sha256": file_sha256(audit_path), "audit_records": len(decisions),
                "below_minimum_chunks": sum(row["rule_id"] == "below_minimum_tokens" for row in decisions),
                "minimum_body_tokens": config.min_chunk_tokens,
            }
        self.save()

    def save(self):
        self.manifest["updated_at"] = timestamp()
        write_json(self.manifest_path, self.manifest)

    def _retrieval_record(self, config, passage):
        chunk = passage.chunk
        text = format_chunk_text(chunk, config.include_section_titles)
        body = chunk.body_text if chunk.body_text is not None else chunk.text
        # Keep the original nested schema as well as the flat interchange fields.
        original = DensePassage.model_validate({
            key: value for key, value in passage.model_dump(mode="json").items()
            if key in DensePassage.model_fields
        }).model_dump(mode="json")
        # Legacy input libraries may have body-only text; exported text always
        # agrees with the actual passage supplied to the model.
        original["chunk"]["text"] = text
        original["chunk"]["body_text"] = body
        return {
            **original,
            "run_id": self.run_id, "query_id": passage.query_id,
            "query_text": passage.query_text, "edos_label": passage.label,
            "chunk_id": chunk.chunk_id, "text": text, "chunk_text": text,
            "body_text": body, "model_input_text": text,
            "document_id": chunk.source.source_id,
            "document_source": chunk.source.pdf_file or chunk.source.markdown_file,
            "retrieval_rank": passage.dense_rank, "retrieval_score": passage.dense_score,
            "retrieval_top_k": config.retrieval_top_k,
            "metadata": {
                "source": chunk.source.model_dump(mode="json"),
                **chunk.model_dump(mode="json", exclude={"text", "body_text", "source"}),
                "query_metadata": passage.query_metadata,
            },
        }

    def _state(self, path, count, config, stage):
        from .cache import StageCache

        settings = StageCache(config).settings[stage]
        return {
            "status": "complete", "path": str(path.relative_to(self.root)),
            "records": count, "sha256": file_sha256(path), "settings": settings,
            "configuration": config.model_dump(mode="json"),
        }

    def retrieval(self, config, passages):
        records = [self._retrieval_record(config, passage) for passage in passages]
        path = self.root / "retrieval" / f"{model_slug(config.embedding_model)}.jsonl"
        write_jsonl(path, records)
        self.manifest["retrieval"][config.embedding_model] = self._state(path, len(records), config, "dense")
        self.save()

    def reranking(self, config, passages):
        records = [
            {
                **self._retrieval_record(config, passage),
                "reranker_model": passage.reranker_model,
                "reranker_rank": passage.reranker_rank, "reranker_score": passage.reranker_score,
                "reranking_model": passage.reranker_model,
                "rerank_rank": passage.reranker_rank, "rerank_score": passage.reranker_score,
                "rerank_top_k": config.rerank_top_k, "pipeline_id": config.pipeline_id,
            }
            for passage in passages
        ]
        full_path = self.root / "reranking_full" / f"{config.pipeline_id}.jsonl"
        path = self.root / "reranking" / f"{config.pipeline_id}.jsonl"
        selected = [record for record in records if record["rerank_rank"] <= config.rerank_top_k]
        write_jsonl(full_path, records)
        write_jsonl(path, selected)
        self.manifest["reranking"][config.pipeline_id] = {
            **self._state(path, len(selected), config, "rerank"),
            "full_path": str(full_path.relative_to(self.root)),
            "full_records": len(records), "full_sha256": file_sha256(full_path),
            "score_type": "raw_logit" if config.reranker_model == "BAAI/bge-reranker-v2-m3" else "qwen_yes_probability",
        }
        self.save()

    def evidence(self, config):
        records = [
            {**record, "run_id": self.run_id, "pipeline_id": config.pipeline_id}
            for record in read_jsonl(config.cards_path)
        ]
        path = self.root / "evidence" / f"{config.pipeline_id}.jsonl"
        write_jsonl(path, records)
        attempts = self.root / "evidence" / f"{config.pipeline_id}__attempts.jsonl"
        write_jsonl(attempts, (
            {**record, "run_id": self.run_id, "pipeline_id": config.pipeline_id}
            for record in read_jsonl(config.attempts_path)
        ))
        self.manifest["evidence"][config.pipeline_id] = self._state(path, len(records), config, "generate")
        self.save()

    def error(self, stage, config, error, query=None):
        self._last_error = error
        append_jsonl(self.root / "errors.jsonl", {
            "run_id": self.run_id, "timestamp": timestamp(), "stage": stage,
            "embedding_model": config.embedding_model, "reranking_model": config.reranker_model,
            "pipeline_id": config.pipeline_id,
            "query_id": query.query_id if query else None,
            "query_text": query.query_text if query else None,
            "edos_label": query.edos_label if query else None,
            "error_type": type(error).__name__, "error": str(error),
        })

    def fail(self, stage, config, error):
        if self._last_error is not error:
            self.error(stage, config, error)
        self.manifest.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed", failed_stage=stage)
        self.save()

    def finish(self):
        for embedding, reranker in self.combinations:
            pipeline_id = f"{model_slug(embedding)}__{model_slug(reranker)}"
            if embedding not in self.manifest["retrieval"] or pipeline_id not in self.manifest["reranking"]:
                raise ValueError("Cannot complete a run with missing retrieval/reranking outputs.")
            if self.include_evidence and pipeline_id not in self.manifest["evidence"]:
                raise ValueError("Cannot complete a run with missing evidence outputs.")
        self.manifest.pop("failed_stage", None)
        self.manifest["status"] = "complete"
        self.save()
