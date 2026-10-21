"""Content-checked stage checkpoints with dependency-aware invalidation.

Only explicitly registered generated files inside output_dir may be removed.
PDFs, Markdown inputs and arbitrary paths from manifests are never deleted.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .config import CardCreationConfig
from .io import file_sha256, fingerprint, read_jsonl, write_json
from .progress import report


DEPENDENCIES = {
    "chunk": (), "index": ("chunk",), "dense": ("index",),
    "rerank": ("dense",), "dedup": ("rerank", "index"),
    "generate": ("dedup",), "index-cards": ("generate",),
}
CONFIG_FIELDS = {
    "parse": ("pdf_dir", "markdown_dir", "llama_base_url", "llama_tier", "llama_version"),
    "chunk": ("markdown_dir", "source_metadata_path", "min_chunk_tokens", "chunk_tokenizer_model", "editorial_filter_enabled"),
    "index": ("embedding_model", "embedding_max_tokens", "include_section_titles"),
    "dense": ("labels_path", "queries_path", "dense_top_j", "embedding_model", "embedding_max_tokens", "include_section_titles"),
    "rerank": ("reranker_model", "reranker_max_tokens", "include_section_titles"),
    "dedup": ("rerank_top_k", "deduplication_enabled", "deduplication_similarity_threshold"),
    "generate": ("generator_model", "generation_max_new_tokens", "generation_temperature", "include_section_titles"),
    "index-cards": ("embedding_model", "embedding_max_tokens"),
}
CODE_FILES = {
    "parse": ("parsing.py",), "chunk": ("chunking.py", "schema.py", "editorial.py"),
    "index": ("vector_store.py", "passage_text.py", "editorial.py", "retrieval.py"),
    "dense": ("retrieval.py", "vector_store.py", "queries.py", "schema.py", "passage_text.py", "editorial.py"),
    "rerank": ("retrieval.py", "reranking.py", "../retrieval/reranker.py", "passage_text.py", "editorial.py"),
    "dedup": ("retrieval.py", "deduplication.py"),
    "generate": ("generation.py", "prompts.py", "schema.py", "card_schema.py", "passage_text.py", "editorial.py"),
    "index-cards": ("card_retrieval.py", "vector_store.py", "schema.py", "card_schema.py"),
}


def file_state(path: Path) -> dict:
    path = path.resolve()
    if not path.is_file():
        return {"path": str(path), "exists": False}
    return {"path": str(path), "exists": True, "sha256": file_sha256(path)}


def snapshot_inputs(descriptions: list[dict]) -> list[dict]:
    snapshots = []
    for description in descriptions:
        path = Path(description["path"])
        if description["kind"] == "file":
            state = file_state(path)
        else:
            paths = sorted({
                item for pattern in description["patterns"]
                for item in path.rglob(pattern) if item.is_file()
            })
            state = {"exists": path.is_dir(), "files": [file_state(item) for item in paths]}
        snapshots.append({"description": description, "state": state})
    return snapshots


@dataclass(frozen=True)
class CacheTicket:
    stage: str
    signature: str
    inputs: list[dict]
    baseline: list[dict]
    mode: Literal["cached", "resume", "run"]


class StageCache:
    def __init__(self, config: CardCreationConfig):
        self.config = config
        self.root = config.output_dir.resolve()
        self.directory = config.cache_dir
        self.outputs = {
            "chunk": [config.chunks_path, config.editorial_audit_path],
            "index": [config.chunk_index_path, config.chunk_index_path.with_suffix(".faiss.json")],
            "dense": [config.dense_path], "rerank": [config.ranked_path],
            "dedup": [config.retrieved_path, config.deduplication_log_path],
            "generate": [config.cards_path, config.attempts_path, config.generation_manifest_path],
            "index-cards": [config.card_index_path, config.card_index_path.with_suffix(".faiss.json")],
        }
        values = config.model_dump(mode="json")
        values["chunk_tokenizer_model"] = config.chunk_tokenizer
        package = Path(__file__).parent
        self.settings = {
            stage: {
                "parameters": {field: values[field] for field in fields},
                "code": {name: file_sha256(package / name) for name in CODE_FILES[stage]},
            }
            for stage, fields in CONFIG_FIELDS.items()
        }
        self._validated: set[str] = set()
        if config.model_scoped_artifacts:
            # Each stage validates its own settings and upstream signatures lazily.
            # Switching models must not erase another combination's checkpoints.
            return
        snapshot_path = self.directory / "configuration.json"
        previous = self.read_manifest(snapshot_path)
        if previous:
            old_settings = previous.get("settings", {})
            invalidated = set()
            for stage, settings in self.settings.items():
                if stage in old_settings and old_settings[stage] != settings:
                    root_stage = "chunk" if stage == "parse" else stage
                    if root_stage not in invalidated:
                        report("cache", f"Configurazione o codice cambiati: {stage}")
                        invalidated.update(self.invalidate(root_stage))
        write_json(snapshot_path, {"version": 1, "settings": self.settings})

    @staticmethod
    def read_manifest(path: Path) -> dict | None:
        if not path.exists():
            return None
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            return value if isinstance(value, dict) else None
        except (ValueError, OSError):
            report("cache", f"Manifest non leggibile, cache da verificare: {path}")
            return None

    def _manifest_path(self, stage: str) -> Path:
        if stage not in DEPENDENCIES:
            raise ValueError(f"Unknown cache stage: {stage}")
        if self.config.model_scoped_artifacts:
            if stage == "chunk":
                directory = self.directory / "shared"
            elif stage in {"index", "dense"}:
                directory = self.config.retrieval_work_dir / "checkpoints"
            else:
                directory = self.config.pipeline_work_dir / "checkpoints"
            return directory / f"{stage}.json"
        return self.directory / f"{stage}.json"

    def _output_states(self, stage: str) -> list[dict]:
        return [file_state(path) for path in self.outputs[stage]]

    def _remove(self, paths: list[Path]) -> None:
        # Validate every target before removing any file. Never use manifest
        # contents as deletion targets and never recursively delete directories.
        for path in paths:
            resolved = path.resolve()
            if resolved == self.root or not resolved.is_relative_to(self.root):
                raise ValueError(f"Refusing to remove a path outside output_dir: {path}")
            if path.exists() and not path.is_file():
                raise ValueError(f"Refusing to remove a directory: {path}")
        for path in paths:
            if path.exists():
                path.unlink()
                report("cache", f"Artefatto obsoleto rimosso: {path}")

    def invalidate(self, stage: str, include_self: bool = True) -> set[str]:
        affected = {stage}
        for candidate, parents in DEPENDENCIES.items():
            if any(parent in affected for parent in parents):
                affected.add(candidate)
        if not include_self:
            affected.discard(stage)
        paths = [
            path for candidate in DEPENDENCIES if candidate in affected
            for path in [*self.outputs[candidate], self._manifest_path(candidate)]
        ]
        self._remove(paths)
        self._validated.difference_update(affected)
        if affected:
            report("cache", "Stadi invalidati: " + ", ".join(item for item in DEPENDENCIES if item in affected))
        return affected

    def _validate_dependencies(self, stage: str) -> None:
        for parent in DEPENDENCIES[stage]:
            if parent in self._validated:
                continue
            self._validate_dependencies(parent)
            state = self.read_manifest(self._manifest_path(parent))
            if state:
                if state.get("status") != "complete":
                    raise ValueError(f"Stadio {parent} incompleto; rieseguire all (o retrieve per dense/rerank/dedup).")
                descriptions = [item["description"] for item in state.get("inputs", [])]
                current_inputs = snapshot_inputs(descriptions)
                expected_signature = fingerprint({
                    "version": 1, "settings": self.settings[parent], "inputs": current_inputs,
                })
                if current_inputs != state.get("inputs") or state.get("signature") != expected_signature:
                    self.invalidate(parent)
                    raise ValueError(f"Input/output dello stadio {parent} cambiati; rieseguire all.")
                outputs = self._output_states(parent)
                if outputs != state.get("outputs"):
                    # A card library is editable: validate and preserve manual edits.
                    old = state.get("outputs", [])
                    changed = {
                        item["path"] for item, prior in zip(outputs, old, strict=True)
                        if item != prior
                    }
                    if parent == "generate" and changed == {str(self.config.cards_path.resolve())}:
                        from .card_schema import EvidenceCard

                        cards = [EvidenceCard.model_validate(row) for row in read_jsonl(self.config.cards_path)]
                        keys = {card.card_id for card in cards}
                        accepted = {
                            row["card_id"] for row in read_jsonl(self.config.attempts_path)
                            if row.get("status") == "accepted"
                        }
                        if len(keys) != len(cards) or not accepted.issubset(keys):
                            raise ValueError("Edited card library is inconsistent; original files preserved.")
                        self.invalidate(parent, include_self=False)
                        state["outputs"] = outputs
                        write_json(self._manifest_path(parent), state)
                        report("cache", "Carte modificate e rivalidate; revisione manuale conservata")
                    else:
                        self.invalidate(parent)
                        raise ValueError(f"Output dello stadio {parent} cambiati; rieseguire all.")
                self._validated.add(parent)

    def prepare(
        self, stage: str, files: list[Path] | None = None,
        directories: list[tuple[Path, tuple[str, ...]]] | None = None,
        force: bool = False, allow_partial: bool = False, preserve_existing: bool = False,
    ) -> CacheTicket:
        self._validate_dependencies(stage)
        descriptions = [
            {"kind": "file", "path": str(path.resolve())} for path in files or []
        ] + [
            {"kind": "directory", "path": str(path.resolve()), "patterns": list(patterns)}
            for path, patterns in directories or []
        ]
        inputs = snapshot_inputs(descriptions)
        signature = fingerprint({"version": 1, "settings": self.settings[stage], "inputs": inputs})
        state = self.read_manifest(self._manifest_path(stage))
        baseline = self._output_states(stage)
        if force or (state and state.get("signature") != signature):
            report("cache", f"{stage}: input/configurazione cambiati o ricostruzione forzata")
            self.invalidate(stage)
            state = None
        elif state and state.get("status") == "complete":
            if all(item["exists"] for item in baseline) and baseline == state.get("outputs"):
                report("cache", f"{stage}: cache valida, stadio saltato; nessun modello caricato")
                self._validated.add(stage)
                return CacheTicket(stage, signature, inputs, baseline, "cached")
            report("cache", f"{stage}: output mancanti o modificati, verifica/ricostruzione")
            self.invalidate(stage, include_self=not preserve_existing)
            state = None
        elif state and allow_partial and state.get("status") == "running":
            report("cache", f"{stage}: ripresa del checkpoint parziale")
            self._validated.discard(stage)
            return CacheTicket(stage, signature, inputs, baseline, "resume")
        if not preserve_existing:
            self._remove(self.outputs[stage])
        if state is None and any(item["exists"] for item in baseline):
            report("cache", f"{stage}: artefatti senza checkpoint compatibile, verifica/ricostruzione iniziale")
        self._validated.discard(stage)
        ticket = CacheTicket(stage, signature, inputs, baseline, "run")
        self.checkpoint(ticket)
        return ticket

    def checkpoint(self, ticket: CacheTicket, complete: bool = False) -> None:
        outputs = self._output_states(ticket.stage)
        if complete and not all(item["exists"] for item in outputs):
            raise ValueError(f"Missing output while completing stage {ticket.stage}.")
        if complete and outputs != ticket.baseline:
            self.invalidate(ticket.stage, include_self=False)
        write_json(self._manifest_path(ticket.stage), {
            "version": 1, "status": "complete" if complete else "running",
            "signature": ticket.signature, "inputs": ticket.inputs, "outputs": outputs,
        })
        if complete:
            self._validated.add(ticket.stage)
            report("cache", f"{ticket.stage}: checkpoint completato e salvato")
