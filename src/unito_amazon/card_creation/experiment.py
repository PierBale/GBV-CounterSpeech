"""Orchestrate the existing card stages with a common corpus and scoped caches."""

from itertools import product

from .artifacts import RunArtifacts
from .cache import StageCache
from .chunking import create_chunks
from .config import CardCreationConfig, EMBEDDING_MODELS, RERANKING_MODELS
from .progress import report
from .queries import load_queries
from .retrieval import index_chunks, retrieve_passages


def run_experiment(config: CardCreationConfig, force_reindex: bool = False, overwrite: bool = False):
    # Freeze the tokenizer before changing embedding models; chunk exactly once.
    common = config.model_copy(update={
        "chunk_tokenizer_model": config.chunk_tokenizer,
        "model_scoped_artifacts": True,
        "deduplication_enabled": False,
    })
    combinations = list(product(EMBEDDING_MODELS, RERANKING_MODELS)) if config.all_combinations else [
        (config.embedding_model, config.reranker_model)
    ]
    queries = load_queries(common)
    create_chunks(common, cache=StageCache(common))
    artifacts = RunArtifacts(common, queries, combinations, include_evidence=True)
    report("experiment", f"Run {artifacts.run_id}: {len(combinations)} combinazioni; archivio {artifacts.root}")
    indexed = set()
    configurations = []
    current = common
    stage = "index"
    try:
        for embedding, reranker in combinations:
            current = common.model_copy(update={
                "embedding_model": embedding, "reranker_model": reranker, "run_id": artifacts.run_id,
            })
            cache = StageCache(current)
            report("experiment", f"Pipeline {current.pipeline_id}")
            if embedding not in indexed:
                stage = "index"
                index_chunks(current, force=force_reindex, cache=cache)
                indexed.add(embedding)
            stage = "retrieve"
            retrieve_passages(current, cache=cache, artifacts=artifacts)
            configurations.append(current)
        if common.generate_evidence:
            from .generation import generate_cards

            for current in configurations:
                stage = "evidence"
                generate_cards(current, overwrite=overwrite, cache=StageCache(current))
                artifacts.evidence(current)
        artifacts.finish()
    except (Exception, KeyboardInterrupt) as error:
        if artifacts.manifest["status"] == "running":
            artifacts.fail(stage, current, error)
        raise
    report("experiment", f"Esperimento completato: {artifacts.manifest_path}")
    return artifacts.root
