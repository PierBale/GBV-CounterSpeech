"""Run from src: python -m unito_amazon.card_creation <stage>."""

import argparse
from pathlib import Path
from time import perf_counter

from .config import PROJECT_ROOT, load_config
from .cache import StageCache
from .progress import report, stage_progress


def main() -> None:
    parser = argparse.ArgumentParser(description="LlamaParse -> EDOS evidence cards -> card FAISS index")
    parser.add_argument(
        "stage", choices=("parse", "chunk", "index", "retrieve", "generate", "index-cards", "all", "experiment"),
    )
    parser.add_argument("--config", type=Path, help="JSON configuration; relative paths use the repo root")
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Reparse existing PDFs and/or replace generation output for the selected stage",
    )
    parser.add_argument("--force-reindex", action="store_true", help="Rebuild FAISS indexes")
    parser.add_argument("--embedding-model")
    parser.add_argument("--reranking-model", "--reranker-model", dest="reranker_model")
    parser.add_argument("--retrieval-top-k", type=int)
    parser.add_argument("--rerank-top-k", type=int)
    parser.add_argument("--generate-evidence", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--evidence-model", "--generator-model", dest="generator_model")
    parser.add_argument("--all-combinations", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--run-id")
    args = parser.parse_args()
    from dotenv import load_dotenv

    started = perf_counter()
    report("pipeline", f"Caricamento configurazione: {args.config or 'default'}")
    load_dotenv(PROJECT_ROOT / ".env")
    config = load_config(args.config)
    updates = {
        name: getattr(args, name) for name in (
            "embedding_model", "reranker_model", "rerank_top_k", "generate_evidence",
            "generator_model", "all_combinations", "run_id",
        ) if getattr(args, name) is not None
    }
    if args.retrieval_top_k is not None:
        updates["dense_top_j"] = args.retrieval_top_k
    config = type(config).model_validate({**config.model_dump(), **updates})
    if config.all_combinations and args.stage not in {"all", "experiment", "retrieve"}:
        parser.error("--all-combinations is supported with experiment, retrieve or all.")
    if args.stage == "experiment" or config.all_combinations:
        if args.stage == "all":
            from .parsing import parse_pdfs

            parse_pdfs(config, overwrite=args.overwrite)
        from .experiment import run_experiment

        run_experiment(config, force_reindex=args.force_reindex, overwrite=args.overwrite)
        return
    cache = StageCache(config)
    all_stages = ("parse", "chunk", "index", "retrieve", "generate", "index-cards")
    stages = all_stages if args.stage == "all" else (args.stage,)
    if not config.generate_evidence:
        stages = tuple(stage for stage in stages if stage not in {"generate", "index-cards"})
    report("pipeline", f"Stadi da eseguire: {', '.join(stages)}")
    report("pipeline", f"Directory Markdown: {config.markdown_dir}")
    report("pipeline", f"Directory output: {config.output_dir}")

    for position, stage in enumerate(stages, start=1):
        with stage_progress(stage, position, len(stages)):
            if stage == "parse":
                from .parsing import parse_pdfs

                parse_pdfs(config, overwrite=args.overwrite)
            elif stage == "chunk":
                from .chunking import create_chunks

                create_chunks(config, cache=cache)
            elif stage == "index":
                from .retrieval import index_chunks

                index_chunks(config, force=args.force_reindex, cache=cache)
            elif stage == "retrieve":
                from .retrieval import retrieve_passages

                retrieve_passages(config, cache=cache)
            elif stage == "generate":
                from .generation import generate_cards

                generate_cards(config, overwrite=args.overwrite, cache=cache)
            elif stage == "index-cards":
                from .card_retrieval import index_cards

                index_cards(config, force=args.force_reindex, cache=cache)

    report("pipeline", f"Elaborazione completata in {perf_counter() - started:.1f}s")


if __name__ == "__main__":
    main()
