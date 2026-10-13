"""Run from src: python -m unito_amazon.card_creation <stage>."""

import argparse
from pathlib import Path
from time import perf_counter

from .config import PROJECT_ROOT, load_config
from .progress import report, stage_progress


def main() -> None:
    parser = argparse.ArgumentParser(description="LlamaParse -> EDOS evidence cards -> card FAISS index")
    parser.add_argument(
        "stage", choices=("parse", "chunk", "index", "retrieve", "generate", "index-cards", "all"),
    )
    parser.add_argument("--config", type=Path, help="JSON configuration; relative paths use the repo root")
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Reparse existing PDFs and/or replace generation output for the selected stage",
    )
    parser.add_argument("--force-reindex", action="store_true", help="Rebuild FAISS indexes")
    args = parser.parse_args()
    from dotenv import load_dotenv

    started = perf_counter()
    report("pipeline", f"Caricamento configurazione: {args.config or 'default'}")
    load_dotenv(PROJECT_ROOT / ".env")
    config = load_config(args.config)
    all_stages = ("parse", "chunk", "index", "retrieve", "generate", "index-cards")
    stages = all_stages if args.stage == "all" else (args.stage,)
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

                create_chunks(config)
            elif stage == "index":
                from .retrieval import index_chunks

                index_chunks(config, force=args.force_reindex)
            elif stage == "retrieve":
                from .retrieval import retrieve_passages

                retrieve_passages(config)
            elif stage == "generate":
                from .generation import generate_cards

                generate_cards(config, overwrite=args.overwrite)
            elif stage == "index-cards":
                from .card_retrieval import index_cards

                index_cards(config, force=args.force_reindex)

    report("pipeline", f"Elaborazione completata in {perf_counter() - started:.1f}s")


if __name__ == "__main__":
    main()
