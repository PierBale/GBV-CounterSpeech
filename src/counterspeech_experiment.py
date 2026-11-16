"""Counter-speech experiment: zero-shot, RAG, card-RAG, card-RAG + predicted EDOS label.

Run from src:
  python counterspeech_experiment.py sample     # 100 messages per dataset (no GPU)
  python counterspeech_experiment.py generate   # retrieval + Gemma 4 E4B (GPU)
  python counterspeech_experiment.py judge      # JudgeLM, all pairs, both orders (GPU)
  python counterspeech_experiment.py summary    # recompute judge_summary.json only

The EDOS Task C labels must exist first, see run_counterspeech_experiment.sh.
"""

import argparse
import dataclasses
from pathlib import Path

from unito_amazon.counterspeech_experiment.config import ExperimentConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("stage", choices=["sample", "generate", "judge", "summary"])
    parser.add_argument("--output-dir", type=Path, help="default: data/counterspeech_v2")
    parser.add_argument("--device", default=None)
    parser.add_argument("--generation-batch-size", type=int)
    parser.add_argument("--judge-batch-size", type=int)
    args = parser.parse_args()

    overrides = {"device": args.device}
    if args.output_dir:
        overrides["output_dir"] = args.output_dir.resolve()
    if args.generation_batch_size:
        overrides["generation_batch_size"] = args.generation_batch_size
    if args.judge_batch_size:
        overrides["judge_batch_size"] = args.judge_batch_size
    config = dataclasses.replace(ExperimentConfig(), **overrides)

    if args.stage == "sample":
        from unito_amazon.counterspeech_experiment.data import load_or_create_subset

        load_or_create_subset(config)
    elif args.stage == "generate":
        from unito_amazon.counterspeech_experiment.generation import run_generation

        run_generation(config)
    elif args.stage == "judge":
        from unito_amazon.counterspeech_experiment.judging import run_judge

        run_judge(config)
    else:
        from unito_amazon.card_creation.io import write_json
        from unito_amazon.counterspeech_experiment.judging import summarize

        write_json(config.output_dir / "judge_summary.json", summarize(config))


if __name__ == "__main__":
    main()
