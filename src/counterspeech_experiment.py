"""Counter-speech experiment: zero-shot, RAG, card-RAG and hybrid RAG (3 cards + 3 chunks); the three
retrieval systems also with the EDOS label (predicted, and gold on EDOS) in the prompt and as a retrieval filter.

Run from src:
  python counterspeech_experiment.py sample     # all unique messages: EDOS test 970, MultiCONAN 560 (no GPU)
  python counterspeech_experiment.py generate   # retrieval + Gemma 4 E4B (GPU); 10 systems on EDOS, 7 on MultiCONAN
  python counterspeech_experiment.py judge      # JudgeLM, config.JUDGED_PAIRS, both orders (GPU)
  python counterspeech_experiment.py metrics    # MultiCONAN reference-based metrics (BERTScore: GPU)
  python counterspeech_experiment.py summary    # recompute judge_summary.json only

The EDOS Task C labels must exist first, see run_counterspeech_experiment.sh.
"""

import argparse
import dataclasses
from pathlib import Path

from unito_amazon.counterspeech_experiment.config import ExperimentConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("stage", choices=["sample", "generate", "judge", "metrics", "summary"])
    parser.add_argument("--output-dir", type=Path, help="default: data/counterspeech_v2")
    parser.add_argument("--device", default=None)
    parser.add_argument("--generation-batch-size", type=int)
    parser.add_argument("--judge-batch-size", type=int)
    parser.add_argument("--sample-size", default=None,
                        help="messages per dataset, or 'all' (default); a larger sample extends the saved one")
    args = parser.parse_args()

    overrides = {"device": args.device}
    if args.sample_size:
        overrides["sample_size"] = None if args.sample_size == "all" else int(args.sample_size)
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
    elif args.stage == "metrics":
        from unito_amazon.counterspeech_experiment.metrics import run_metrics

        run_metrics(config)
    else:
        from unito_amazon.card_creation.io import write_json
        from unito_amazon.counterspeech_experiment.judging import summarize

        write_json(config.output_dir / "judge_summary.json", summarize(config))


if __name__ == "__main__":
    main()
