#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from edos_khan.official_task_c import (
    DEFAULT_CHECKPOINT_ROOT,
    OFFICIAL_TASK_C_MODELS,
)

PINNED_REVISIONS = {
    "deberta9": "e278903e8cc36bc03235df6f16a5293ce9f01e4e",
    "dtfn3": "3051e6cc6fd6307ac439e8046fd7581916cb3568",
    "dtfn7": "3a0de2ba351a7335031cf6d6ce09c472ae918d41",
    "dtfn8": "bb983c762702c873e648d69882f7c7a1f81d5bf1",
    "mistral": "d526216101f58aa34d505a3ff1c61fe70bbd51b6",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download the five official Khan et al. EDOS Task C checkpoints."
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_CHECKPOINT_ROOT),
        help="Destination root. Each Hugging Face repository gets one subdirectory.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(OFFICIAL_TASK_C_MODELS),
        default=list(OFFICIAL_TASK_C_MODELS),
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional override; otherwise use the pinned inspected commit.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_root = Path(args.output_dir)
    for model_key in args.models:
        spec = OFFICIAL_TASK_C_MODELS[model_key]
        destination = output_root / spec.repo_id.split("/")[-1]
        destination.mkdir(parents=True, exist_ok=True)
        downloaded = hf_hub_download(
            repo_id=spec.repo_id,
            filename=spec.filename,
            revision=args.revision or PINNED_REVISIONS[model_key],
            local_dir=destination,
        )
        if not Path(downloaded).is_file():
            raise FileNotFoundError(
                f"Hugging Face reported a download, but no file exists at {downloaded}"
            )
        print(f"{model_key}: {downloaded}")


if __name__ == "__main__":
    main()

