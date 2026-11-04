#!/bin/bash
#SBATCH --job-name=edos-v2-words-clean-20261104
#SBATCH --partition=epito
#SBATCH --gres=gpu:1
#SBATCH --output=edos-v2-%j.out
#SBATCH --error=edos-v2-%j.err

set -euo pipefail

# Submit from the repository root after activating the cluster Python environment.
cd "${SLURM_SUBMIT_DIR:?Submit this script using sbatch from the repository root}"
cd src
python -u -m unito_amazon.card_creation experiment --config unito_amazon/card_creation/config.edos-v2-words-clean.json
cd ..

python Qualtrics/retrieval_internal_annotation_final/build_surveys.py
