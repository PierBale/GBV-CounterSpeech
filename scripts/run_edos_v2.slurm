#!/bin/bash
#SBATCH --job-name=edos-v2-words-clean-v4-20261104
#SBATCH --partition=epito
#SBATCH --gres=gpu:1
#SBATCH --output=edos-v2-%j.out
#SBATCH --error=edos-v2-%j.err

set -euo pipefail

# Submit from the repository root after activating the cluster Python environment.
# Rebuilds chunks (editorial rules 4.0), both FAISS indexes and all four
# embedder x reranker rankings into a new output directory; the previous run
# (edos-v2-words-clean-20261104) is left untouched.
cd "${SLURM_SUBMIT_DIR:?Submit this script using sbatch from the repository root}"
cd src
python -u -m unito_amazon.card_creation experiment --config unito_amazon/card_creation/config.edos-v2-words-clean-v4.json
cd ..

# The questionnaires are not regenerated automatically: the number of unique
# label-chunk items changes with the new rankings and the 27 + 3 x n split
# must be checked first. Then run:
#   python Qualtrics/retrieval_internal_annotation_final/build_surveys.py --overwrite
