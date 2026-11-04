#!/bin/bash
#SBATCH --job-name=edos-query-instruction-diagnostic
#SBATCH --partition=epito
#SBATCH --gres=gpu:1
#SBATCH --output=edos-diag-%j.out
#SBATCH --error=edos-diag-%j.err

set -euo pipefail

# Submit from the repository root after activating the cluster Python environment.
# Dense retrieval only: compares query formats and Qwen instructions on the v4
# chunks. Builds (or reuses) the same FAISS indexes the experiment uses, so
# running this first does not waste the embedding time of the experiment.
# Output: data/card_creation_comparison_edos_v2_words_clean_v4_20261104/diagnostics/query_instructions/
cd "${SLURM_SUBMIT_DIR:?Submit this script using sbatch from the repository root}"
python -u scripts/diagnose_query_instructions.py \
    --config src/unito_amazon/card_creation/config.edos-v2-words-clean-v4.json
