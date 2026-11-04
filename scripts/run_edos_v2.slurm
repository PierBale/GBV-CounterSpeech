#!/bin/bash
#SBATCH --job-name=edos-v2-20261104
#SBATCH --partition=epito
#SBATCH --gres=gpu:1
#SBATCH --output=edos-v2-%j.out
#SBATCH --error=edos-v2-%j.err

set -euo pipefail

# Submit from the repository root after activating the cluster Python environment.
cd "${SLURM_SUBMIT_DIR:?Submit this script using sbatch from the repository root}"
cd src
python -m unito_amazon.card_creation experiment --config unito_amazon/card_creation/config.edos-v2.json
cd ..

NEW=data/card_creation_comparison_edos_v2_20261104
python Qualtrics/build_retrieval_survey.py \
  --input-dir "$NEW/runs/edos-v2-20261104/reranking" \
  --template "$NEW/inputs/survey_template.qsf" \
  --top-n 3 --merge-common-chunks \
  --output "$NEW/survey/Best_retrieval_config.qsf" \
  --csv-output "$NEW/survey/Best_retrieval_config_question_map.csv"
