#!/bin/bash
#SBATCH -o edos-rerank-diag-%j.out
#SBATCH -e edos-rerank-diag-%j.out
#SBATCH --gres=gpu:1
#SBATCH -p epito

# Submit with sbatch from the src directory, as the experiment.
# Re-scores the v4 dense top-100 candidates with Qwen3-Reranker-8B under four
# instructions. Output:
# data/card_creation_comparison_edos_v2_words_clean_v4_20261104/diagnostics/reranker_instructions/

source ~/mambaforge/etc/profile.d/conda.sh
conda activate llm_new_env
export HPCX_HOME=/opt/hpcx
export PYTHONPATH=/opt/pytorch/lib/python3.12/site-packages

python -u ../scripts/diagnose_reranker_instructions.py \
  --config unito_amazon/card_creation/config.edos-v2-words-clean-v4.json
