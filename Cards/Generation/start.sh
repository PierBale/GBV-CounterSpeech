#!/bin/bash
#SBATCH -o Snakefile-ces-ukr2.out
#SBATCH -e Snakefile-ces-ukr2.out
#SBATCH --gres=gpu:1
#SBATCH -p epito


# mettere SBATCH -p epito se si vuole usare epito oppure gracehopper se si vuole usare gracehopper

# mettere sbatch --reservation mike quando prenoto

source ~/mambaforge/etc/profile.d/conda.sh

# Attiva l'ambiente
# conda activate llm
conda activate llm_new_env
# Esporta il PYTHONPATH di PyTorch
export HPCX_HOME=/opt/hpcx
export PYTHONPATH=/opt/pytorch/lib/python3.12/site-packages


# Verifica PyTorch e CUDA
python -c "import torch; print('Torch Version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('Compiled with CUDA:', torch.version.cuda)"

# Riprende gli output incrementali esistenti e genera solo quelli mancanti.
python scripts/03_generate_candidate_cards_hf.py --resume
