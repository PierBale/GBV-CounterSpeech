#!/usr/bin/env bash
#SBATCH --job-name=edos_khan
#SBATCH --output=logs/edos_khan_%j.out
#SBATCH --error=logs/edos_khan_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

MODE=ensemble bash scripts/run_all.sh
