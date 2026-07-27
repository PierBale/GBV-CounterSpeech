#!/bin/bash
#SBATCH -o Snakefile-ces-ukr2.out
#SBATCH -e Snakefile-ces-ukr2.out
#SBATCH --gres=gpu:1
#SBATCH -p gracehopper


# mettere SBATCH -p epito se si vuole usare epito oppure gracehopper se si vuole usare gracehopper

# mettere sbatch --reservation mike quando prenoto

source ~/mambaforge/etc/profile.d/conda.sh

# Attiva l'ambiente
# conda activate llm
conda activate llm_new_env

# Esporta il PYTHONPATH di PyTorch
export HPCX_HOME=/opt/hpcx
export PYTHONPATH=/opt/pytorch/lib/python3.12/site-packages

set -euo pipefail

# Slurm keeps the submission directory as cwd; always enter this script directory.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
PYTHON_BIN="${PYTHON_BIN:-python}"
EDOS_CSV="${EDOS_CSV:-data/edos/raw/edos_labelled_aggregated.csv}"
CONAN_JSON="${CONAN_JSON:-data/conan/WOMAN-Multitarget-CONAN.json}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-models/official_task_c}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/official_task_c}"
DEVICE="${DEVICE:-cuda:0}"
MAX_LENGTH="${MAX_LENGTH:-150}"
BASE_BATCH_SIZE="${BASE_BATCH_SIZE:-8}"
MISTRAL_BATCH_SIZE="${MISTRAL_BATCH_SIZE:-2}"
DOWNLOAD_MODELS="${DOWNLOAD_MODELS:-0}"
ALLOW_UNSAFE_CHECKPOINT_LOAD="${ALLOW_UNSAFE_CHECKPOINT_LOAD:-0}"

if [[ "$DOWNLOAD_MODELS" == "1" ]]; then
  "$PYTHON_BIN" scripts/11_download_official_task_c.py \
    --output-dir "$CHECKPOINT_ROOT"
fi

checkpoint_load_args=()
if [[ "$ALLOW_UNSAFE_CHECKPOINT_LOAD" == "1" ]]; then
  checkpoint_load_args+=(--allow-unsafe-checkpoint-load)
fi

mkdir -p "$OUTPUT_ROOT/edos" "$OUTPUT_ROOT/conan"

for model_key in deberta9 dtfn3 dtfn7 dtfn8 mistral; do
  batch_size="$BASE_BATCH_SIZE"
  if [[ "$model_key" == "mistral" ]]; then
    batch_size="$MISTRAL_BATCH_SIZE"
  fi

  "$PYTHON_BIN" scripts/12_infer_official_task_c.py \
    --model "$model_key" \
    --dataset edos-test \
    --input "$EDOS_CSV" \
    --checkpoint-root "$CHECKPOINT_ROOT" \
    --output-csv "$OUTPUT_ROOT/edos/${model_key}.csv" \
    --metrics-json "$OUTPUT_ROOT/edos/${model_key}_metrics.json" \
    --batch-size "$batch_size" \
    --max-length "$MAX_LENGTH" \
    --device "$DEVICE" \
    "${checkpoint_load_args[@]}"

  "$PYTHON_BIN" scripts/12_infer_official_task_c.py \
    --model "$model_key" \
    --dataset conan \
    --input "$CONAN_JSON" \
    --checkpoint-root "$CHECKPOINT_ROOT" \
    --output-csv "$OUTPUT_ROOT/conan/${model_key}.csv" \
    --batch-size "$batch_size" \
    --max-length "$MAX_LENGTH" \
    --device "$DEVICE" \
    "${checkpoint_load_args[@]}"
done

"$PYTHON_BIN" scripts/13_ensemble_official_task_c.py \
  --deberta9 "$OUTPUT_ROOT/edos/deberta9.csv" \
  --dtfn3 "$OUTPUT_ROOT/edos/dtfn3.csv" \
  --dtfn7 "$OUTPUT_ROOT/edos/dtfn7.csv" \
  --dtfn8 "$OUTPUT_ROOT/edos/dtfn8.csv" \
  --mistral "$OUTPUT_ROOT/edos/mistral.csv" \
  --output-csv "$OUTPUT_ROOT/edos/m7fe.csv" \
  --metrics-json "$OUTPUT_ROOT/edos/m7fe_metrics.json"

"$PYTHON_BIN" scripts/13_ensemble_official_task_c.py \
  --deberta9 "$OUTPUT_ROOT/conan/deberta9.csv" \
  --dtfn3 "$OUTPUT_ROOT/conan/dtfn3.csv" \
  --dtfn7 "$OUTPUT_ROOT/conan/dtfn7.csv" \
  --dtfn8 "$OUTPUT_ROOT/conan/dtfn8.csv" \
  --mistral "$OUTPUT_ROOT/conan/mistral.csv" \
  --output-csv "$OUTPUT_ROOT/conan/m7fe.csv" \
  --conan-json "$CONAN_JSON" \
  --annotated-json \
    "data/conan/WOMAN-Multitarget-CONAN_EDOS_TASK_C_M7FE.json"

echo "EDOS metrics: $OUTPUT_ROOT/edos/m7fe_metrics.json"
echo "Annotated CONAN: data/conan/WOMAN-Multitarget-CONAN_EDOS_TASK_C_M7FE.json"
