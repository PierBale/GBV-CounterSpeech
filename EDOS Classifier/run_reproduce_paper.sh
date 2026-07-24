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

python -m pip install peft

# Esporta il PYTHONPATH di PyTorch
export HPCX_HOME=/opt/hpcx
export PYTHONPATH=/opt/pytorch/lib/python3.12/site-packages

set -euo pipefail

# Load defaults if available, then set safe fallbacks.
if [[ -f config/defaults.env ]]; then
  set -a
  source config/defaults.env
  set +a
fi

EDOS_CSV="${EDOS_CSV:-data/edos/raw/edos_labelled_aggregated.csv}"
AUG_CSV="${AUG_CSV:-data/edos/raw/variations_augmentation_gpt4o_five_classes.csv}"
CONAN_JSON="${CONAN_JSON:-data/conan/WOMAN-Multitarget-CONAN.json}"

DEBERTA_MODEL="${DEBERTA_MODEL:-microsoft/deberta-v3-large}"
ROBERTA_MODEL="${ROBERTA_MODEL:-roberta-large}"
MISTRAL_MODEL="${MISTRAL_MODEL:-mistralai/Mistral-7B-v0.1}"

MAX_LENGTH="${MAX_LENGTH:-150}"
SELECTION_SPLIT="${SELECTION_SPLIT:-test}"

TASKS="${TASKS:-b c}"
MODE="${MODE:-full}"  # full | deberta

if [[ "$MODE" == "deberta" ]]; then
  RUN_DEBERTA=1
  RUN_DTFN=0
  RUN_MISTRAL=0
  RUN_ENSEMBLE=0
else
  RUN_DEBERTA=1
  RUN_DTFN=1
  RUN_MISTRAL=1
  RUN_ENSEMBLE=1
fi

echo "=== EDOS Khan-style reproduction ==="
echo "MODE=$MODE"
echo "TASKS=$TASKS"
echo "SELECTION_SPLIT=$SELECTION_SPLIT"

python scripts/01_check_data.py

for TASK in $TASKS; do
  if [[ "$TASK" == "b" ]]; then
    DATA_DIR="data/edos/processed/task_b"
    MODEL_DIR="models/task_b"
    OUTPUT_DIR="outputs/task_b"
  elif [[ "$TASK" == "c" ]]; then
    DATA_DIR="data/edos/processed/task_c"
    MODEL_DIR="models/task_c"
    OUTPUT_DIR="outputs/task_c"
  else
    echo "Unknown task: $TASK"
    exit 1
  fi

  mkdir -p "$DATA_DIR" "$MODEL_DIR" "$OUTPUT_DIR"

  echo ""
  echo "=== Task ${TASK^^}: prepare data ==="
  python scripts/02_prepare_edos.py \
    --task "$TASK" \
    --edos-csv "$EDOS_CSV" \
    --augmentation-csv "$AUG_CSV" \
    --output-dir "$DATA_DIR"

  if [[ "$RUN_DEBERTA" == "1" ]]; then
    echo ""
    echo "=== Task ${TASK^^}: train DeBERTa ==="
    python scripts/03_train_deberta.py \
      --task "$TASK" \
      --data-dir "$DATA_DIR" \
      --output-dir "$MODEL_DIR/deberta" \
      --model-name "$DEBERTA_MODEL" \
      --epochs 30 \
      --lr 6e-6 \
      --batch-size 10 \
      --max-length "$MAX_LENGTH" \
      --selection-split "$SELECTION_SPLIT"
  fi

  if [[ "$RUN_DTFN" == "1" ]]; then
    echo ""
    echo "=== Task ${TASK^^}: train DTFN ==="
    python scripts/04_train_dtfn.py \
      --task "$TASK" \
      --data-dir "$DATA_DIR" \
      --output-dir "$MODEL_DIR/dtfn" \
      --deberta-model "$DEBERTA_MODEL" \
      --roberta-model "$ROBERTA_MODEL" \
      --epochs 30 \
      --lr 6e-6 \
      --batch-size 4 \
      --max-length "$MAX_LENGTH" \
      --selection-split "$SELECTION_SPLIT"
  fi

  if [[ "$RUN_MISTRAL" == "1" ]]; then
    echo ""
    echo "=== Task ${TASK^^}: train Mistral-7B LoRA ==="
    python scripts/05_train_mistral.py \
      --task "$TASK" \
      --data-dir "$DATA_DIR" \
      --output-dir "$MODEL_DIR/mistral" \
      --model-name "$MISTRAL_MODEL" \
      --epochs 10 \
      --lr 1e-4 \
      --batch-size 4 \
      --max-length "$MAX_LENGTH" \
      --selection-split "$SELECTION_SPLIT"
  fi

  if [[ "$RUN_ENSEMBLE" == "1" ]]; then
    echo ""
    echo "=== Task ${TASK^^}: M7 fallback ensemble on EDOS test ==="
    python scripts/06_ensemble_test_predictions.py \
      --base-preds \
        "$MODEL_DIR/deberta/test_predictions.csv" \
        "$MODEL_DIR/dtfn/test_predictions.csv" \
      --fallback-preds "$MODEL_DIR/mistral/test_predictions.csv" \
      --output-csv "$OUTPUT_DIR/m7_fe_predictions.csv" \
      --metrics-json "$OUTPUT_DIR/m7_fe_metrics.json"
  fi
done

echo ""
echo "=== Reproduction finished ==="
bash scripts/show_results.sh
