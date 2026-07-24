#!/usr/bin/env bash
set -euo pipefail

source config/defaults.env

MODE="${MODE:-deberta}"  # deberta | ensemble
INPUT_JSON="${INPUT_JSON:-$CONAN_JSON}"

echo "=== CONAN-WOMEN annotation ==="
echo "MODE=$MODE"
echo "INPUT_JSON=$INPUT_JSON"

python scripts/01_check_data.py

if [[ "$MODE" == "deberta" ]]; then
  echo ""
  echo "=== Predict Task B with DeBERTa ==="
  python scripts/07_predict_dataset_transformer.py \
    --task b \
    --input "$INPUT_JSON" \
    --output-csv outputs/conan/task_b_deberta_predictions.csv \
    --model-kind hf_seqcls \
    --model-dir models/task_b/deberta/best_model \
    --batch-size 16 \
    --max-length "$MAX_LENGTH" \
    --include-probs

  echo ""
  echo "=== Predict Task C with DeBERTa ==="
  python scripts/07_predict_dataset_transformer.py \
    --task c \
    --input "$INPUT_JSON" \
    --output-csv outputs/conan/task_c_deberta_predictions.csv \
    --model-kind hf_seqcls \
    --model-dir models/task_c/deberta/best_model \
    --batch-size 16 \
    --max-length "$MAX_LENGTH" \
    --include-probs

  echo ""
  echo "=== Write final annotated dataset ==="
  python scripts/10_write_dataset_annotations.py \
    --input-json "$INPUT_JSON" \
    --task-b-preds outputs/conan/task_b_deberta_predictions.csv \
    --task-c-preds outputs/conan/task_c_deberta_predictions.csv \
    --output-json data/conan/WOMAN-Multitarget-CONAN_EDOS_deberta.json \
    --output-csv data/conan/WOMAN-Multitarget-CONAN_EDOS_deberta.csv \
    --overwrite

elif [[ "$MODE" == "ensemble" ]]; then
  for TASK in b c; do
    if [[ "$TASK" == "b" ]]; then
      DATA_DIR="data/edos/processed/task_b"
      MODEL_DIR="models/task_b"
    else
      DATA_DIR="data/edos/processed/task_c"
      MODEL_DIR="models/task_c"
    fi

    TASK_OUT="outputs/conan/task_${TASK}"
    mkdir -p "$TASK_OUT"

    echo ""
    echo "=== Task ${TASK^^}: DeBERTa predictions ==="
    python scripts/07_predict_dataset_transformer.py \
      --task "$TASK" \
      --input "$INPUT_JSON" \
      --output-csv "$TASK_OUT/deberta_predictions.csv" \
      --model-kind hf_seqcls \
      --model-dir "$MODEL_DIR/deberta/best_model" \
      --batch-size 16 \
      --max-length "$MAX_LENGTH" \
      --include-probs

    echo ""
    echo "=== Task ${TASK^^}: DTFN predictions ==="
    python scripts/08_predict_dataset_dtfn.py \
      --task "$TASK" \
      --input "$INPUT_JSON" \
      --output-csv "$TASK_OUT/dtfn_predictions.csv" \
      --model-dir "$MODEL_DIR/dtfn" \
      --label-map "$DATA_DIR/label_map.json" \
      --deberta-model "$DEBERTA_MODEL" \
      --roberta-model "$ROBERTA_MODEL" \
      --batch-size 4 \
      --max-length "$MAX_LENGTH" \
      --include-probs

    echo ""
    echo "=== Task ${TASK^^}: Mistral fallback predictions ==="
    python scripts/07_predict_dataset_transformer.py \
      --task "$TASK" \
      --input "$INPUT_JSON" \
      --output-csv "$TASK_OUT/mistral_predictions.csv" \
      --model-kind mistral_lora \
      --base-model "$MISTRAL_MODEL" \
      --adapter-dir "$MODEL_DIR/mistral/best_adapter" \
      --batch-size 4 \
      --max-length "$MAX_LENGTH" \
      --include-probs

    echo ""
    echo "=== Task ${TASK^^}: M7 fallback ensemble ==="
    python scripts/09_ensemble_dataset_predictions.py \
      --task "$TASK" \
      --base-preds \
        "$TASK_OUT/deberta_predictions.csv" \
        "$TASK_OUT/dtfn_predictions.csv" \
      --fallback-preds "$TASK_OUT/mistral_predictions.csv" \
      --output-csv "$TASK_OUT/m7_fe_predictions.csv"
  done

  echo ""
  echo "=== Write final annotated dataset ==="
  python scripts/10_write_dataset_annotations.py \
    --input-json "$INPUT_JSON" \
    --task-b-preds outputs/conan/task_b/m7_fe_predictions.csv \
    --task-c-preds outputs/conan/task_c/m7_fe_predictions.csv \
    --output-json data/conan/WOMAN-Multitarget-CONAN_EDOS_m7fe.json \
    --output-csv data/conan/WOMAN-Multitarget-CONAN_EDOS_m7fe.csv \
    --overwrite

else
  echo "Unknown MODE=$MODE. Use MODE=deberta or MODE=ensemble."
  exit 1
fi

echo ""
echo "=== Annotation finished ==="
bash scripts/show_results.sh
