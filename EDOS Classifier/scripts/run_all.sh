#!/usr/bin/env bash
set -euo pipefail

MODE="${MODE:-deberta}"  # deberta | ensemble

if [[ "$MODE" == "deberta" ]]; then
  echo "=== Full pipeline: DeBERTa-only ==="
  MODE=deberta bash scripts/run_reproduce_paper.sh
  MODE=deberta bash scripts/run_annotate_conan.sh
elif [[ "$MODE" == "ensemble" ]]; then
  echo "=== Full pipeline: Khan-style ensemble ==="
  MODE=full bash scripts/run_reproduce_paper.sh
  MODE=ensemble bash scripts/run_annotate_conan.sh
else
  echo "Unknown MODE=$MODE. Use MODE=deberta or MODE=ensemble."
  exit 1
fi
