#!/usr/bin/env bash
set -euo pipefail

echo ""
echo "=== EDOS metrics ==="
for f in \
  models/task_b/deberta/test_metrics.json \
  models/task_c/deberta/test_metrics.json \
  outputs/task_b/m7_fe_metrics.json \
  outputs/task_c/m7_fe_metrics.json
do
  if [[ -f "$f" ]]; then
    echo ""
    echo "--- $f ---"
    python - <<PY
import json
p="$f"
with open(p, encoding="utf-8") as fh:
    d=json.load(fh)
print("accuracy  =", d.get("accuracy"))
print("macro_f1  =", d.get("macro_f1"))
print("weighted_f1 =", d.get("weighted_f1"))
PY
  fi
done

echo ""
echo "=== Annotated CONAN files ==="
ls -lh data/conan/*EDOS* 2>/dev/null || true
