#!/bin/bash
#SBATCH -o counterspeech_experiment.out
#SBATCH -e counterspeech_experiment.out
#SBATCH --gres=gpu:1
#SBATCH -p gracehopper

# Counter-speech experiment: zero-shot, RAG, card-RAG and hybrid RAG (3 cards + 3 chunks); the
# retrieval systems also with the EDOS label (predicted, and gold on EDOS) in the prompt and as a retrieval filter.
# Submit from the src folder:
#   sbatch run_counterspeech_experiment.sh
# Only some stages (default: all, in this order):
#   STAGES="generate judge" sbatch run_counterspeech_experiment.sh
#   STAGES="metrics" sbatch run_counterspeech_experiment.sh   # MultiCONAN reference-based metrics
# Every stage resumes: finished outputs are reused, not recomputed.

source ~/mambaforge/etc/profile.d/conda.sh
conda activate llm_new_env
export HPCX_HOME=/opt/hpcx
export PYTHONPATH=/opt/pytorch/lib/python3.12/site-packages

set -euo pipefail

# Under sbatch, BASH_SOURCE points to the spool directory: use the submit directory.
if [[ -f "${SLURM_SUBMIT_DIR:-}/counterspeech_experiment.py" ]]; then
  SRC_DIR="$SLURM_SUBMIT_DIR"
elif [[ -f "${SLURM_SUBMIT_DIR:-}/src/counterspeech_experiment.py" ]]; then
  SRC_DIR="$SLURM_SUBMIT_DIR/src"
else
  SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
PROJECT_DIR="$(cd "$SRC_DIR/.." && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/data/counterspeech_v2}"
CLASSIFIER_OUTPUT="$OUTPUT_DIR/edos_classifier"
STAGES="${STAGES:-classifier sample generate judge metrics}"

python -c "import torch; print('Torch', torch.__version__, '| CUDA', torch.cuda.is_available())"
echo "Project: $PROJECT_DIR"
echo "Output:  $OUTPUT_DIR"
echo "Stages:  $STAGES"

for stage in $STAGES; do
  echo "===== $stage ($(date)) ====="
  case "$stage" in
    classifier)
      # Official Khan et al. M7-FE checkpoints on the whole EDOS sexist test set
      # (also gives the macro F1 to check the reproduction) and on CONAN-WOMEN.
      if [[ -f "$CLASSIFIER_OUTPUT/edos/m7fe.csv" && -f "$CLASSIFIER_OUTPUT/conan/m7fe.csv" ]]; then
        echo "Classifier labels already present in $CLASSIFIER_OUTPUT: skipped."
      else
        EDOS_CLASSIFIER_ROOT="$PROJECT_DIR/EDOS Classifier" \
        OUTPUT_ROOT="$CLASSIFIER_OUTPUT" \
          bash "$PROJECT_DIR/EDOS Classifier/run_official_task_c.sh"
      fi
      ;;
    sample|generate|judge|metrics)
      (cd "$SRC_DIR" && python counterspeech_experiment.py "$stage" --output-dir "$OUTPUT_DIR")
      ;;
    *)
      echo "Unknown stage: $stage" >&2
      exit 2
      ;;
  esac
done

echo "Done ($(date)). Results: $OUTPUT_DIR/judge_summary.json"
