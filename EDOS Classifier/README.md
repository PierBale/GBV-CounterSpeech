# EDOS Task B/C reproduction and CONAN-WOMEN annotation

Clean plug-and-play repository for:

1. reproducing/testing EDOS Task B and Task C classifiers inspired by Khan et al. ACL 2025;
2. applying the trained classifiers to the CONAN-WOMEN hate-speech dataset;
3. saving a new dataset with predicted EDOS labels.

The repo already includes:

```text
data/edos/raw/edos_labelled_aggregated.csv
data/edos/raw/variations_augmentation_gpt4o_five_classes.csv
data/conan/WOMAN-Multitarget-CONAN.json
```

---

## Repository structure

```text
config/
  defaults.env

data/
  edos/
    raw/
      edos_labelled_aggregated.csv
      variations_augmentation_gpt4o_five_classes.csv
    processed/
  conan/
    WOMAN-Multitarget-CONAN.json

models/
  task_b/
  task_c/

outputs/
  task_b/
  task_c/
  conan/

scripts/
  01_check_data.py
  02_prepare_edos.py
  03_train_deberta.py
  04_train_dtfn.py
  05_train_mistral.py
  06_ensemble_test_predictions.py
  07_predict_dataset_transformer.py
  08_predict_dataset_dtfn.py
  09_ensemble_dataset_predictions.py
  10_write_dataset_annotations.py

  run_reproduce_paper.sh
  run_annotate_conan.sh
  run_all.sh
  show_results.sh

src/
  edos_khan/
```

---

## Setup

```bash
unzip edos_khan_conan_clean_repo.zip
cd edos_khan_conan_clean_repo

python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

Check that the included files are readable:

```bash
python scripts/01_check_data.py
```

---

# Option 1: quick DeBERTa-only pipeline

This is the simplest full run:

```bash
MODE=deberta bash scripts/run_all.sh
```

It does:

```text
train/evaluate DeBERTa for Task B and Task C
→ predict Task B and Task C on CONAN-WOMEN
→ write annotated JSON and CSV
```

Outputs:

```text
data/conan/WOMAN-Multitarget-CONAN_EDOS_deberta.json
data/conan/WOMAN-Multitarget-CONAN_EDOS_deberta.csv
```

---

# Option 2: full Khan-style ensemble pipeline

This is closer to the paper-style setup and is heavier:

```bash
MODE=ensemble bash scripts/run_all.sh
```

It does:

```text
train/evaluate DeBERTa
train/evaluate DTFN
train/evaluate Mistral-7B LoRA
combine with M7 fallback ensemble
annotate CONAN-WOMEN with the ensemble predictions
```

Outputs:

```text
data/conan/WOMAN-Multitarget-CONAN_EDOS_m7fe.json
data/conan/WOMAN-Multitarget-CONAN_EDOS_m7fe.csv
```

---

## Run only reproduction

DeBERTa-only:

```bash
MODE=deberta bash scripts/run_reproduce_paper.sh
```

Full setup:

```bash
MODE=full bash scripts/run_reproduce_paper.sh
```

Only Task C:

```bash
TASKS="c" MODE=full bash scripts/run_reproduce_paper.sh
```

---

## Run only annotation

After models have been trained:

```bash
MODE=deberta bash scripts/run_annotate_conan.sh
```

or:

```bash
MODE=ensemble bash scripts/run_annotate_conan.sh
```

---

## Show results

```bash
bash scripts/show_results.sh
```

---

## Output format

Each CONAN instance keeps the original fields and receives:

```json
"HATE_SPEECH_EDOS_PREDICTIONS": {
  "TASK_B": {
    "label": "3. animosity",
    "confidence": 0.82,
    "source": "outputs/conan/task_b_deberta_predictions.csv"
  },
  "TASK_C": {
    "label": "3.2 immutable gender differences and gender stereotypes",
    "confidence": 0.74,
    "parent_task_b": "3. animosity",
    "source": "outputs/conan/task_c_deberta_predictions.csv"
  }
}
```

The CSV contains:

```text
instance_id
HATE_SPEECH
COUNTER_NARRATIVE
TARGET
VERSION
task_b_pred
task_b_confidence
task_b_used_fallback
task_c_pred
task_c_confidence
task_c_parent_task_b
task_c_used_fallback
```

---

## Notes

- The labels added to CONAN-WOMEN are predicted labels, not gold labels.
- `MODE=deberta` is fast enough for a first full test.
- `MODE=ensemble` is the heavier paper-style setup.
- The Mistral step requires a suitable CUDA GPU.
- Exact paper scores may still differ if the original authors used local pretraining checkpoints that are not included in their public repository.


---

## SLURM / cluster note

The runner scripts set safe defaults internally, so they work better with `set -u`.

Direct run from repository root:

```bash
MODE=deberta bash scripts/run_all.sh
```

or:

```bash
MODE=ensemble bash scripts/run_all.sh
```

If your cluster script uses `set -u`, you can also explicitly export the paths:

```bash
export EDOS_CSV=data/edos/raw/edos_labelled_aggregated.csv
export AUG_CSV=data/edos/raw/variations_augmentation_gpt4o_five_classes.csv
export CONAN_JSON=data/conan/WOMAN-Multitarget-CONAN.json

MODE=ensemble bash scripts/run_all.sh
```

A SLURM template is available in:

```text
scripts/slurm_example_full_pipeline.sh
```


---

## Fix notes

This version includes the missing internal package files:

```text
src/edos_khan/common.py
src/edos_khan/labels.py
```

These are required by the training scripts.
