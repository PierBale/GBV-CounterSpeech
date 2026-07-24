#!/usr/bin/env python3
from pathlib import Path
import json
import pandas as pd

REQUIRED = {
    "EDOS labelled CSV": Path("data/edos/raw/edos_labelled_aggregated.csv"),
    "Khan DDA augmentation CSV": Path("data/edos/raw/variations_augmentation_gpt4o_five_classes.csv"),
    "CONAN-WOMEN JSON": Path("data/conan/WOMAN-Multitarget-CONAN.json"),
}

def main():
    print("=== Data check ===")
    missing = []
    for name, path in REQUIRED.items():
        if path.exists():
            print(f"[OK] {name}: {path}")
        else:
            print(f"[MISSING] {name}: {path}")
            missing.append(str(path))

    if missing:
        raise SystemExit(f"Missing required files: {missing}")

    edos = pd.read_csv(REQUIRED["EDOS labelled CSV"])
    aug = pd.read_csv(REQUIRED["Khan DDA augmentation CSV"])
    conan = json.loads(REQUIRED["CONAN-WOMEN JSON"].read_text(encoding="utf-8"))

    print("\n=== Dataset sizes ===")
    print(f"EDOS rows: {len(edos)}")
    print(f"Khan DDA rows: {len(aug)}")
    print(f"CONAN-WOMEN instances: {len(conan) if isinstance(conan, dict) else 'not a JSON object'}")

    print("\n=== Required columns ===")
    edos_required = {"text", "label_sexist", "label_category", "label_vector", "split"}
    aug_required = {"generated_text", "label_vector"}

    print(f"EDOS columns OK: {edos_required.issubset(edos.columns)}")
    print(f"Augmentation columns OK: {aug_required.issubset(aug.columns)}")

    if not edos_required.issubset(edos.columns):
        raise SystemExit(f"EDOS missing: {sorted(edos_required - set(edos.columns))}")
    if not aug_required.issubset(aug.columns):
        raise SystemExit(f"Augmentation missing: {sorted(aug_required - set(aug.columns))}")

    if isinstance(conan, dict):
        first_key = next(iter(conan))
        first = conan[first_key]
        conan_required = {"HATE_SPEECH", "COUNTER_NARRATIVE", "TARGET"}
        print(f"CONAN sample id: {first_key}")
        print(f"CONAN fields OK: {conan_required.issubset(first.keys())}")
        if not conan_required.issubset(first.keys()):
            raise SystemExit(f"CONAN missing fields in sample: {sorted(conan_required - set(first.keys()))}")

    print("\nAll required files are ready.")

if __name__ == "__main__":
    main()
