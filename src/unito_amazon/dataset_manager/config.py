from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
EDOS_DATASET_PATH = PROJECT_ROOT / "Dataset" / "Original" / "edos_labelled_aggregated.csv"
CONAN_DATASET_PATH = PROJECT_ROOT / "Dataset" / "Original" / "Multitarget-CONAN.json"
