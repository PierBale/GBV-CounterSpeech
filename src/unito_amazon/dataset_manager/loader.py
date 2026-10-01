from typing import Optional

import pandas as pd

from unito_amazon.dataset_manager.config import EDOS_DATASET_PATH, CONAN_DATASET_PATH
from unito_amazon.dataset_manager.schema import DatasetRecord


COMMON_COLUMNS = list(DatasetRecord.model_fields)


def _validate_records(df: pd.DataFrame) -> pd.DataFrame:
    clean_df = df.astype(object).where(df.notna(), None)
    records = [
        DatasetRecord.model_validate(record).model_dump()
        for record in clean_df.to_dict(orient="records")
    ]
    return pd.DataFrame.from_records(records, columns=COMMON_COLUMNS)


class DatasetLoader:
    @staticmethod
    def load_dataset(dataset_name: str, split: Optional[str] = None, targets: Optional[list[str]] = None) -> pd.DataFrame:
        normalized_name = dataset_name.strip().lower()

        if normalized_name == "edos":
            return DatasetLoader.load_edos(split, targets)
        if normalized_name in {"conan", "multiconan", "multitarget-conan"}:
            return DatasetLoader.load_conan(split, targets)

        raise ValueError(
            f"Unknown dataset: {dataset_name!r}. "
            "Available datasets: 'edos', 'conan'/'multiconan'."
        )

    @staticmethod
    def load_edos(split: Optional[str] = None, targets: Optional[list[str]] = None) -> pd.DataFrame:
        if targets is not None and "WOMEN" not in targets:
            return pd.DataFrame(columns=COMMON_COLUMNS)

        df = pd.read_csv(EDOS_DATASET_PATH)

        if split is not None:
            df = df[df["split"] == split]

        normalized = pd.DataFrame(
            {
                "id": df["rewire_id"].astype(str),
                "hate_speech": df["text"],
                "counter_speech": pd.NA,
                "edos_label": df["label_vector"].replace("none", pd.NA),
                "target": "WOMEN",
                "split": df["split"],
                "source": "edos",
                "label_sexist": df["label_sexist"],
                "label_category": df["label_category"].replace("none", pd.NA),
                "version": pd.NA,
            }
        )
        return _validate_records(normalized)


    @staticmethod
    def load_conan(split: Optional[str] = None, targets: Optional[list[str]] = None) -> pd.DataFrame:
        if split is not None:
            raise ValueError(
                "MultiCONAN does not provide official train/dev/test splits. "
                "Load it with split=None."
            )

        df = pd.read_json(CONAN_DATASET_PATH, orient="index")

        if targets is not None:
            df = df[df["TARGET"].isin(targets)]

        normalized = pd.DataFrame(
            {
                "id": "multiconan-" + df.index.astype(str),
                "hate_speech": df["HATE_SPEECH"],
                "counter_speech": df["COUNTER_NARRATIVE"],
                "edos_label": pd.NA,
                "target": df["TARGET"],
                "split": pd.NA,
                "source": "multiconan",
                "label_sexist": pd.NA,
                "label_category": pd.NA,
                "version": df["VERSION"],
            }
        )
        return _validate_records(normalized)
