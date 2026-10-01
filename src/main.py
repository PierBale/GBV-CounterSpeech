from unito_amazon.dataset_manager.loader import DatasetLoader

def main():
    DATASETS = {
        "EDOS": "test",
        "MultiCONAN": None,
    }

    TARGETS = [
        "WOMEN",
    ]

    STRATEGIES = [
        "zero-shot",
        "zero-shot + EDOS-based generation",
        "RAG",
        "RAG + EDOS-based retrieval",
        "RAG + EDOS-based generation",
        "RAG + EDOS-based retrieval + EDOS-based generation",
        #"CARD-RAG",
        #"CARD-RAG + EDOS-based retrieval",
        #"CARD-RAG + EDOS-based generation",
        #"CARD-RAG + EDOS-based retrieval + EDOS-based generation",
    ]

    for dataset, split in DATASETS.items():
        df = DatasetLoader.load_dataset(dataset, split, targets=TARGETS)
        split_description = f" ({split} split)" if split else ""
        print(f"{dataset}{split_description}: {len(df)} records")

        if df["edos_label"].notna().any():
            print("EDOS label distribution:")
            print(df["edos_label"].value_counts())

        if df["target"].notna().any():
            print("Target distribution:")
            print(df["target"].value_counts())

        print()

    # STRATEGIES is ready for the generation/evaluation loop.
    print(f"Configured strategies: {len(STRATEGIES)}")


if __name__ == "__main__":
    main()
