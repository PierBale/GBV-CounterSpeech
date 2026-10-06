import gc
import importlib.util
import json
from typing import Iterable

import pandas as pd

from config import (
    COUNTERSPEECH_CITE_SOURCE,
    COUNTERSPEECH_LANGUAGE,
    COUNTERSPEECH_MAX_SENTENCES,
    DATASET,
    DEVICE,
    DOCUMENTS_DIR,
    DTYPE,
    EDOS_SPLIT,
    EMBEDDING_BATCH_SIZE,
    FORCE_REINDEX,
    GENERATION_BATCH_SIZE,
    GENERATION_MODELS,
    GENERATION_OUTPUT_DIR,
    INCLUDE_NON_SEXIST,
    INDEX_DIR,
    LIMIT,
    RAG_RETRIEVERS,
    RUN_WITHOUT_RAG,
    TARGETS,
    TRUST_REMOTE_CODE,
)
from unito_amazon.dataset_manager.loader import DatasetLoader
from unito_amazon.generation.counter_speech_generator import CounterSpeechGenerator
from unito_amazon.generation.rag_generator import RAGGenerator
from unito_amazon.generation.prompts import (
    CounterSpeechPrompt,
    RAGCounterSpeechPrompt,
)
from unito_amazon.model_registry import resolve_decoder_model, resolve_encoder_model
from unito_amazon.retrieval.bm25 import BM25Retriever
from unito_amazon.retrieval.documents import markdown_directory_to_documents


def validate_config() -> None:
    if not GENERATION_MODELS:
        raise ValueError("At least one decoder model must be configured.")
    if DATASET not in {"edos", "multiconan", "both"}:
        raise ValueError(f"Unsupported dataset in config.py: {DATASET!r}.")
    if not RUN_WITHOUT_RAG and not RAG_RETRIEVERS:
        raise ValueError("Enable without-RAG generation or at least one retriever.")

    supported_retrievers = {"bm25", "qwen3-emb-0.6b"}
    unknown_retrievers = set(RAG_RETRIEVERS) - supported_retrievers
    if unknown_retrievers:
        raise ValueError(
            f"Unsupported retrievers in config.py: {sorted(unknown_retrievers)}."
        )

    if (
        "qwen3-emb-0.6b" in RAG_RETRIEVERS
        and importlib.util.find_spec("sentence_transformers") is None
    ):
        raise RuntimeError(
            "qwen3-emb-0.6b requires sentence-transformers. Install "
            "requirements-generation.txt or remove it from RAG_RETRIEVERS "
            "in config.py."
        )


def load_generation_data() -> pd.DataFrame:
    requested_datasets = (
        ("edos", "multiconan")
        if DATASET == "both"
        else (DATASET,)
    )
    frames = []

    for name in requested_datasets:
        split = EDOS_SPLIT if name == "edos" else None
        frame = DatasetLoader.load_dataset(
            name,
            split=split,
            targets=list(TARGETS),
        )
        if name == "edos" and not INCLUDE_NON_SEXIST:
            frame = frame[frame["label_sexist"] == "sexist"]
        if LIMIT > 0:
            frame = frame.head(LIMIT)
        frames.append(frame)
        print(f"Loaded {len(frame)} records from {name}.")

    return pd.concat(frames, ignore_index=True)


def batched(
    df: pd.DataFrame,
    batch_size: int,
) -> Iterable[pd.DataFrame]:
    if batch_size < 1:
        raise ValueError("generation_batch_size must be at least 1.")
    for start in range(0, len(df), batch_size):
        yield df.iloc[start : start + batch_size]


def build_generators(llm) -> dict:
    generators = {}
    prompt_options = {
        "language": COUNTERSPEECH_LANGUAGE,
        "max_sentences": COUNTERSPEECH_MAX_SENTENCES,
        "cite_source": COUNTERSPEECH_CITE_SOURCE,
    }
    if RUN_WITHOUT_RAG:
        generators["without_rag"] = CounterSpeechGenerator(
            llm=llm,
            prompt=CounterSpeechPrompt(**prompt_options),
        )

    if not RAG_RETRIEVERS:
        return generators

    print(f"Loading RAG documents from {DOCUMENTS_DIR}...")
    documents = markdown_directory_to_documents(DOCUMENTS_DIR)

    for retriever_name in RAG_RETRIEVERS:
        if retriever_name == "bm25":
            print(f"Indexing {len(documents)} retrieval units with BM25...")
            retriever = BM25Retriever()
            retriever.index(documents)
        elif retriever_name == "qwen3-emb-0.6b":
            from unito_amazon.retrieval.dense import DenseRetriever

            encoder = resolve_encoder_model(retriever_name)
            print(
                f"Loading/indexing {len(documents)} retrieval units with "
                f"{encoder.paper_name}..."
            )
            retriever = DenseRetriever(
                model_name=encoder.huggingface_id,
                index_folder=INDEX_DIR,
                index_name="parsed_results",
                retriever_name=encoder.paper_name,
                query_prompt_name="query",
                device=DEVICE,
                batch_size=EMBEDDING_BATCH_SIZE,
            )
            retriever.load_or_index(
                documents,
                force_reindex=FORCE_REINDEX,
                save=True,
            )
        else:
            raise ValueError(
                f"Unsupported RAG retriever: {retriever_name!r}. "
                "Available: 'bm25', 'qwen3-emb-0.6b'."
            )

        strategy_name = f"rag_{retriever_name.replace('-', '_')}"
        generators[strategy_name] = RAGGenerator(
            llm=llm,
            retriever=retriever,
            prompt=RAGCounterSpeechPrompt(**prompt_options),
            top_k=3,
        )

    return generators


def serialize_result(
    row: pd.Series,
    strategy: str,
    result,
    model_alias: str,
) -> dict:
    return {
        "id": row["id"],
        "source": row["source"],
        "split": _optional_value(row["split"]),
        "target": _optional_value(row["target"]),
        "edos_label": _optional_value(row["edos_label"]),
        "hateful_message": result.hateful_message,
        "reference_counter_speech": _optional_value(row["counter_speech"]),
        "generation_strategy": strategy,
        "generation_model": model_alias,
        "generated_counter_speech": result.counter_speech,
        "system_prompt": result.system_prompt,
        "prompt": result.prompt,
        "evidence": [
            {
                "id": retrieval.document.id,
                "text": retrieval.document.text,
                "source_file": retrieval.document.metadata.get("source_file"),
                "score": retrieval.score,
                "rank": retrieval.rank,
            }
            for retrieval in result.retrieval_results
        ],
        "metadata": result.metadata,
    }


def _optional_value(value):
    return None if value is None or pd.isna(value) else value


def run_model(data: pd.DataFrame, model_alias: str) -> None:
    from unito_amazon.llm.factory import LLMFactory

    model = resolve_decoder_model(model_alias)
    output_path = GENERATION_OUTPUT_DIR / (
        f"{model.alias}_{DATASET}_counterspeech.jsonl"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    llm = LLMFactory.create(
        {
            "provider": "huggingface",
            "model_name": model.huggingface_id,
            "device": DEVICE,
            "dtype": DTYPE,
            "trust_remote_code": TRUST_REMOTE_CODE,
        }
    )
    generators = build_generators(llm)
    if not generators:
        llm.destroy()
        raise ValueError("No generation strategy is enabled in config.py.")

    completed = 0
    try:
        with output_path.open("w", encoding="utf-8") as output_file:
            for batch in batched(data, GENERATION_BATCH_SIZE):
                messages = batch["hate_speech"].tolist()
                for strategy, generator in generators.items():
                    results = generator.generate_batch(messages)
                    for (_, row), result in zip(
                        batch.iterrows(),
                        results,
                        strict=True,
                    ):
                        record = serialize_result(
                            row,
                            strategy,
                            result,
                            model_alias=model.paper_name,
                        )
                        output_file.write(
                            json.dumps(record, ensure_ascii=False) + "\n"
                        )

                completed += len(batch)
                print(
                    f"[{model.paper_name}] Generated {completed}/{len(data)} records."
                )
    finally:
        generators.clear()
        llm.destroy()
        gc.collect()

    print(f"Results written to {output_path.resolve()}")


def main() -> None:
    validate_config()
    data = load_generation_data()
    if data.empty:
        raise ValueError("No dataset records match the configuration.")

    for model_alias in GENERATION_MODELS:
        run_model(data, model_alias)


if __name__ == "__main__":
    main()
