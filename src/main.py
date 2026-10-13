import gc
import importlib.util
import json
from typing import Iterable

import pandas as pd

from config import (
    CARD_EMBEDDING_BATCH_SIZE,
    CARD_EMBEDDING_MODEL,
    CARD_INDEX_PATH,
    CARD_LIBRARY_PATH,
    CHUNK_LIBRARY_PATH,
    CHUNK_RAG_INDEX_PATH,
    CHUNK_RAG_EMBEDDING_BATCH_SIZE,
    COUNTERSPEECH_CITE_SOURCE,
    COUNTERSPEECH_LANGUAGE,
    COUNTERSPEECH_MAX_SENTENCES,
    DATASET,
    DATASET_SAMPLE_SEED,
    DEVICE,
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
from unito_amazon.dataset_manager.sampling import sample_messages
from unito_amazon.card_creation.io import file_sha256, write_json
from unito_amazon.generation.context_stats import measure_context
from unito_amazon.generation.counter_speech_generator import CounterSpeechGenerator
from unito_amazon.generation.rag_generator import RAGGenerator
from unito_amazon.generation.prompts import (
    CounterSpeechPrompt,
    RAGCounterSpeechPrompt,
)
from unito_amazon.model_registry import resolve_decoder_model, resolve_encoder_model
from unito_amazon.retrieval.bm25 import BM25Retriever
from unito_amazon.retrieval.experiment import load_chunk_documents, validate_card_sources


def validate_config() -> None:
    if not GENERATION_MODELS:
        raise ValueError("At least one decoder model must be configured.")
    if DATASET not in {"edos", "multiconan", "both"}:
        raise ValueError(f"Unsupported dataset in config.py: {DATASET!r}.")
    if not RUN_WITHOUT_RAG and not RAG_RETRIEVERS:
        raise ValueError("Enable without-RAG generation or at least one retriever.")

    supported_retrievers = {"bm25", "qwen3-emb-0.6b", "qwen3-emb-4b", "cards"}
    unknown_retrievers = set(RAG_RETRIEVERS) - supported_retrievers
    if unknown_retrievers:
        raise ValueError(
            f"Unsupported retrievers in config.py: {sorted(unknown_retrievers)}."
        )

    if (
        ({"qwen3-emb-0.6b", "qwen3-emb-4b", "cards"} & set(RAG_RETRIEVERS))
        and importlib.util.find_spec("sentence_transformers") is None
    ):
        raise RuntimeError(
            "Dense retrieval requires sentence-transformers. Install "
            "requirements-generation.txt or remove it from RAG_RETRIEVERS "
            "in config.py."
        )

    if {"cards", "qwen3-emb-4b"} & set(RAG_RETRIEVERS):
        if importlib.util.find_spec("faiss") is None:
            raise RuntimeError("FAISS RAG requires requirements-card-creation.txt.")
    if RAG_RETRIEVERS:
        chunks = load_chunk_documents(CHUNK_LIBRARY_PATH)
    if "cards" in RAG_RETRIEVERS:
        from unito_amazon.card_creation.card_retrieval import load_card_documents

        cards = load_card_documents(CARD_LIBRARY_PATH)
        if len(cards) < 3:
            raise ValueError("Card RAG requires at least three usable cards.")
        validate_card_sources(cards, chunks)
    if {"qwen3-emb-4b", "cards"}.issubset(RAG_RETRIEVERS):
        if CARD_EMBEDDING_MODEL != resolve_encoder_model("qwen3-emb-4b").huggingface_id:
            raise ValueError("Controlled chunk/card comparison requires the same Qwen3-Embedding-4B model.")
        if CHUNK_RAG_INDEX_PATH.resolve() == CARD_INDEX_PATH.resolve():
            raise ValueError("Chunk RAG and card RAG require distinct FAISS index paths.")
    if len(set(RAG_RETRIEVERS)) != len(RAG_RETRIEVERS):
        raise ValueError("RAG_RETRIEVERS contains duplicate strategies.")


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
        frame = sample_messages(frame, LIMIT, DATASET_SAMPLE_SEED)
        if frame.empty:
            raise ValueError(f"No eligible unique messages found for {name}.")
        frames.append(frame)
        print(f"Sampled {len(frame)} unique messages from {name}; seed={DATASET_SAMPLE_SEED}.")

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

    print(f"Loading the shared source chunk corpus from {CHUNK_LIBRARY_PATH}...")
    documents = load_chunk_documents(CHUNK_LIBRARY_PATH)
    corpus_sha256 = file_sha256(CHUNK_LIBRARY_PATH)
    shared_encoders = {}

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
                index_name="shared_card_creation_chunks",
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
        elif retriever_name in {"qwen3-emb-4b", "cards"}:
            from unito_amazon.card_creation.card_retrieval import load_card_documents
            from unito_amazon.card_creation.vector_store import CARD_INSTRUCTION, FaissEvidenceRetriever

            if retriever_name == "cards":
                retrieval_documents = load_card_documents(CARD_LIBRARY_PATH)
                validate_card_sources(retrieval_documents, documents)
                model_name = CARD_EMBEDDING_MODEL
                index_path = CARD_INDEX_PATH
                batch_size = CARD_EMBEDDING_BATCH_SIZE
            else:
                retrieval_documents = documents
                model_name = resolve_encoder_model(retriever_name).huggingface_id
                index_path = CHUNK_RAG_INDEX_PATH
                batch_size = CHUNK_RAG_EMBEDDING_BATCH_SIZE
            retriever = FaissEvidenceRetriever(
                model_name=model_name,
                batch_size=batch_size,
                device=DEVICE,
                retriever_name=f"{retriever_name}_faiss:{model_name}",
                instruction=CARD_INSTRUCTION,
            )
            # One shared 4B encoder for both indices; do not load its weights twice.
            retriever.encoder = shared_encoders.get(model_name)
            retriever.load_or_index(
                retrieval_documents, index_path, force_reindex=FORCE_REINDEX,
            )
            shared_encoders[model_name] = retriever._encoder()
        else:
            raise ValueError(
                f"Unsupported RAG retriever: {retriever_name!r}. "
                "Available: 'bm25', 'qwen3-emb-0.6b', 'qwen3-emb-4b', 'cards'."
            )

        strategy_name = f"rag_{retriever_name.replace('-', '_')}"
        generators[strategy_name] = RAGGenerator(
            llm=llm,
            retriever=retriever,
            prompt=RAGCounterSpeechPrompt(**prompt_options),
            top_k=3,
            experiment_metadata={
                "source_corpus_path": str(CHUNK_LIBRARY_PATH),
                "source_corpus_sha256": corpus_sha256,
                "source_chunk_count": len(documents),
                "retrieval_unit": "card" if retriever_name == "cards" else "chunk",
                "embedding_model": getattr(retriever, "model_name", None),
                "query_instruction": getattr(retriever, "instruction", None),
                "reranking": False,
                "retrieval_top_k": 3,
            },
        )

    return generators


def serialize_result(
    row: pd.Series,
    strategy: str,
    result,
    model_alias: str,
    tokenizer=None,
    tokenizer_name: str | None = None,
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
        "context_length": measure_context(
            result.prompt, len(result.retrieval_results), tokenizer, tokenizer_name,
        ),
        "evidence": [
            {
                "id": retrieval.document.id,
                "text": retrieval.document.text,
                "source_file": retrieval.document.metadata.get("source_file"),
                "score": retrieval.score,
                "rank": retrieval.rank,
                "metadata": retrieval.document.metadata,
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
    generators = {}
    completed = 0
    try:
        processor_or_tokenizer = getattr(llm, "tokenizer", None)
        text_tokenizer = getattr(processor_or_tokenizer, "tokenizer", processor_or_tokenizer)
        generators = build_generators(llm)
        if not generators:
            raise ValueError("No generation strategy is enabled in config.py.")
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
                            tokenizer=text_tokenizer,
                            tokenizer_name=model.huggingface_id,
                        )
                        output_file.write(
                            json.dumps(record, ensure_ascii=False) + "\n"
                        )

                completed += len(batch)
                print(
                    f"[{model.paper_name}] Generated {completed}/{len(data)} records."
                )
    finally:
        for generator in generators.values():
            retriever = getattr(generator, "retriever", None)
            if hasattr(retriever, "close"):
                retriever.close()
        generators.clear()
        llm.destroy()
        gc.collect()

    print(f"Results written to {output_path.resolve()}")


def main() -> None:
    validate_config()
    data = load_generation_data()
    if data.empty:
        raise ValueError("No dataset records match the configuration.")
    subset_path = GENERATION_OUTPUT_DIR / f"subset_{DATASET}.json"
    write_json(subset_path, {
        "dataset": DATASET, "limit_per_dataset": LIMIT, "seed": DATASET_SAMPLE_SEED,
        "items": [
            {"id": row["id"], "source": row["source"],
             "split": _optional_value(row["split"]), "hateful_message": row["hate_speech"]}
            for _, row in data.iterrows()
        ],
    })
    print(f"Dataset subset saved to {subset_path}.")

    for model_alias in GENERATION_MODELS:
        run_model(data, model_alias)


if __name__ == "__main__":
    main()
