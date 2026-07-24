# Qwen3 chunk embeddings and MMR retrieval

This pipeline uses Hugging Face `sentence-transformers`, not Ollama.

From `Cards/Generation`, run:

```bash
python scripts/02_parse_sources.py
python scripts/03_encode_chunks.py
python scripts/08_retrieve_chunks.py \
  --query "Women are too emotional to make rational decisions."
```

`02_parse_sources.py` parses every source listed in `configs/sources.yaml` and
creates overlapping chunks in `data/processed/document_chunks.jsonl`.

`03_encode_chunks.py` encodes every chunk with
`Qwen/Qwen3-Embedding-0.6B`. The model is loaded through
`sentence-transformers`; document embeddings are L2-normalized and stored with
their chunk IDs in `data/processed/document_chunk_embeddings.npz`. The first
run downloads the model from Hugging Face.

`08_retrieve_chunks.py`:

1. encodes the query with Qwen's built-in `query` prompt;
2. selects the 100 chunks with the highest cosine similarity;
3. applies Maximal Marginal Relevance to that candidate pool;
4. returns 10 chunks.

Defaults are defined in `configs/retrieval_config.yaml`:

```yaml
embedding_model: "Qwen/Qwen3-Embedding-0.6B"
embedding_batch_size: 8
top_k: 10
candidate_pool_size: 100
mmr_lambda: 0.50
```

The MMR objective is:

```text
lambda * similarity(query, chunk)
  - (1 - lambda) * max_similarity(chunk, already_selected)
```

`lambda=0.5` is the neutral setting: relevance and novelty receive equal
weight. Tune it on a validation set if the application needs a different
precision/diversity trade-off.

The existing candidate-card extraction script is a separate generative
workflow. Qwen3-Embedding is an encoder and cannot generate card JSON.
