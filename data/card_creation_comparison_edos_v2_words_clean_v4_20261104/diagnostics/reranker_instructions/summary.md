# Qwen3-Reranker instruction diagnostic

Re-scores the run's dense top-100 candidates; no human labels.
own_label_rank@4: rank (1-11) of the selecting label when the passage is scored against all 11 queries; 1 = label-specific, ~6 = chance.
jaccard_vs_qwen_run must be ~1 for 'current' (reproduces the run).

## Candidates from qwen3-embedding-8b

| instruction | distinct_chunks@4 | chunks_in_3plus_labels@4 | max_labels_per_chunk@4 | distinct_chunks@10 | chunks_in_3plus_labels@10 | max_labels_per_chunk@10 | distinct_documents@4 | top_document_share@4 | own_label_rank@4 | own_label_top1@4 | jaccard_vs_bge_reference@4 | jaccard_vs_qwen_run@4 | mean_top1_score |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen_default | 32 | 3 | 5 | 59 | 12 | 8 | 12 | 0.205 | 2.32 | 0.545 | 0.212 | 0.186 | 0.847 |
| old_verifiable_evidence | 28 | 2 | 10 | 59 | 12 | 11 | 9 | 0.273 | 3.23 | 0.432 | 0.217 | 0.328 | 0.567 |
| current | 16 | 5 | 10 | 31 | 14 | 11 | 5 | 0.5 | 4.11 | 0.295 | 0.126 | 1.0 | 0.858 |
| behaviour_specific | 32 | 3 | 4 | 63 | 9 | 8 | 11 | 0.182 | 2.14 | 0.636 | 0.154 | 0.202 | 0.75 |

## Candidates from bge-m3

| instruction | distinct_chunks@4 | chunks_in_3plus_labels@4 | max_labels_per_chunk@4 | distinct_chunks@10 | chunks_in_3plus_labels@10 | max_labels_per_chunk@10 | distinct_documents@4 | top_document_share@4 | own_label_rank@4 | own_label_top1@4 | jaccard_vs_bge_reference@4 | jaccard_vs_qwen_run@4 | mean_top1_score |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen_default | 24 | 7 | 6 | 55 | 14 | 8 | 12 | 0.227 | 2.66 | 0.477 | 0.199 | 0.197 | 0.843 |
| old_verifiable_evidence | 24 | 2 | 11 | 53 | 10 | 11 | 8 | 0.318 | 3.57 | 0.341 | 0.228 | 0.455 | 0.473 |
| current | 16 | 4 | 11 | 38 | 16 | 11 | 5 | 0.455 | 4.39 | 0.273 | 0.15 | 1.0 | 0.85 |
| behaviour_specific | 25 | 6 | 6 | 57 | 12 | 8 | 11 | 0.227 | 2.75 | 0.477 | 0.158 | 0.228 | 0.731 |
