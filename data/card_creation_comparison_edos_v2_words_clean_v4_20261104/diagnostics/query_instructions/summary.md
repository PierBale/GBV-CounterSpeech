# Query format and instruction diagnostic

Dense retrieval only (before reranking); 11 EDOS queries; no human labels.
Lower query cosine, more distinct chunks and fewer chunks shared by 3+ labels mean less hubness.
Reference for Jaccard: new_format + current_long.

## Qwen/Qwen3-Embedding-8B

| query_format | instruction | mean_query_cosine | max_query_cosine | distinct_chunks@4 | chunks_in_3plus_labels@4 | max_labels_per_chunk@4 | distinct_chunks@10 | chunks_in_3plus_labels@10 | max_labels_per_chunk@10 | distinct_chunks@100 | chunks_in_3plus_labels@100 | max_labels_per_chunk@100 | distinct_documents@10 | top_document_share@10 | statistical_share@10 | jaccard_vs_reference@10 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| old_format | no_instruction | 0.668 | 0.847 | 17 | 4 | 9 | 38 | 17 | 11 | 272 | 146 | 11 | 10 | 0.373 | 0.091 | 0.224 |
| old_format | old_verifiable_evidence | 0.597 | 0.792 | 24 | 7 | 6 | 58 | 16 | 7 | 401 | 142 | 11 | 11 | 0.200 | 0.109 | 0.418 |
| old_format | current_long | 0.609 | 0.797 | 27 | 3 | 8 | 55 | 14 | 9 | 353 | 145 | 11 | 12 | 0.227 | 0.064 | 0.650 |
| old_format | short_task_only | 0.611 | 0.826 | 29 | 5 | 5 | 54 | 13 | 7 | 352 | 153 | 11 | 12 | 0.291 | 0.045 | 0.436 |
| new_format | no_instruction | 0.587 | 0.834 | 31 | 3 | 5 | 60 | 14 | 6 | 375 | 162 | 11 | 15 | 0.318 | 0.118 | 0.224 |
| new_format | old_verifiable_evidence | 0.604 | 0.782 | 26 | 5 | 6 | 59 | 15 | 8 | 392 | 142 | 11 | 12 | 0.182 | 0.091 | 0.447 |
| new_format | current_long | 0.608 | 0.788 | 26 | 4 | 8 | 59 | 12 | 9 | 358 | 148 | 11 | 12 | 0.145 | 0.045 | 1.000 |
| new_format | short_task_only | 0.614 | 0.806 | 29 | 5 | 5 | 58 | 10 | 8 | 366 | 155 | 11 | 13 | 0.191 | 0.045 | 0.474 |

## BAAI/bge-m3

| query_format | instruction | mean_query_cosine | max_query_cosine | distinct_chunks@4 | chunks_in_3plus_labels@4 | max_labels_per_chunk@4 | distinct_chunks@10 | chunks_in_3plus_labels@10 | max_labels_per_chunk@10 | distinct_chunks@100 | chunks_in_3plus_labels@100 | max_labels_per_chunk@100 | distinct_documents@10 | top_document_share@10 | statistical_share@10 | jaccard_vs_reference@10 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| old_format | no_instruction | 0.779 | 0.905 | 17 | 4 | 9 | 30 | 16 | 11 | 257 | 138 | 11 | 10 | 0.264 | 0.091 | 0.274 |
| new_format | no_instruction | 0.688 | 0.875 | 28 | 3 | 3 | 54 | 14 | 7 | 332 | 145 | 11 | 12 | 0.191 | 0.073 | 1.000 |
