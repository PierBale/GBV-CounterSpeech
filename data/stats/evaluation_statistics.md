# Evaluation statistics

JudgeLM protocol: pairwise RAG vs No-RAG and MultiCONAN reference, following the paper prompt.
Response 1 is the No-RAG or MultiCONAN baseline; Response 2 is the RAG challenger.
Automatic-metric variance is the sample variance (denominator `n - 1`).

## Overview

- Pairwise judge files: 1
- Automatic-metric files: 1
- Valid pairwise comparisons: 738 / 800
- Pairwise parse failures: 62
- Recovered from saved raw responses: 701
- Unique generations with metrics: 600

## Automatic metrics comparison

Reference-based metrics are populated only for records sourced from MultiCONAN; reference-free metrics are reported for every system. Each metric has separate mean and sample-variance columns.

| Model | System | BLEU-4 mean | BLEU-4 variance | METEOR mean | METEOR variance | ROUGE-L mean | ROUGE-L variance | BERTScore-F1 mean | BERTScore-F1 variance | BERTScore-F1-rescaled mean | BERTScore-F1-rescaled variance | Distinct-1 mean | Distinct-1 variance | Distinct-2 mean | Distinct-2 variance | Repetition Rate mean | Repetition Rate variance |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Llama-3.1-8B | BM25 | 0.043540 | 0.000180 | 0.153907 | 0.002929 | 0.125933 | 0.002450 | 0.862076 | 0.000214 | 0.182788 | 0.007503 | 0.855123 | 0.003124 | 0.995852 | 0.000126 | 0.000000 | 0.000000 |
| Llama-3.1-8B | Qwen3-Emb-0.6B | 0.043121 | 0.000223 | 0.159598 | 0.004113 | 0.130114 | 0.002456 | 0.862742 | 0.000215 | 0.186736 | 0.007558 | 0.842068 | 0.002995 | 0.993205 | 0.000308 | 0.000849 | 0.000144 |
| Llama-3.1-8B | No-RAG | 0.049275 | 0.000359 | 0.154116 | 0.006127 | 0.135476 | 0.003212 | 0.865802 | 0.000281 | 0.204864 | 0.009862 | 0.874581 | 0.002838 | 0.995811 | 0.000168 | 0.000754 | 0.000058 |

## JudgeLM wins: RAG vs No-RAG

Cells report RAG wins out of 176 valid comparisons per cell, followed by RAG, baseline and tie percentages over valid judgments. `(lost)` means the baseline won more comparisons; `(equal wins)` means equal RAG and baseline wins.

| Model | BM25 | Qwen3-Emb-0.6B |
| --- | --- | --- |
| Llama-3.1-8B | 123 - RAG 69.89%; baseline 23.30%; tie 6.82% | 131 - RAG 74.43%; baseline 18.18%; tie 7.39% |

## JudgeLM wins: RAG vs MultiCONAN reference

Cells report RAG wins over the valid comparisons in each cell, followed by RAG, baseline and tie percentages over valid judgments. `(lost)` means the baseline won more comparisons; `(equal wins)` means equal RAG and baseline wins.

| Model | BM25 | Qwen3-Emb-0.6B |
| --- | --- | --- |
| Llama-3.1-8B | 180 - RAG 94.24%; baseline 2.62%; tie 3.14% | 183 - RAG 93.85%; baseline 3.08%; tie 3.08% |

## JudgeLM pairwise results

| Generator | Baseline | RAG challenger | Judge | Valid / total | Reparsed | RAG wins | RAG win % | Baseline wins | Baseline win % | Ties | Tie % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Llama-3.1-8B | multiconan_reference | rag_bm25 | JudgeLM-7B | 191 / 200 | 185 | 180 | 94.24% | 5 | 2.62% | 6 | 3.14% |
| Llama-3.1-8B | multiconan_reference | rag_qwen3_emb_0.6b | JudgeLM-7B | 195 / 200 | 189 | 183 | 93.85% | 6 | 3.08% | 6 | 3.08% |
| Llama-3.1-8B | without_rag | rag_bm25 | JudgeLM-7B | 176 / 200 | 164 | 123 | 69.89% | 41 | 23.30% | 12 | 6.82% |
| Llama-3.1-8B | without_rag | rag_qwen3_emb_0.6b | JudgeLM-7B | 176 / 200 | 163 | 131 | 74.43% | 32 | 18.18% | 13 | 7.39% |

## Automatic metrics by system

### Llama-3.1-8B - rag_bm25

| Metric | n | Mean | Variance | Std. dev. |
| --- | --- | --- | --- | --- |
| bleu_4 | 200 | 0.043540 | 0.000180 | 0.013422 |
| rouge_l | 200 | 0.125933 | 0.002450 | 0.049493 |
| meteor | 200 | 0.153907 | 0.002929 | 0.054123 |
| bertscore_precision | 200 | 0.857653 | 0.000175 | 0.013245 |
| bertscore_recall | 200 | 0.866715 | 0.000401 | 0.020026 |
| bertscore_f1 | 200 | 0.862076 | 0.000214 | 0.014619 |
| bertscore_rescaled_precision | 200 | 0.155180 | 0.006179 | 0.078608 |
| bertscore_rescaled_recall | 200 | 0.209019 | 0.014124 | 0.118844 |
| bertscore_rescaled_f1 | 200 | 0.182788 | 0.007503 | 0.086621 |
| distinct_1 | 200 | 0.855123 | 0.003124 | 0.055895 |
| distinct_2 | 200 | 0.995852 | 0.000126 | 0.011210 |
| repetition_rate | 200 | 0.000000 | 0.000000 | 0.000000 |

### Llama-3.1-8B - rag_qwen3_emb_0.6b

| Metric | n | Mean | Variance | Std. dev. |
| --- | --- | --- | --- | --- |
| bleu_4 | 200 | 0.043121 | 0.000223 | 0.014919 |
| rouge_l | 200 | 0.130114 | 0.002456 | 0.049562 |
| meteor | 200 | 0.159598 | 0.004113 | 0.064130 |
| bertscore_precision | 200 | 0.857762 | 0.000176 | 0.013261 |
| bertscore_recall | 200 | 0.867998 | 0.000447 | 0.021142 |
| bertscore_f1 | 200 | 0.862742 | 0.000215 | 0.014673 |
| bertscore_rescaled_precision | 200 | 0.155829 | 0.006194 | 0.078704 |
| bertscore_rescaled_recall | 200 | 0.216635 | 0.015742 | 0.125469 |
| bertscore_rescaled_f1 | 200 | 0.186736 | 0.007558 | 0.086937 |
| distinct_1 | 200 | 0.842068 | 0.002995 | 0.054726 |
| distinct_2 | 200 | 0.993205 | 0.000308 | 0.017551 |
| repetition_rate | 200 | 0.000849 | 0.000144 | 0.012001 |

### Llama-3.1-8B - without_rag

| Metric | n | Mean | Variance | Std. dev. |
| --- | --- | --- | --- | --- |
| bleu_4 | 200 | 0.049275 | 0.000359 | 0.018950 |
| rouge_l | 200 | 0.135476 | 0.003212 | 0.056671 |
| meteor | 200 | 0.154116 | 0.006127 | 0.078278 |
| bertscore_precision | 200 | 0.863974 | 0.000231 | 0.015212 |
| bertscore_recall | 200 | 0.867807 | 0.000478 | 0.021868 |
| bertscore_f1 | 200 | 0.865802 | 0.000281 | 0.016761 |
| bertscore_rescaled_precision | 200 | 0.192695 | 0.008151 | 0.090281 |
| bertscore_rescaled_recall | 200 | 0.215500 | 0.016841 | 0.129775 |
| bertscore_rescaled_f1 | 200 | 0.204864 | 0.009862 | 0.099308 |
| distinct_1 | 200 | 0.874581 | 0.002838 | 0.053276 |
| distinct_2 | 200 | 0.995811 | 0.000168 | 0.012968 |
| repetition_rate | 200 | 0.000754 | 0.000058 | 0.007602 |
