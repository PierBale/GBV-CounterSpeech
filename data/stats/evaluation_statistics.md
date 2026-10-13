# Evaluation statistics by dataset

Automatic metrics: MultiCONAN with reference only.

# Dataset: edos

# Evaluation statistics

JudgeLM protocol: pairwise comparisons between generation systems on this dataset only.
Response 1 is the indicated baseline; Response 2 is the challenger. No gold-reference comparisons are added.
Automatic-metric variance is the sample variance (denominator `n - 1`).

## Overview

- Pairwise judge files: 1
- Automatic-metric files: 1
- Valid pairwise comparisons: 450 / 450
- Pairwise parse failures: 0
- Recovered from saved raw responses: 0
- Unique generations with metrics: 0

## Automatic metrics comparison

Automatic metrics are reported only for MultiCONAN records with a reference; no automatic metrics are reported for EDOS. Each metric has separate mean and sample-variance columns.

_No data available._

## JudgeLM wins: RAG vs No-RAG

Cells report RAG wins out of 150 valid comparisons per cell, followed by RAG, baseline and tie percentages over valid judgments. `(lost)` means the baseline won more comparisons; `(equal wins)` means equal RAG and baseline wins.

| Model | Chunk-RAG (Qwen3-Emb-4B) | Card-RAG (Qwen3-Emb-4B) |
| --- | --- | --- |
| Llama-3.1-8B | 65 (lost) - RAG 43.33%; baseline 56.00%; tie 0.67% | 67 (lost) - RAG 44.67%; baseline 54.67%; tie 0.67% |

## JudgeLM pairwise results

| Generator | Baseline | RAG challenger | Judge | Valid / total | Reparsed | RAG wins | RAG win % | Baseline wins | Baseline win % | Ties | Tie % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Llama-3.1-8B | rag_qwen3_emb_4b | rag_cards | JudgeLM-7B | 150 / 150 | 0 | 55 | 36.67% | 95 | 63.33% | 0 | 0.00% |
| Llama-3.1-8B | without_rag | rag_cards | JudgeLM-7B | 150 / 150 | 0 | 67 | 44.67% | 82 | 54.67% | 1 | 0.67% |
| Llama-3.1-8B | without_rag | rag_qwen3_emb_4b | JudgeLM-7B | 150 / 150 | 0 | 65 | 43.33% | 84 | 56.00% | 1 | 0.67% |

## Automatic metrics by system

# Dataset: multiconan

# Evaluation statistics

JudgeLM protocol: pairwise comparisons between generation systems on this dataset only.
Response 1 is the indicated baseline; Response 2 is the challenger. No gold-reference comparisons are added.
Automatic-metric variance is the sample variance (denominator `n - 1`).

## Overview

- Pairwise judge files: 1
- Automatic-metric files: 1
- Valid pairwise comparisons: 450 / 450
- Pairwise parse failures: 0
- Recovered from saved raw responses: 0
- Unique generations with metrics: 450

## Automatic metrics comparison

Automatic metrics are reported only for MultiCONAN records with a reference; no automatic metrics are reported for EDOS. Each metric has separate mean and sample-variance columns.

| Dataset | Model | System | BLEU-4 mean | BLEU-4 variance | METEOR mean | METEOR variance | ROUGE-L mean | ROUGE-L variance | BERTScore-F1 mean | BERTScore-F1 variance | BERTScore-F1-rescaled mean | BERTScore-F1-rescaled variance | Distinct-1 mean | Distinct-1 variance | Distinct-2 mean | Distinct-2 variance | Repetition Rate mean | Repetition Rate variance |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| multiconan | Llama-3.1-8B | Card-RAG (Qwen3-Emb-4B) | 0.054810 | 0.000602 | 0.165491 | 0.007987 | 0.144604 | 0.004721 | 0.870527 | 0.000278 | 0.232865 | 0.009760 | 0.871314 | 0.001823 | 0.998294 | 0.000043 | 0.000000 | 0.000000 |
| multiconan | Llama-3.1-8B | Chunk-RAG (Qwen3-Emb-4B) | 0.047157 | 0.000390 | 0.171203 | 0.006282 | 0.138588 | 0.003814 | 0.867231 | 0.000272 | 0.213335 | 0.009535 | 0.851915 | 0.002673 | 0.995380 | 0.000193 | 0.001300 | 0.000129 |
| multiconan | Llama-3.1-8B | No-RAG | 0.054895 | 0.000644 | 0.157180 | 0.006361 | 0.132609 | 0.003931 | 0.869906 | 0.000259 | 0.229182 | 0.009097 | 0.882892 | 0.002871 | 0.998555 | 0.000044 | 0.000000 | 0.000000 |

## JudgeLM wins: RAG vs No-RAG

Cells report RAG wins out of 150 valid comparisons per cell, followed by RAG, baseline and tie percentages over valid judgments. `(lost)` means the baseline won more comparisons; `(equal wins)` means equal RAG and baseline wins.

| Model | Chunk-RAG (Qwen3-Emb-4B) | Card-RAG (Qwen3-Emb-4B) |
| --- | --- | --- |
| Llama-3.1-8B | 107 - RAG 71.33%; baseline 21.33%; tie 7.33% | 102 - RAG 68.00%; baseline 19.33%; tie 12.67% |

## JudgeLM pairwise results

| Generator | Baseline | RAG challenger | Judge | Valid / total | Reparsed | RAG wins | RAG win % | Baseline wins | Baseline win % | Ties | Tie % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Llama-3.1-8B | rag_qwen3_emb_4b | rag_cards | JudgeLM-7B | 150 / 150 | 0 | 74 | 49.33% | 46 | 30.67% | 30 | 20.00% |
| Llama-3.1-8B | without_rag | rag_cards | JudgeLM-7B | 150 / 150 | 0 | 102 | 68.00% | 29 | 19.33% | 19 | 12.67% |
| Llama-3.1-8B | without_rag | rag_qwen3_emb_4b | JudgeLM-7B | 150 / 150 | 0 | 107 | 71.33% | 32 | 21.33% | 11 | 7.33% |

## Automatic metrics by system

### Llama-3.1-8B - rag_cards

| Metric | n | Mean | Variance | Std. dev. |
| --- | --- | --- | --- | --- |
| bleu_4 | 150 | 0.054810 | 0.000602 | 0.024537 |
| rouge_l | 150 | 0.144604 | 0.004721 | 0.068712 |
| meteor | 150 | 0.165491 | 0.007987 | 0.089370 |
| bertscore_precision | 150 | 0.866014 | 0.000219 | 0.014797 |
| bertscore_recall | 150 | 0.875241 | 0.000474 | 0.021780 |
| bertscore_f1 | 150 | 0.870527 | 0.000278 | 0.016673 |
| bertscore_rescaled_precision | 150 | 0.204803 | 0.007712 | 0.087817 |
| bertscore_rescaled_recall | 150 | 0.259615 | 0.016706 | 0.129252 |
| bertscore_rescaled_f1 | 150 | 0.232865 | 0.009760 | 0.098791 |
| distinct_1 | 150 | 0.871314 | 0.001823 | 0.042697 |
| distinct_2 | 150 | 0.998294 | 0.000043 | 0.006549 |
| repetition_rate | 150 | 0.000000 | 0.000000 | 0.000000 |

### Llama-3.1-8B - rag_qwen3_emb_4b

| Metric | n | Mean | Variance | Std. dev. |
| --- | --- | --- | --- | --- |
| bleu_4 | 150 | 0.047157 | 0.000390 | 0.019744 |
| rouge_l | 150 | 0.138588 | 0.003814 | 0.061756 |
| meteor | 150 | 0.171203 | 0.006282 | 0.079260 |
| bertscore_precision | 150 | 0.860402 | 0.000232 | 0.015241 |
| bertscore_recall | 150 | 0.874333 | 0.000456 | 0.021349 |
| bertscore_f1 | 150 | 0.867231 | 0.000272 | 0.016480 |
| bertscore_rescaled_precision | 150 | 0.171495 | 0.008181 | 0.090451 |
| bertscore_rescaled_recall | 150 | 0.254228 | 0.016052 | 0.126697 |
| bertscore_rescaled_f1 | 150 | 0.213335 | 0.009535 | 0.097645 |
| distinct_1 | 150 | 0.851915 | 0.002673 | 0.051706 |
| distinct_2 | 150 | 0.995380 | 0.000193 | 0.013887 |
| repetition_rate | 150 | 0.001300 | 0.000129 | 0.011355 |

### Llama-3.1-8B - without_rag

| Metric | n | Mean | Variance | Std. dev. |
| --- | --- | --- | --- | --- |
| bleu_4 | 150 | 0.054895 | 0.000644 | 0.025382 |
| rouge_l | 150 | 0.132609 | 0.003931 | 0.062694 |
| meteor | 150 | 0.157180 | 0.006361 | 0.079757 |
| bertscore_precision | 150 | 0.867012 | 0.000212 | 0.014569 |
| bertscore_recall | 150 | 0.872980 | 0.000448 | 0.021169 |
| bertscore_f1 | 150 | 0.869906 | 0.000259 | 0.016097 |
| bertscore_rescaled_precision | 150 | 0.210727 | 0.007476 | 0.086464 |
| bertscore_rescaled_recall | 150 | 0.246197 | 0.015782 | 0.125627 |
| bertscore_rescaled_f1 | 150 | 0.229182 | 0.009097 | 0.095378 |
| distinct_1 | 150 | 0.882892 | 0.002871 | 0.053582 |
| distinct_2 | 150 | 0.998555 | 0.000044 | 0.006664 |
| repetition_rate | 150 | 0.000000 | 0.000000 | 0.000000 |
