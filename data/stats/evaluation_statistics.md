# Evaluation statistics

The variance reported below is the sample variance (denominator `n - 1`).

## Overview

- Judge files: 1
- Automatic-metric files: 1
- Valid judgments: 600 / 600
- Unique generations with metrics: 600

## Overall LLM-as-a-judge dimensions

| Dimension | n | Mean | Variance | Std. dev. |
| --- | --- | --- | --- | --- |
| relevance | 600 | 4.518333 | 0.904505 | 0.951055 |
| respectfulness | 600 | 4.968333 | 0.034054 | 0.184537 |
| persuasiveness | 600 | 3.708333 | 0.587577 | 0.766535 |
| self_contained | 600 | 4.788333 | 0.377493 | 0.614405 |
| conciseness | 600 | 4.550000 | 1.429883 | 1.195777 |
| evidence_grounding | 400 | 1.760000 | 1.335739 | 1.155742 |
| overall | 600 | 3.476667 | 0.673912 | 0.820921 |

## Systems

### Llama-3.1-8B - rag_bm25

| Valid judgments | Total judgments | Parse failures |
| --- | --- | --- |
| 200 | 200 | 0 |

#### LLM-as-a-judge dimensions

| Dimension | n | Mean | Variance | Std. dev. |
| --- | --- | --- | --- | --- |
| relevance | 200 | 4.150000 | 1.575377 | 1.255140 |
| respectfulness | 200 | 4.950000 | 0.057789 | 0.240393 |
| persuasiveness | 200 | 3.420000 | 0.817688 | 0.904261 |
| self_contained | 200 | 4.720000 | 0.473970 | 0.688455 |
| conciseness | 200 | 4.355000 | 1.888417 | 1.374197 |
| evidence_grounding | 200 | 1.640000 | 1.186332 | 1.089189 |
| overall | 200 | 3.055000 | 0.625101 | 0.790633 |

#### Automatic metrics

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

| Valid judgments | Total judgments | Parse failures |
| --- | --- | --- |
| 200 | 200 | 0 |

#### LLM-as-a-judge dimensions

| Dimension | n | Mean | Variance | Std. dev. |
| --- | --- | --- | --- | --- |
| relevance | 200 | 4.465000 | 0.712337 | 0.844000 |
| respectfulness | 200 | 4.955000 | 0.043191 | 0.207824 |
| persuasiveness | 200 | 3.650000 | 0.540201 | 0.734984 |
| self_contained | 200 | 4.645000 | 0.591935 | 0.769373 |
| conciseness | 200 | 4.295000 | 2.108518 | 1.452074 |
| evidence_grounding | 200 | 1.880000 | 1.462915 | 1.209510 |
| overall | 200 | 3.320000 | 0.660905 | 0.812960 |

#### Automatic metrics

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

| Valid judgments | Total judgments | Parse failures |
| --- | --- | --- |
| 200 | 200 | 0 |

#### LLM-as-a-judge dimensions

| Dimension | n | Mean | Variance | Std. dev. |
| --- | --- | --- | --- | --- |
| relevance | 200 | 4.940000 | 0.116985 | 0.342031 |
| respectfulness | 200 | 5.000000 | 0.000000 | 0.000000 |
| persuasiveness | 200 | 4.055000 | 0.202990 | 0.450544 |
| self_contained | 200 | 5.000000 | 0.000000 | 0.000000 |
| conciseness | 200 | 5.000000 | 0.000000 | 0.000000 |
| evidence_grounding | 0 | n/a | n/a | n/a |
| overall | 200 | 4.055000 | 0.202990 | 0.450544 |

#### Automatic metrics

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
