# Model annotation pilot

Import each `.qsf` into Qualtrics as a separate survey:

| Survey | Items | Completed by |
| --- | ---: | --- |
| `comune.qsf` | 45 | All three annotators |
| `marem.qsf` | 29 | marem |
| `arianna.qsf` | 29 | arianna |
| `marta.qsf` | 29 | marta |

Each annotator completes the common survey and their private survey: 74 items,
with two ratings per item (148 rating answers) and optional notes. The common
survey includes an annotator-name selector. Rating validation and the optional
notes field retain the draft's settings.

The builder clones the task block and the example block from
`Draft_best-model.qsf`. Every item retains the instruction, Faithfulness rating,
Answer Relevance rating, and notes question, in that order. Both rating scales
and their prompts are preserved. `evidence` supplies Argument;
`primary_edos_label` and `edos_definition` supply EDOS Label; `chunk.text`
supplies Paragraph. Only the leading `Section:` marker is removed: its section
title/path and all body text remain. Source HTML tables remain readable.

The two source files are the `deepseek-v4.1-flash` and `gpt-6.1-sol`
`__qwen3-embedding-8b__qwen3-reranker-8b__top15__prompt-v4/cards.jsonl` files
under `data/card_creation_comparison_edos_v2_words_clean_v4_20261104/generation`.
Select `retrieval.reranker_rank` 1–6 for each model and each of the 11 EDOS
labels: 132 cards. Cards remain separate even when their paragraphs coincide.

The 45 common cards use proportional allocation by label with the
largest-remainder method and seeded tie-breaks. The equally sized label pools
produce ten common quotas of 4 and one quota of 5. Each label includes two
OpenAI cards and two DeepSeek cards. For the five-card label, the extra card
comes from either model with equal probability, using the same seed. Cards
are sampled within each label/model group. The common survey therefore has
22 cards from one model and 23 from the other. The remaining 87 cards form
three disjoint private pools of exactly 29; for each label, private counts
differ by at most one. Private pools have no additional model-balance constraint.

All randomness uses local `random.Random` instances derived from `--seed`.
Each survey orders EDOS labels numerically and independently shuffles cards
within a label, starting from sorted logical item IDs, using
`random.Random(f"{seed}:presentation:{name}")`. Every block sets
`RandomizeQuestions` to the string `"false"`; the survey flow contains no
randomizer.

Logical question IDs combine `card_id` with an export-safe `generator_model`.
Each rating's `DataExportTag` adds its dimension to that identity. Qualtrics
internal `QuestionID` values use conventional `QID<number>` identifiers.
The four `*_question_map.csv` files and `annotation_assignment.json` link the
original card/model identifiers to both rating questions, notes, source chunks,
and presentation positions. The manifest also records allocation counts,
the seed, and SHA-256 hashes of the inputs, draft, builder, and outputs.

From the repository root, regenerate with Python 3.9 or newer:

```sh
python3 Qualtrics/model_annotation_pilot/build_model_surveys.py --seed 20261114
```

Identical existing outputs can be regenerated directly. To replace outputs
after changing the seed, draft, or builder, add `--overwrite`. Use
`--output-dir PATH` to create a separate package. `--input-files PATH PATH`
and `--template PATH` can override the defaults.

Run validation with:

```sh
python3 -m unittest discover -s Qualtrics/model_annotation_pilot/tests -v
```

Validation checks coverage, allocation and disjointness, draft settings and
source text, QSF references and flow, exported hashes, offline order, invalid
ranks, and byte-identical regeneration across processes. Qualtrics import and
visual preview must still be checked in Qualtrics.
