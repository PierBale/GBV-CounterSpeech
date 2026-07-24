# Card-Routed RAG for EDOS-specific Counterspeech

This repository builds and reviews **quote-first, EDOS-specific evidence cards** for gender-based counterspeech generation.

The current pipeline is intentionally split into four parts:

1. **Card construction**: download sources, parse them, extract 5 candidate cards per EDOS label.
2. **Card validation**: normalize cards, inspect them in a visual interface, and adjudicate expert reviews.
3. **Card retrieval**: compare basic retrieval, literature-style dense retrieval, and our card-aware retrieval.
4. **Generation from one card**: prompt Ollama to generate one counter-narrative grounded in one selected card.

The repository is local-first and can run with **Ollama**. No OpenAI API key is required.

---

## 0. Repository structure

```text
card_routed_rag_quote_first/
├── configs/
│   ├── sources.yaml
│   ├── extraction_config.yaml
│   └── retrieval_config.yaml
├── data/
│   ├── sources/
│   ├── processed/
│   ├── cards/
│   │   ├── candidates/
│   │   └── validated/
│   ├── validation/
│   ├── retrieval_outputs/
│   └── generated/
├── scripts/
│   ├── 01_download_sources.py
│   ├── 02_parse_sources.py
│   ├── 03_extract_candidate_cards.py
│   ├── 04_normalize_cards.py
│   ├── 05_prepare_expert_validation.py
│   ├── 06_adjudicate_validated_cards.py
│   ├── 08_retrieve_cards.py
│   ├── 10_card_review_app.py
│   └── 11_generate_from_card.py
├── examples/
├── requirements.txt
├── Makefile
└── README.md
```

Script `07` has been removed because the current retrieval module builds the retrieval space directly from the validated card library.

---

## 1. Install Ollama

Install Ollama from:

```text
https://ollama.com/download
```

Then pull a local model.

For lightweight tests:

```bash
ollama pull llama3.2:3b
```

For better card extraction and generation quality:

```bash
ollama pull llama3.1:8b
```

Check that Ollama is available:

```bash
ollama -v
```

If needed, start the Ollama server manually:

```bash
ollama serve
```

---

## 2. Create Python environment

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

For the visual review app:

```bash
pip install streamlit pandas
```

For dense retrieval with sentence-transformers:

```bash
pip install sentence-transformers
```

If `sentence-transformers` is not installed, the dense retrieval method falls back to TF-IDF similarity.

---

## 3. Optional: force Streamlit light theme

If the Streamlit app opens in dark mode and you want a light theme, create:

```bash
mkdir -p .streamlit
nano .streamlit/config.toml
```

Add:

```toml
[theme]
base = "light"
primaryColor = "#4f46e5"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f8fafc"
textColor = "#111827"
font = "sans serif"
```

Then restart Streamlit.

---

## 4. Test the repository

Run syntax/basic checks:

```bash
make test
```

Run the mock pipeline without Ollama:

```bash
make mock
```

Run mock retrieval:

```bash
make mock-retrieval
```

---

# Part A — Card construction

The cards are now generated with a **quote-first, EDOS-specific** approach.

The extraction logic is:

```text
EDOS label
→ targeted source passages
→ original source_quote
→ atomic claim supported by that quote
→ EDOS alignment explanation
→ candidate card
→ expert validation
```

Each EDOS fine-grained label receives an arbitrary fixed target of **5 candidate cards**.

The 11 EDOS labels are:

```text
1.1 threats of harm
1.2 incitement and encouragement of harm
2.1 descriptive attacks
2.2 aggressive and emotive attacks
2.3 dehumanising attacks & overt sexual objectification
3.1 casual use of gendered slurs, profanities, and insults
3.2 immutable gender differences and gender stereotypes
3.3 backhanded gendered compliments
3.4 condescending explanations or unwelcome advice
4.1 supporting mistreatment of individual women
4.2 supporting systemic discrimination against women as a group
```

---

## 5. Download source documents

```bash
python scripts/01_download_sources.py \
  --config configs/sources.yaml
```

Source documents are saved under:

```text
data/sources/
```

---

## 6. Parse source documents

```bash
python scripts/02_parse_sources.py \
  --sources configs/sources.yaml \
  --config configs/extraction_config.yaml \
  --output data/processed/document_chunks.jsonl
```

This produces:

```text
data/processed/document_chunks.jsonl
```

Each row contains a source passage plus source metadata.

---

## 7. Extract quote-first candidate cards with Ollama

Small test run:

```bash
python scripts/03_extract_candidate_cards.py \
  --chunks data/processed/document_chunks.jsonl \
  --output data/cards/candidates/candidate_cards.jsonl \
  --model llama3.2:3b \
  --cards-per-label 5 \
  --limit 10
```

Full run:

```bash
python scripts/03_extract_candidate_cards.py \
  --chunks data/processed/document_chunks.jsonl \
  --output data/cards/candidates/candidate_cards.jsonl \
  --model llama3.2:3b \
  --cards-per-label 5
```

The output is:

```text
data/cards/candidates/candidate_cards.jsonl
```

---

## 8. Card JSON schema

Each card has the following minimal schema:

```json
{
  "card_id": "...",
  "status": "candidate",
  "source": {
    "source_id": "...",
    "title": "...",
    "publisher": "...",
    "year": 2021,
    "page": 12,
    "section": "...",
    "url": "..."
  },
  "source_quote": "short original fragment from the source",
  "claim": "atomic claim directly supported by the source_quote",
  "primary_edos_label": "3.2 immutable gender differences and gender stereotypes",
  "secondary_edos_labels": [],
  "edos_alignment": "why this card is useful for this specific EDOS label",
  "retrieval_keywords": ["gender stereotypes", "roles", "sexism"],
  "validation": {
    "status": "not_validated",
    "faithfulness": null,
    "edos_alignment": null,
    "usefulness": null,
    "notes": null
  }
}
```

The following fields were intentionally removed from the previous prototype:

```text
card_type
source_checksum
evidence_type
mapped_speaker_positions
risk_notes
risk_level
source_reliability
countering_use
compatible_strategies
```

---

## 9. Normalize cards and generate coverage report

```bash
python scripts/04_normalize_cards.py \
  --input data/cards/candidates/candidate_cards.jsonl \
  --output data/cards/candidates/candidate_cards_normalized.jsonl \
  --dedupe
```

This produces:

```text
data/cards/candidates/candidate_cards_normalized.jsonl
data/cards/candidates/candidate_cards_coverage.csv
```

The coverage report checks how many cards are available for each EDOS label.

Expected target:

```text
5 candidate cards per EDOS label
55 candidate cards total
```

---

# Part B — Card validation

## 10. Prepare expert validation CSV

```bash
python scripts/05_prepare_expert_validation.py \
  --input data/cards/candidates/candidate_cards_normalized.jsonl \
  --output data/validation/expert_validation_sheet.csv
```

This creates a CSV that experts can fill manually.

Suggested validation dimensions:

```text
faithfulness
edos_alignment
usefulness
notes
```

---

## 11. Visual card review interface

Add the Streamlit module:

```text
scripts/10_card_review_app.py
```

Run:

```bash
streamlit run scripts/10_card_review_app.py
```

By default, the app reads:

```text
data/cards/candidates/candidate_cards_normalized.jsonl
```

and saves local reviews to:

```text
data/validation/card_review_annotations.csv
```

The interface lets reviewers inspect one card at a time and validate each field:

```text
source
source_quote
claim
primary_edos_label
secondary_edos_labels
edos_alignment
retrieval_keywords
```

Reviewers can assign:

```text
OK
Revise
Reject
Unsure
```

and can add comments or revised field values.

For multiple annotators, use different output paths in the sidebar, for example:

```text
data/validation/card_review_annotations_annotator_A.csv
data/validation/card_review_annotations_annotator_B.csv
data/validation/card_review_annotations_annotator_C.csv
```

---

## 12. Adjudicate validated cards

Once the validation CSV is completed:

```bash
python scripts/06_adjudicate_validated_cards.py \
  --cards data/cards/candidates/candidate_cards_normalized.jsonl \
  --validation data/validation/expert_validation_sheet.csv \
  --output data/cards/validated/validated_cards.jsonl \
  --report data/validation/validation_report.csv
```

Output:

```text
data/cards/validated/validated_cards.jsonl
data/validation/validation_report.csv
```

This file becomes the frozen card library used for retrieval and generation.

---

# Part C — Retrieval only

The new retrieval module is:

```text
scripts/08_retrieve_cards.py
```

It does **not** generate counter-narratives. It only retrieves cards.

It supports three methods:

```text
basic
dense_mmr
card_aware
```

and one comparison mode:

```text
all
```

---

## 13. Basic retrieval

Basic retrieval uses lexical similarity over the card text.

```bash
python scripts/08_retrieve_cards.py \
  --method basic \
  --cards data/cards/validated/validated_cards.jsonl \
  --hate-speech "Women are too emotional to make rational decisions." \
  --top-k 5 \
  --output data/retrieval_outputs/basic_example.json
```

This is the simplest baseline.

---

## 14. Literature-style retrieval: dense + MMR

Dense retrieval uses sentence embeddings when `sentence-transformers` is installed.  
MMR reranking reduces redundancy among selected cards.

```bash
python scripts/08_retrieve_cards.py \
  --method dense_mmr \
  --cards data/cards/validated/validated_cards.jsonl \
  --hate-speech "Women are too emotional to make rational decisions." \
  --top-k 5 \
  --output data/retrieval_outputs/dense_mmr_example.json
```

Optional dense model:

```bash
python scripts/08_retrieve_cards.py \
  --method dense_mmr \
  --cards data/cards/validated/validated_cards.jsonl \
  --hate-speech "Women are too emotional to make rational decisions." \
  --embedding-model sentence-transformers/all-MiniLM-L6-v2 \
  --top-k 5 \
  --output data/retrieval_outputs/dense_mmr_example.json
```

---

## 15. Our retrieval method: Card-Aware Retrieval

Card-aware retrieval uses the card structure directly.

It scores candidate cards using:

```text
score(card) =
    w_lex     * lexical_similarity
  + w_sem     * semantic_similarity
  + w_primary * primary_EDOS_match
  + w_second  * secondary_EDOS_match
  + w_valid   * validation_quality
  + w_kw      * keyword_overlap
  - w_red     * redundancy_penalty
```

It prioritizes:

```text
semantic relevance
lexical relevance
EDOS-label compatibility
validated card quality
retrieval keyword overlap
non-redundant card selection
```

Example:

```bash
python scripts/08_retrieve_cards.py \
  --method card_aware \
  --cards data/cards/validated/validated_cards.jsonl \
  --hate-speech "Women are too emotional to make rational decisions." \
  --edos-label "3.2 immutable gender differences and gender stereotypes" \
  --top-k 5 \
  --output data/retrieval_outputs/card_aware_example.json
```

---

## 16. Run all retrieval methods for comparison

```bash
python scripts/08_retrieve_cards.py \
  --method all \
  --cards data/cards/validated/validated_cards.jsonl \
  --hate-speech "Women are too emotional to make rational decisions." \
  --edos-label "3.2 immutable gender differences and gender stereotypes" \
  --top-k 5 \
  --output data/retrieval_outputs/all_methods_example.json
```

This produces a single JSON file containing the outputs of:

```text
basic
dense_mmr
card_aware
```

---

# Part D — Generation from one card

The generation module is:

```text
scripts/11_generate_from_card.py
```

It receives:

```text
one hate-speech sentence
one selected card
one local Ollama model
```

and produces a grounded counter-narrative.

This module does **not** perform retrieval. Use script `08` first to select cards.

---

## 17. Generate from a card in a JSONL library

```bash
python scripts/11_generate_from_card.py \
  --hate-speech "Women are too emotional to make rational decisions." \
  --cards data/cards/validated/validated_cards.jsonl \
  --card-id EPRS_2021_EDOS_3_2_001 \
  --model llama3.2:3b \
  --output data/generated/example_generation.json
```

---

## 18. Generate from a standalone card JSON

```bash
python scripts/11_generate_from_card.py \
  --hate-speech "Women are too emotional to make rational decisions." \
  --card-json examples/example_card.json \
  --model llama3.2:3b \
  --output data/generated/example_generation.json
```

---

## 19. Optional generation parameters

Specify a counterspeech strategy:

```bash
python scripts/11_generate_from_card.py \
  --hate-speech "Women are too emotional to make rational decisions." \
  --cards data/cards/validated/validated_cards.jsonl \
  --card-id EPRS_2021_EDOS_3_2_001 \
  --strategy "Fact-Checking" \
  --model llama3.2:3b \
  --output data/generated/example_generation.json
```

Generate in Italian:

```bash
python scripts/11_generate_from_card.py \
  --hate-speech "Women are too emotional to make rational decisions." \
  --cards data/cards/validated/validated_cards.jsonl \
  --card-id EPRS_2021_EDOS_3_2_001 \
  --language Italian \
  --model llama3.2:3b \
  --output data/generated/example_generation_it.json
```

Limit length:

```bash
python scripts/11_generate_from_card.py \
  --hate-speech "Women are too emotional to make rational decisions." \
  --cards data/cards/validated/validated_cards.jsonl \
  --card-id EPRS_2021_EDOS_3_2_001 \
  --max-sentences 2 \
  --model llama3.2:3b \
  --output data/generated/example_generation.json
```

Show the prompt:

```bash
python scripts/11_generate_from_card.py \
  --hate-speech "Women are too emotional to make rational decisions." \
  --cards data/cards/validated/validated_cards.jsonl \
  --card-id EPRS_2021_EDOS_3_2_001 \
  --model llama3.2:3b \
  --show-prompt
```

Allow explicit source mention:

```bash
python scripts/11_generate_from_card.py \
  --hate-speech "Women are too emotional to make rational decisions." \
  --cards data/cards/validated/validated_cards.jsonl \
  --card-id EPRS_2021_EDOS_3_2_001 \
  --cite-source \
  --model llama3.2:3b \
  --output data/generated/example_generation.json
```

---

## 20. Generation output format

The script writes JSON:

```json
{
  "input": {
    "hate_speech": "...",
    "strategy": "Fact-Checking",
    "language": "English",
    "max_sentences": 2
  },
  "card": {
    "card_id": "...",
    "primary_edos_label": "...",
    "claim": "...",
    "source_quote": "...",
    "source": {}
  },
  "generation": {
    "counter_narrative": "...",
    "used_claim": "...",
    "grounding_note": "..."
  },
  "model": {
    "provider": "ollama",
    "model": "llama3.1:8b",
    "temperature": 0.2,
    "ollama_url": "http://localhost:11434"
  },
  "raw_model_output": "..."
}
```

---

# Recommended end-to-end workflow

## Candidate cards

```bash
python scripts/01_download_sources.py --config configs/sources.yaml

python scripts/02_parse_sources.py \
  --sources configs/sources.yaml \
  --config configs/extraction_config.yaml \
  --output data/processed/document_chunks.jsonl

python scripts/03_extract_candidate_cards.py \
  --chunks data/processed/document_chunks.jsonl \
  --output data/cards/candidates/candidate_cards.jsonl \
  --model llama3.1:8b \
  --cards-per-label 5

python scripts/04_normalize_cards.py \
  --input data/cards/candidates/candidate_cards.jsonl \
  --output data/cards/candidates/candidate_cards_normalized.jsonl \
  --dedupe

python scripts/05_prepare_expert_validation.py \
  --input data/cards/candidates/candidate_cards_normalized.jsonl \
  --output data/validation/expert_validation_sheet.csv
```

## Visual review

```bash
streamlit run scripts/10_card_review_app.py
```

## Adjudication

```bash
python scripts/06_adjudicate_validated_cards.py \
  --cards data/cards/candidates/candidate_cards_normalized.jsonl \
  --validation data/validation/expert_validation_sheet.csv \
  --output data/cards/validated/validated_cards.jsonl \
  --report data/validation/validation_report.csv
```

## Retrieval comparison

```bash
python scripts/08_retrieve_cards.py \
  --method all \
  --cards data/cards/validated/validated_cards.jsonl \
  --hate-speech "Women are too emotional to make rational decisions." \
  --edos-label "3.2 immutable gender differences and gender stereotypes" \
  --top-k 5 \
  --output data/retrieval_outputs/all_methods_example.json
```

## Generation from selected card

```bash
python scripts/11_generate_from_card.py \
  --hate-speech "Women are too emotional to make rational decisions." \
  --cards data/cards/validated/validated_cards.jsonl \
  --card-id EPRS_2021_EDOS_3_2_001 \
  --strategy "Fact-Checking" \
  --model llama3.2:3b \
  --output data/generated/example_generation.json
```

---

# Notes

- Candidate cards are not gold data.
- Cards become usable for experiments only after expert validation.
- Gold CONAN counter-narratives should be used only for evaluation, not for card extraction or retrieval.
- Retrieval and generation are intentionally separate modules.
- The current system retrieves cards; it does not yet automatically select the best final card for generation.
