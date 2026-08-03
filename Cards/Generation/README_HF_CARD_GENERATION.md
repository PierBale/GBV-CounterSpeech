# Multi-model Hugging Face card generation

This pipeline generates quote-grounded candidate cards from
`data/retrieval_outputs/edos_label_chunks.json`. It does not use Ollama.

Configured models:

- `Qwen/Qwen3.5-9B`
- `mistralai/Ministral-3-8B-Instruct-2512-BF16`
- `meta-llama/Llama-3.1-8B-Instruct`

The models are loaded sequentially. Each model processes the 10 MMR-retrieved
chunks for every EDOS Task C label, is unloaded, and then the next model is
loaded. With 11 labels, the maximum is 110 valid cards per model and 330 cards
overall. Invalid JSON, schema violations, and quotes not found verbatim in the
source chunk are rejected and recorded.

## Requirements

Activate the virtual environment:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
& C:\Users\micha\OneDrive\Desktop\Università\Progetti\Unito-CREA-RAG\venv\Scripts\Activate.ps1
```

Install the generation-specific dependencies:

```powershell
python -m pip install -r requirements-generation-hf.txt
```

Llama 3.1 is gated. Before running it:

1. accept the Meta Llama 3.1 license on its Hugging Face model page;
2. authenticate locally with a Hugging Face token that has access.

```powershell
hf auth login
```

## Configuration

Model IDs, backends, 4-bit NF4 quantization and memory limits are defined in:

```text
configs/generation_models_hf.yaml
```

The default GPU limit is 5 GiB for the detected 6 GiB RTX 4050 Laptop GPU.
Layers that do not fit may be offloaded to CPU RAM. Generation will therefore
be substantially slower than on a larger GPU.

## Run

After adding or changing PDFs, rebuild both the embeddings and the
EDOS-specific retrieval output before generating cards:

```powershell
python scripts/02_parse_sources.py
python scripts/03_encode_chunks.py
python scripts/08_retrieve_edos_chunks.py
```

Run all three models:

```powershell
python scripts/03_generate_candidate_cards_hf.py
```

Run one model:

```powershell
python scripts/03_generate_candidate_cards_hf.py --models qwen35_9b
```

Small smoke run:

```powershell
python scripts/03_generate_candidate_cards_hf.py `
  --models qwen35_9b `
  --labels "1.1 threats of harm" `
  --cards-per-label 1 `
  --output-dir data/cards/candidates/huggingface_smoke
```

Resume after interruption:

```powershell
python scripts/03_generate_candidate_cards_hf.py --resume
```

Resume is accepted only when the SHA-256 fingerprint of the retrieved chunks
matches the run being resumed. This prevents cards from different PDF sets
from being mixed. Legacy outputs remain under `huggingface`; new PDF-based
outputs use `huggingface_pdf` by default.

## Outputs

For every model the script writes:

```text
data/cards/candidates/huggingface_pdf/<model>_candidate_cards.jsonl
data/cards/candidates/huggingface_pdf/<model>_attempts.jsonl
```

The candidate-card file contains only schema-valid cards. The attempt log
records accepted, rejected, skipped and failed chunks. A cumulative
`generation_summary.json` reports the result for each model.
