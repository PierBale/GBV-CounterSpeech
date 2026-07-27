# Candidate Card Annotation

Interfaccia locale per annotare le candidate card prodotte da:

- `llama31_8b_candidate_cards.jsonl`
- `ministral3_8b_candidate_cards.jsonl`
- `qwen35_9b_candidate_cards.jsonl`

## Installazione

Da PowerShell:

```powershell
cd "Cards\Generation"
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements-annotation.txt
```

## Avvio

Sempre dalla cartella `Cards\Generation`:

```powershell
.\.venv\Scripts\python -m streamlit run scripts\12_candidate_card_annotation_app.py
```

Streamlit apre automaticamente la pagina nel browser. Se non accade, aprire
l'indirizzo locale mostrato nel terminale, normalmente `http://localhost:8501`.

## Dati e salvataggio

Per impostazione predefinita l'app legge i tre JSONL dalla cartella:

```text
data/cards/candidates/huggingface/
```

Ogni click su **Salva** o **Salva e prossima** aggiorna direttamente:

```text
data/validation/candidate_card_annotations.xlsx
```

Il salvataggio usa `card_id` come chiave: una seconda valutazione della stessa
card aggiorna la riga esistente e non crea un duplicato. Il workbook contiene:

- `annotations`: punteggi, note, annotatore e tutti i metadati della card;
- `rubric`: descrizione sintetica dei punteggi da 1 a 3.

La barra laterale consente anche di scaricare una copia del workbook, filtrare
per modello, label e stato di annotazione, e modificare i percorsi locali.

Se il file `.xlsx` è aperto in Excel, chiuderlo prima di salvare una nuova
annotazione.
