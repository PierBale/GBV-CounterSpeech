# Nuova annotazione EDOS: run edos-v2-words-clean-20261104

La revisione corrente aggiunge filtri editoriali conservativi (versione 3.0).
Usa `config.edos-v2-words-clean.json`, run `edos-v2-words-clean-20261104` e output
`data/card_creation_comparison_edos_v2_words_clean_20261104/`. Riutilizza i 18
Markdown già parsificati. Il minimo di 100 parole e i confini delle sottosezioni
restano invariati. I vecchi chunk e la loro revisione restano nel precedente output.

I filtri escludono ringraziamenti e directory con intestazioni esplicite,
riconoscono indici testuali/HTML e identificano citazioni bibliografiche nelle
note o nelle sezioni dedicate. Introduzioni sotto un titolo `Contents`, note
esplicative e blocchi misti ambigui vengono conservati. Le decisioni sono
registrate nell'audit, comprese le rimozioni parziali delle tabelle dell'indice.

Nel job precedente, dalla directory `src`, usare:

```bash
python -u -m unito_amazon.card_creation experiment \
  --config unito_amazon/card_creation/config.edos-v2-words-clean.json \
  --no-generate-evidence --run-id edos-v2-words-clean-20261104
```

Lo script `scripts/run_edos_v2.slurm` usa questa configurazione e genera anche
QSF e mappa sotto il nuovo output. Per la successiva analisi, usare le nuove
cartelle `survey/` e `runs/edos-v2-words-clean-20261104/`.

## Versione precedente per parole

Il confronto attuale usa **minimo 100 parole nel corpo**, con
`config.edos-v2-words.json`, output `data/card_creation_comparison_edos_v2_words_20261104/`
e run `edos-v2-words-20261104`. I 18 documenti sono già parsificati nel corpus
`data/card_creation_comparison_edos_v2_20261104/markdown/` e vengono riutilizzati.
Il conteggio esclude titoli, tag HTML e punteggiatura; parole con apostrofi o
trattini interni contano una volta. Paragrafi e sezioni restano le unità di base,
senza suddivisione dei paragrafi lunghi e senza overlap. Le sezioni sotto soglia
sono registrate nell'audit. Non viene caricato alcun tokenizer per il chunking.

Da `src`, nel job GPU precedente, sostituire il comando finale con:

```bash
python -u -m unito_amazon.card_creation experiment \
  --config unito_amazon/card_creation/config.edos-v2-words.json \
  --no-generate-evidence --run-id edos-v2-words-20261104
```

Oppure inviare dalla radice `sbatch scripts/run_edos_v2.slurm`: lo script usa ora
la configurazione per parole e genera anche il questionario nel nuovo output.
Sul server trasferire il codice aggiornato e il corpus parsificato. La precedente
configurazione `config.edos-v2.json` rimane disponibile per il confronto a token;
i suoi output, quelli di `confronto-02` e il parsing non vengono sovrascritti.

Per la futura analisi, usare `data/card_creation_comparison_edos_v2_words_20261104/survey/`
per risultati, QSF, mappa e output, e il relativo `runs/edos-v2-words-20261104/`
per i ranking.

## Preparazione precedente e comandi per il run a token

Le 11 definizioni in `src/unito_amazon/card_creation/edos_labels.json` sono ora quelle
fornite per la nuova annotazione, con gli apostrofi del testo HTML normalizzati.
Il titolo della categoria 2.3 usa `and`, come corretto nel file principale.
Ogni definizione è parte della query: tutte le 11 query hanno quindi nuovi ID.

Il nuovo esperimento è isolato in `data/card_creation_comparison_edos_v2_20261104/`.
La preparazione locale è già completata: 18 PDF congelati, 13 Markdown copiati
senza modifiche, cinque PDF ancora da parsificare (`en_doc_28`–`en_doc_32`).
Parsing remoto, embedding e ranking non sono stati eseguiti durante la preparazione.

## File conservati e file nuovi

`data/card_creation_comparison/`, `src/1_parsed_results/` e i risultati precedenti
in `Qualtrics/` rimangono nelle posizioni originali. Il manifest di preparazione
contiene gli SHA-256 dei 127 file precedenti, compresi PDF e metadati attuali.
Il wrapper aggiunto `rebuild_retrieval.py` permette facoltativamente di controllare
questi hash. Per eseguire la pipeline sono sufficienti i comandi esistenti sotto.

Nel nuovo output:

- `inputs/`: copie dei 18 PDF, definizioni nuove e precedenti, manifest di
  `confronto-02`, configurazione completa, query, migrazione degli ID,
  template Qualtrics e copia del file `metadata.xlsx` attuale;
- `markdown/`: i 13 parsing precedenti e, dopo `parse`, i cinque nuovi;
- `work/` e `checkpoints/`: nuovi indici e checkpoint, separati da quelli vecchi;
- `runs/edos-v2-20261104/`: risultati e manifest delle quattro combinazioni;
- `survey/`: nuovo QSF, nuova mappa delle domande e future risposte/analisi.

Gli originali Markdown senza sidecar di parsing sono riutilizzati come nel run
precedente: la copia è verificata byte per byte, ma la loro provenienza originaria
non può essere ricostruita. `metadata.xlsx` è conservato per provenienza; non viene
passato come `source_metadata_path`, che richiede JSON e nel confronto precedente
era assente.

## Comandi sul cluster

Portare sul cluster il checkout aggiornato **includendo la nuova cartella dati**,
`documents/`, `src/1_parsed_results/`, `data/card_creation_comparison/` e `Qualtrics/`.
Gli hash sono relativi al repository: la cartella può cambiare percorso tra
Windows e Linux, ma i file elencati nel manifest devono conservare gli stessi byte.
La cartella preparata contiene anche copie non ancora aggiunte a Git. Trasferire
i dati direttamente dalla cartella locale, per esempio con un archivio o una
copia che preservi i byte: un semplice `git pull` non trasferisce i file non
versionati e può cambiare i fine riga dei file di testo tra Windows e Linux.

Dalla radice del repository, attivare l'ambiente usato per il precedente confronto:

```bash
source ~/mambaforge/etc/profile.d/conda.sh
conda activate llm_new_env
```

Se mancano dipendenze, installarle nell'ambiente del cluster:

```bash
python -m pip install -r requirements-card-creation.txt
```

La fase `parse` usa LlamaParse EU, tier `agentic`, versione `2026-09-13`.
La chiave deve essere disponibile sul cluster come `LLAMA_CLOUD_API_KEY_MOLIVERI`
nell'ambiente o nel `.env` alla radice. Le credenziali non sono incluse negli snapshot.
Eseguire il parsing da un nodo autorizzato all'accesso di rete; non serve una GPU.

```bash
cd src
python -m unito_amazon.card_creation parse \
  --config unito_amazon/card_creation/config.edos-v2.json --no-all-combinations
cd ..
sbatch scripts/run_edos_v2.slurm
```

Il job usa la partizione `epito` e una GPU, come lo script cluster esistente,
ed eredita l'ambiente Python attivato. La GPU deve poter ospitare i modelli 8B
con batch 4; il codice mantiene la configurazione del confronto precedente.
Per scegliere la partizione o la prenotazione disponibili sul proprio cluster:

```bash
sbatch --partition=gracehopper scripts/run_edos_v2.slurm
# Se si dispone della prenotazione mike:
sbatch --reservation=mike scripts/run_edos_v2.slurm
```

`--no-all-combinations` disabilita la matrice solo per il comando `parse`, che
non esegue modelli. Il job usa poi il comando esistente `experiment` con le quattro
combinazioni e genera il questionario con `Qualtrics/build_retrieval_survey.py`.

In alternativa, da un nodo GPU già allocato, il comando esistente `all` esegue
parsing, chunking, embedding, retrieval e reranking in sequenza:

```bash
cd src
python -m unito_amazon.card_creation all \
  --config unito_amazon/card_creation/config.edos-v2.json
cd ..
```

Per il solo confronto dopo il parsing, sostituire `all` con `experiment`.
Il comando `all` non genera il questionario; quello viene creato dal comando
`Qualtrics/build_retrieval_survey.py` riportato nello script Slurm.

Il rank ricrea i chunk dell'intero corpus con il tokenizer comune
`Qwen/Qwen3-Embedding-8B`, filtro editoriale e minimo di 100 token. Ricostruisce
due indici (`Qwen/Qwen3-Embedding-8B`, `BAAI/bge-m3`) sul corpus ampliato e applica
entrambi i reranker (`BAAI/bge-reranker-v2-m3`, `Qwen/Qwen3-Reranker-8B`).
Mantiene 100 candidati per query e i primi 10 dopo il reranking. Non genera card.
Gli embedding dei vecchi chunk sono ricalcolati nel nuovo esperimento: la pipeline
attuale non riutilizza vettori tra indici di corpus diversi. I vecchi indici restano intatti.

## Ripresa e nuova annotazione

In caso di interruzione, ripetere lo stesso comando o inviare di nuovo lo stesso
job. Il parsing salta i Markdown già presenti e i checkpoint del nuovo output
consentono di riprendere retrieval/reranking. Non usare `--overwrite` o
`--force-reindex` sugli output precedenti. Se cambiano corpus, definizioni o
configurazione dopo l'avvio del confronto, creare un altro output/run. Il nuovo
`output_dir` separa anche indici e cache di lavoro: cambiare soltanto `run_id`
archivierebbe i risultati separatamente, ma riutilizzerebbe le cache precedenti.

Il JSON del nuovo run legge le etichette da `inputs/edos_labels.json`, una copia
aggiornata del file principale. Per correggere ancora le etichette **prima di
avviare il run**, aggiornare anche questa copia; le modifiche al file principale
non si propagano automaticamente allo snapshot.

Solo dopo il completamento dei ranking vengono creati:

```text
data/card_creation_comparison_edos_v2_20261104/survey/Best_retrieval_config.qsf
data/card_creation_comparison_edos_v2_20261104/survey/Best_retrieval_config_question_map.csv
```

Il campione contiene i primi tre chunk per ciascuna query e configurazione.
Un medesimo `(query_id, chunk_id)` appare una volta sola; `source_matches`
conserva tutti i modelli e i rispettivi rank. Il QSF mostra le definizioni nuove
e le scelte umane 0, 1, 2. Importarlo come **nuovo questionario** in Qualtrics.
Conservare le risposte precedenti con il vecchio QSF e la vecchia mappa.

Esportare le nuove risposte come `survey/Best-retrieval-results.csv`. Per analizzarle
senza sovrascrivere l'analisi precedente:

```bash
python -m pip install -r Qualtrics/requirements-analysis.txt
NEW=data/card_creation_comparison_edos_v2_20261104
python Qualtrics/analyze_retrieval.py \
  --results "$NEW/survey/Best-retrieval-results.csv" \
  --mapping "$NEW/survey/Best_retrieval_config_question_map.csv" \
  --survey "$NEW/survey/Best_retrieval_config.qsf" \
  --run-dir "$NEW/runs/edos-v2-20261104" \
  --out-dir "$NEW/survey/analysis"
```

Questa analisi mantiene gli score umani 0–2, `krippendorff.alpha`, l'NDCG classico
di scikit-learn e i grafici SVG. Le vecchie annotazioni non sono trasferite
automaticamente alle nuove query.
