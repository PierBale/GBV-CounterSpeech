# Creazione delle evidence card

Pipeline autonoma implementata in `src/unito_amazon/card_creation`.
Non dipende dal prototipo `Cards/Generation`.

1. PDF -> Markdown con **LlamaParse**, tramite SDK LlamaCloud.
2. Chunking per paragrafi con minimo desiderato di **100 token**.
3. Embedding **Qwen/Qwen3-Embedding-4B** e indice FAISS esatto.
4. Query composta da **label EDOS + descrizione**, top **J=100** passaggi.
5. Reranking **Qwen/Qwen3-Reranker-4B** di tutti i J candidati.
6. Rimozione di duplicati esatti e quasi-duplicati tramite similarità coseno,
   poi selezione di al massimo **K=10** passaggi per categoria.
7. Generazione di una card per passaggio con **google/gemma-4-12B-it**.
8. Creazione di un indice FAISS separato sulle card per il counter-speech RAG.

J, K e la soglia di deduplicazione sono configurabili. Non vengono applicati MMR
o quote di diversificazione. La deduplicazione opera separatamente per categoria:
un passaggio può essere selezionato per più label. Le label e le descrizioni iniziali sono in `edos_labels.json`;
le descrizioni sono quelle già utilizzate nel progetto e possono essere riviste.

## Installazione e configurazione

Eseguire i seguenti comandi sulla macchina destinata all'inferenza.
Python 3.10 o successivo; GPU e memoria adeguate ai modelli 4B/12B.
L'embedding model viene rilasciato prima del reranker; il reranker viene
rilasciato prima di Gemma, che viene a sua volta rilasciato prima dell'indice card.

Dalla root del repository:

```bash
pip install -r requirements-card-creation.txt
```

Gemma richiede una versione di Transformers che includa
`AutoModelForMultimodalLM` e l'architettura `gemma4_unified`. Usare una release
aggiornata compatibile con il [model card ufficiale](https://huggingface.co/google/gemma-4-12B).
Il requisito generale `transformers>=5.0` del progetto, da solo, non garantisce
che un'installazione già esistente contenga questa architettura.

Inserire la chiave LlamaCloud nel file `.env` alla root (già ignorato da Git),
oppure esportarla nell'ambiente:

```dotenv
LLAMA_CLOUD_API_KEY=your_api_key
```

Per utilizzare un nome già presente nel proprio ambiente, ad esempio
`LLAMA_CLOUD_API_KEY_MOLIVERI`, impostare `llama_api_key_env` nel JSON di configurazione.
La chiave non viene salvata negli artefatti. Il parser usa l'endpoint EU, tier
`agentic` e versione `2026-09-13`, come il notebook di parsing esistente;
tutti e tre sono configurabili. Il servizio è
[LlamaParse tramite LlamaCloud](https://developers.llamaindex.ai/llamaparse/parse/getting_started/).

`config.example.json` contiene i valori predefiniti. Le configurazioni sono JSON
parziali: i campi omessi conservano i default. I percorsi contenuti nel JSON sono
relativi alla **root del repository**, indipendentemente dalla directory corrente.

Percorsi predefiniti:

| Contenuto | Percorso dalla root |
| --- | --- |
| PDF sorgente | `documents/` |
| Markdown LlamaParse | `src/1_parsed_results/` |
| Chunk | `data/card_creation/chunks.jsonl` |
| Indice paragrafi | `data/card_creation/indexes/chunks.faiss` |
| Ranking completo prima del filtro | `data/card_creation/edos_passages_ranked.jsonl` |
| Decisioni di deduplicazione | `data/card_creation/deduplication_decisions.jsonl` |
| Passaggi selezionati per la generazione | `data/card_creation/edos_passages.jsonl` |
| Card | `data/card_creation/cards.jsonl` |
| Tentativi/scarti | `data/card_creation/generation_attempts.jsonl` |
| Indice card | `data/card_creation/indexes/cards.faiss` |

Gli indici hanno un file `.faiss.json` associato con documenti, modello,
istruzione, fingerprint e checksum. Non separare i due file. Il controllo dei
fingerprint impedisce di usare un indice con una libreria diversa.
`data/card_creation/` è ignorata da Git.

## Comandi

Dalla cartella `src`:

```bash
python -m unito_amazon.card_creation all
```

Oppure eseguire separatamente gli stadi:

```bash
python -m unito_amazon.card_creation parse
python -m unito_amazon.card_creation chunk
python -m unito_amazon.card_creation index
python -m unito_amazon.card_creation retrieve
python -m unito_amazon.card_creation generate
python -m unito_amazon.card_creation index-cards
```

Per passare una configurazione:

```bash
python -m unito_amazon.card_creation all --config unito_amazon/card_creation/config.example.json
```

`parse` conserva la struttura delle sottocartelle dei PDF. Riutilizza i Markdown
già presenti, compresi quelli prodotti dal notebook; per questi ultimi, se manca
il manifest, segnala che la provenienza non può essere verificata automaticamente.
I nuovi Markdown hanno un manifest `.parse.json` con checksum del PDF e parametri.
Se il PDF o i parametri cambiano, richiede `parse --overwrite`.

`generate` riprende automaticamente: salta card completate e passaggi rifiutati,
mentre ritenta gli output non validi. Registra ogni tentativo con risposta originale
e motivo dell'errore/scarto. Se cambiano input, prompt o modello, utilizzare un
nuovo `output_dir` o `generate --overwrite`. Questa opzione sostituisce card e log
della generazione precedente; su `all` rigenera anche i Markdown.

`index --force-reindex` e `index-cards --force-reindex` forzano la ricostruzione.
Altrimenti riutilizzano l'indice compatibile e ricostruiscono quello obsoleto.
Se si cambiano i chunk, eseguire nuovamente `index` prima di `retrieve`.

## Deduplicazione dopo il reranking

Il riferimento scientifico è **Abbas, A., Tirumala, K., Simig, D., Ganguli, S.,
e Morcos, A. S. (2023), [SemDeDup: Data-efficient learning at web-scale through
semantic deduplication](https://arxiv.org/abs/2303.09540)**, in particolare la
[Sezione 3](https://arxiv.org/html/2303.09540v2#S3). Il paper individua duplicati
semantici confrontando gli embedding con una soglia di similarità coseno.

Questa implementazione è un **adattamento**, non una replica di SemDeDup:
non usa il clustering k-means del paper, perché lavora su piccoli pool di
candidati, e sceglie il rappresentante in base al ranking del reranker, non
alla distanza dal centroide. Non è stato validato un miglioramento della
qualità delle card su questo corpus.

Per ciascun tipo di sessismo, `retrieve`:

1. Ordina tutti i J candidati per punteggio del reranker (a parità di punteggio,
   conserva l'ordine del dense retrieval).
2. Scorre la lista e confronta ogni candidato con i passaggi già selezionati.
   Un hash SHA-256 del testo normalizzato identifica i duplicati esatti:
   si uniformano composizione Unicode NFC, spazi e a capo, senza rimuovere
   numeri, negazioni o cambiare maiuscole/minuscole.
3. Per i quasi-duplicati riutilizza gli embedding dei documenti già presenti
   nell'indice FAISS, senza nuove inferenze. Scarta il candidato se la massima
   similarità coseno con un passaggio selezionato è **maggiore o uguale** alla
   soglia. Sopravvive quindi il passaggio meglio classificato dal reranker.
4. Riempie fino a K posti, considerando anche i candidati più in basso nella
   lista. Un passaggio scartato non può causare ulteriori rimozioni a catena.
   Se restano meno di K passaggi, li conserva senza abbassare la soglia.

Il confronto semantico usa soltanto il coseno, senza un secondo filtro lessicale
o un LLM aggiuntivo. Non vengono cancellati chunk dal corpus né dall'indice.
Lo stesso chunk può essere selezionato per categorie diverse, poiché la
generazione dipende anche dal tipo di sessismo e dalla sua descrizione.

Configurazione (già attiva per default):

```json
{
  "deduplication_enabled": true,
  "deduplication_similarity_threshold": 0.97
}
```

`0.97` è un valore iniziale da calibrare su coppie del proprio corpus, **non**
una soglia proposta o validata da SemDeDup per Qwen. Una soglia più alta rende
il filtro meno aggressivo. Anche il paper calibra la soglia per dataset
([Sezione 6.5](https://arxiv.org/html/2303.09540v2#S6.SS5)). La similarità elevata
non garantisce identità fattuale: passaggi con numeri, anni o popolazioni diversi
possono essere vicini nello spazio degli embedding. Controllare gli scarti
prima di usare il filtro per gli esperimenti definitivi.

`deduplication_decisions.jsonl` registra una decisione per ciascun candidato:
`selected`, `exact_duplicate`, `near_duplicate` oppure `top_k_limit`.
Include categoria, fonti, rank originale, punteggio reranker, soglia, similarità
e riferimento al passaggio mantenuto per ogni duplicato. `top_k_limit` indica
un candidato non selezionato per il limite K, **non** un duplicato. Per i record
non duplicati, `compared_to` e `cosine_similarity` indicano il passaggio
selezionato più simile al momento del confronto, se presente.

Il ranking completo resta in `edos_passages_ranked.jsonl`. Solo la selezione
finale in `edos_passages.jsonl` viene letta da `generate`. `reranker_rank`
mantiene la posizione nel ranking originale, quindi può avere salti.
Impostando `deduplication_enabled: false` si ripristina la selezione dei primi K
senza alcuna deduplicazione, utile come baseline.

Non cambia il comando: rieseguire `retrieve` e poi `generate` con la stessa
configurazione. Se cambiano i passaggi selezionati e sono già presenti card,
la protezione della generazione richiederà un nuovo `output_dir` oppure
`generate --overwrite` (che sostituisce card e log precedenti).

I test sintetici sono in `tests/test_deduplication.py` e non richiedono modelli
o GPU. Possono essere eseguiti dalla cartella `src` con
`python -m unittest unito_amazon.card_creation.tests.test_deduplication`.

## Avanzamento in console

Ogni stadio stampa messaggi con orario, nome dello stadio, avvio, completamento
e durata. In caso di errore o interruzione viene indicato lo stadio coinvolto;
l'errore originale non viene nascosto. Le stampe usano `flush=True` per essere
visibili subito anche quando l'output viene rediretto.

Durante l'elaborazione vengono mostrati:

- PDF convertiti o Markdown riutilizzati, con contatore dei file;
- paragrafi rilevati e chunk prodotti per ciascun Markdown;
- caricamento e rilascio dei modelli, encoding degli embedding e stato degli indici;
- categorie elaborate, candidati recuperati e batch di reranking completati;
- duplicati scartati, passaggio mantenuto, similarità coseno e conteggi per categoria;
- passaggi gia elaborati alla ripresa, tentativi di generazione, carte salvate,
  scarti e output non validi;
- conteggi finali e percorsi degli artefatti salvati.

Le risposte complete del generatore restano nel log JSONL dei tentativi, non
vengono stampate in console. I messaggi non cambiano prompt, ranking o card.

## Regole di chunking

I token sono contati con il tokenizer Qwen dell'embedding model, senza token
speciali. Le righe separate da un semplice a capo rimangono nello stesso
paragrafo; righe vuote e titoli delimitano i paragrafi. I titoli diventano metadati
di sezione. Viene rimossa solo la formattazione Markdown di presentazione.

I paragrafi brevi consecutivi vengono uniti in ordine fino a raggiungere almeno
100 token. Se il successivo paragrafo è già lungo, il gruppo breve viene unito
a quel paragrafo intero. Un residuo breve a fine sezione viene aggiunto al chunk
precedente della stessa sezione. Se l'intera sezione resta sotto 100 token,
viene conservata: la soglia non implica la perdita di testo.

Nessun paragrafo viene spezzato, nessun testo viene duplicato per overlap e non
si aggregano sezioni o documenti differenti. Non esiste una lunghezza massima
artificiale: `long_chunk_warning_tokens` genera solo un avviso. Se un input supera
il contesto ammesso dall'embedding model, dal reranker o da Gemma, lo stadio termina
con un errore esplicito anziché troncare il testo.

Si possono aggiungere metadati bibliografici attraverso `source_metadata_path`:
un file JSON che associa percorsi Markdown relativi a campi `title`, `publisher`,
`year`, `url` e/o `pdf_file`. Senza metadati il titolo deriva dal nome del file;
anno, editore, URL e pagina non vengono inventati.

## Prompt e schema delle card

Il prompt è in `prompts.py`: mantiene la struttura di ragionamento conciso e
argomento del generatore già presente nel progetto, aggiungendo una citazione
testuale verificabile. Il modello restituisce solo `source_quote`, `reasoning`
e `argument` dentro un oggetto `card`, oppure `card: null` con un motivo se il
passaggio non supporta una card utile. I metadati vengono aggiunti dal codice.

Ogni riga di `cards.jsonl` è un oggetto JSON validato tramite lo schema Pydantic
in `schema.py`. Contiene fonte, sezione, passaggio completo, citazione, label,
descrizione, ranking dense/reranker, modello e identificativi stabili.
La citazione deve essere una sottostringa esatta del passaggio. Sono consentiti
due tentativi aggiuntivi di correzione degli output non validi, configurabili.

La validazione automatica garantisce struttura e corrispondenza della citazione,
**non** l'implicazione semantica dell'argomento né la pertinenza alla label.
Le card sono quindi `status: "candidate"`; la revisione può promuoverle a
`"validated"`. Non sono garantite esattamente 10 card per label: si selezionano
al massimo 10 passaggi dopo la deduplicazione, e il generatore può rifiutare
quelli privi di evidenza utile.

## Counter-speech con retrieval sulle card

In `src/config.py`, aggiungere `"cards"` a `RAG_RETRIEVERS` e poi utilizzare
il normale `src/main.py`. Ad esempio, per eseguire soltanto il card-RAG:

```python
RUN_WITHOUT_RAG = False
RAG_RETRIEVERS = ("cards",)
```

Per il confronto tra zero-shot, chunk-RAG e card-RAG, mantenere gli altri
retriever e `RUN_WITHOUT_RAG = True`. I risultati delle card hanno strategia
`rag_cards`; ogni evidence item include i metadati della card e la sua fonte.

Il retrieval confronta l'hate speech con **argomento, citazione, reasoning,
label e keyword della card**, usando Qwen3-Embedding-4B e FAISS. Il passaggio
completo resta nella libreria per verifica ma non viene indicizzato né inserito
nel contesto del counter-speech. Non viene usata la label gold del messaggio.

Il prompt di counter-speech rimane `RAGCounterSpeechPrompt`, generico rispetto
al tipo di evidence; recupera le migliori tre card, senza diversificazione.
Servono almeno tre card utilizzabili, controllate prima di caricare il decoder.

`CARD_LIBRARY_PATH`, `CARD_INDEX_PATH`, `CARD_EMBEDDING_MODEL` e
`CARD_EMBEDDING_BATCH_SIZE` sono configurabili in `src/config.py`. Se si cambia
`output_dir` nella creazione, aggiornare questi percorsi. Per usare soltanto card
approvate, impostare `CARD_VALIDATED_ONLY = True`: il filtro ammette solo
`status: "validated"` e ricostruisce l'indice se la selezione cambia.

La pipeline genera le card e il loro indice; l'esecuzione del counter-speech
rimane un passaggio esplicito attraverso `main.py`.
