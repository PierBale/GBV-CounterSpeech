# Creazione delle evidence card

Pipeline autonoma implementata in `src/unito_amazon/card_creation`.
Non dipende dal prototipo `Cards/Generation`.

1. PDF -> Markdown con **LlamaParse**, tramite SDK LlamaCloud.
2. Chunking per paragrafi con minimo effettivo predefinito di **100 token** nel corpo e nell'input completo.
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

## Esperimenti di retrieval e reranking

Il comando `experiment` orchestra gli stessi stadi di chunking, indicizzazione,
retrieval, reranking e generazione descritti sopra. Non introduce valutazione
umana, annotazioni o metriche. I comandi precedenti e i default 4B restano
disponibili.

L'analisi del codice ha individuato `retrieval.py` come coordinatore delle tre
fasi dense/reranking/selezione, `vector_store.py` come backend SentenceTransformers
e FAISS e `generation.py` come generatore delle carte. Il reranker BGE riutilizza
`SequenceClassificationReranker` del package `unito_amazon.retrieval`, aggiungendo
scoring senza troncamento. Le label sono la mappa label/definizione in
`edos_labels.json`; `schema.py` descrive chunk e passaggi e `card_schema.py`
descrive le carte correnti. I vecchi schemi delle carte restano disponibili.
Le dipendenze esistenti (Pydantic, NumPy, SentenceTransformers, Transformers,
PyTorch, FAISS) sono sufficienti; non serve FlagEmbedding.

Dalla directory `src`, per le quattro combinazioni A–D senza LLM:

```bash
python -m unito_amazon.card_creation experiment --config unito_amazon/card_creation/config.experiment.example.json --run-id confronto-01
```

La configurazione di esempio usa `all_combinations: true`, `retrieval_top_k: 100`,
`rerank_top_k: 10`, `generate_evidence: false` e budget input di 8192 token per
entrambi gli stadi. I limiti effettivi rispettano anche il contesto del modello;
non si tronca il testo per adattarlo a un limite. Un superamento causa errore.
`experiment` parte dai Markdown e crea/riutilizza i chunk: non chiama LlamaParse.
`all --all-combinations` esegue anche il parsing prima dell'esperimento.
`retrieve --all-combinations` equivale a `experiment --all-combinations`.
Questi comandi seguono `generate_evidence`, quindi aggiungere
`--no-generate-evidence` quando non si vuole caricare il generatore.

Per una singola combinazione, tutti i parametri principali sono anche CLI:

```bash
python -m unito_amazon.card_creation experiment --embedding-model BAAI/bge-m3 --reranking-model Qwen/Qwen3-Reranker-8B --retrieval-top-k 100 --rerank-top-k 10 --no-generate-evidence --run-id singola-01
```

`--all-combinations` abilita la matrice fissa:

| Pipeline | Embedding | Reranker |
| --- | --- | --- |
| A | Qwen/Qwen3-Embedding-8B | BAAI/bge-reranker-v2-m3 |
| B | Qwen/Qwen3-Embedding-8B | Qwen/Qwen3-Reranker-8B |
| C | BAAI/bge-m3 | BAAI/bge-reranker-v2-m3 |
| D | BAAI/bge-m3 | Qwen/Qwen3-Reranker-8B |

Il JSON usa `embedding_model`, `reranker_model`, `retrieval_top_k`,
`rerank_top_k`, `generate_evidence`, `generator_model`, `all_combinations` e
`run_id`. `dense_top_j` resta un alias compatibile di `retrieval_top_k`:
specificare un solo nome. `--no-all-combinations` permette di disabilitare
la matrice quando è abilitata nel JSON. `--evidence-model` e
`--generator-model` sono alias CLI di `generator_model`.

Il chunking legacy usa il tokenizer dell'embedding quando
`chunk_tokenizer_model` non è specificato. In `experiment`, il tokenizer viene
fissato **prima** di cambiare embedding: si usa `chunk_tokenizer_model` oppure
il modello embedding della configurazione iniziale. La configurazione di
esempio lo fissa esplicitamente a Qwen3-Embedding-8B. Questo tokenizer è solo
il contatore per aggregare i paragrafi, non viene usato per fare encoding con
BGE. Corpus, ordine dei chunk, ID e testo rimangono identici tra le pipeline.
La modalità esperimento disabilita la deduplicazione nella selezione per
l'evidence: prende i primi K del reranker, conservando anche testi uguali con
ID distinti. Il flusso legacy conserva la sua deduplicazione configurabile.
Gli archivi di reranking contengono sempre i primi K prima della deduplicazione.

### Query riproducibili e più query per label

Senza `queries_path`, ogni label genera la query storica
`Sexism type: <label>\nDescription: <definition>`. Il suo `query_id` deriva da
SHA-256 di label e testo, non dalla sola label. Non c'è campionamento casuale.

Per più query della stessa label, impostare `queries_path` a un JSONL:

```jsonl
{"query_id":"q-001","edos_label":"1.1 threats of harm","query_text":"Evidence on threats against women","metadata":{"group":"first"}}
{"query_id":"q-002","edos_label":"1.1 threats of harm","query_text":"Consequences of gendered intimidation"}
```

Le label devono esistere in `labels_path`; la definizione viene presa da quel
file. Gli ID devono essere distinti e le query non vuote. `query_id` omesso
viene calcolato da label e testo. Query identiche con la stessa label richiedono
ID espliciti distinti. L'ordine del file viene conservato e tutte le
combinazioni processano lo stesso elenco completo. Anche le carte e i tentativi
di generazione conservano `query_id`, `query_text` e `pipeline_id`.

### Archivi JSONL e cache

Ogni run è archiviato in `<output_dir>/runs/<run_id>/`. Senza un ID esplicito
viene generato un ID UTC con suffisso casuale. La configurazione di esempio
scrive in `data/card_creation_comparison/`. All'interno del run:

```text
manifest.json
chunking/editorial_exclusions.jsonl
retrieval/qwen3-embedding-8b.jsonl
retrieval/bge-m3.jsonl
reranking/qwen3-embedding-8b__bge-reranker-v2-m3.jsonl
reranking/qwen3-embedding-8b__qwen3-reranker-8b.jsonl
reranking/bge-m3__bge-reranker-v2-m3.jsonl
reranking/bge-m3__qwen3-reranker-8b.jsonl
reranking_full/<embedding>__<reranker>.jsonl
evidence/<embedding>__<reranker>.jsonl
evidence/<embedding>__<reranker>__attempts.jsonl
errors.jsonl
```

`evidence/` esiste solo se richiesta e completata; `errors.jsonl` esiste se
sono stati registrati errori. Ogni riga retrieval rappresenta una coppia
query/chunk e contiene `run_id`, `query_id`, `query_text`, `edos_label`,
`embedding_model`, `chunk_id`, `chunk_text`, `document_id`, `document_source`,
`retrieval_rank`, `retrieval_score`, `retrieval_top_k`, `metadata`.
`model_input_text` conserva il testo del passaggio effettivamente fornito a
embedding e reranker, inclusi i titoli quando abilitati. `text`, `chunk_text`,
`model_input_text` e `chunk.text` contengono lo stesso passaggio completo;
`body_text` conserva il corpo separato dalle intestazioni.
Conserva anche il record nested originario: `chunk`, `label`, `definition`,
`dense_rank`, `dense_score` e metadati della query. In `metadata` restano fonte,
percorso delle sezioni, ID dei paragrafi e conteggio token. Non vengono inventati
pagine o offset assenti nel corpus.

Le righe reranking preservano tutti questi campi e aggiungono
`reranking_model`, `rerank_rank`, `rerank_score`, `rerank_top_k`, `pipeline_id`,
oltre ai nomi legacy `reranker_*`. `reranking/` conserva i primi K per query;
`reranking_full/` conserva tutti i J candidati con entrambi gli ordinamenti.
Se il corpus ha meno di J o K chunk, si salvano i chunk effettivamente
disponibili; i valori richiesti restano nei record. A parità di score reranker
si conserva l'ordine del retrieval. Nessun record viene fuso tra query o pipeline.

Il retrieval è scritto prima di caricare il reranker; il reranking prima
di caricare il generatore. L'esperimento completa retrieval e reranking di
tutte le combinazioni prima di iniziare la generazione opzionale: un errore
del LLM lascia disponibili tutti questi risultati. Il manifest contiene configurazioni effettive,
query complete, hash del corpus, checksum e conteggi dei file, istruzioni,
versioni delle librerie e hash del codice degli stadi. Gli stati sono
`running`, `complete`, `failed`, `interrupted`. Per un confronto completo
controllare `status: complete` e le combinazioni registrate nel manifest.
Un errore salva fase, modelli, pipeline, tipo/messaggio e, quando disponibile,
query ID, testo e label. Il comando termina con errore senza dichiarare completo
il run. I checkpoint di lavoro rimangono incompleti e riprendibili.

I file di lavoro sono separati in `work/retrieval/<embedding>/` e
`work/reranking/<embedding>__<reranker>/`, con checkpoint indipendenti.
L'indice e il retrieval sono eseguiti una volta per embedding; i due reranker
ricevono esattamente lo stesso pool ordinato. Una seconda esecuzione compatibile
riusa anche i ranking e le carte completate senza caricare modelli. Gli stessi
risultati di lavoro possono alimentare un nuovo archivio con diverso `run_id`.
Cambiare input o impostazioni sperimentali mantenendo lo stesso `run_id`
genera un errore: usare un nuovo ID. Per riprendere un run interrotto,
rilanciare lo stesso comando con lo stesso ID. Non eseguire job concorrenti
sullo stesso `output_dir` e non modificare corpus o query durante un run.

### Encoding, score ed evidence opzionale

Le implementazioni seguono le model card ufficiali:
[Qwen embedding](https://huggingface.co/Qwen/Qwen3-Embedding-8B),
[BGE-M3](https://huggingface.co/BAAI/bge-m3),
[Qwen reranker](https://huggingface.co/Qwen/Qwen3-Reranker-8B),
[BGE reranker](https://huggingface.co/BAAI/bge-reranker-v2-m3).
Qwen embedding usa l'istruzione di retrieval solo sulle query, nel formato
`Instruct: ...\nQuery: ...`; BGE-M3 usa query e documenti senza prefissi,
soltanto dense retrieval. SentenceTransformers fornisce pooling e tokenizer
del modello; entrambi gli embedding sono normalizzati a norma unitaria per
il coseno FAISS. Non vengono attivati sparse retrieval o ColBERT.
Qwen reranker conserva il protocollo instruction/query/document e lo score
ufficiale `P(yes)` tra i token `no`/`yes`; BGE restituisce il logit grezzo
del classificatore, senza sigmoid. Non si applicano normalizzazioni ulteriori
agli score e non sono direttamente confrontabili tra modelli diversi.

Con `generate_evidence: false` oppure `--no-generate-evidence` non viene
costruito né invocato alcun generatore LLM. Anche `all` salta `generate` e
`index-cards`. Per generare evidence nell'esperimento, aggiungere
`--generate-evidence --evidence-model <modello>`. Il generatore conserva il
backend Gemma 4 per il default e supporta anche modelli chat testuali caricabili
con `AutoModelForCausalLM` e `AutoTokenizer`. Il contratto rimane
`reasoning`/`evidence`, con i metadati aggiunti dal codice. L'indicizzazione
delle carte non fa parte del comando `experiment`.

Test locali senza scaricare pesi o richiedere GPU:

```bash
python -m unittest discover -s unito_amazon/card_creation/tests
```

I test della matrice usano modelli simulati ma gli stadi e i checkpoint reali.
Non sostituiscono una prova di inferenza con i pesi 8B sulla macchina di esecuzione.

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
| Candidati dense per categoria | `data/card_creation/edos_passages_dense.jsonl` |
| Ranking completo prima del filtro | `data/card_creation/edos_passages_ranked.jsonl` |
| Decisioni di deduplicazione | `data/card_creation/deduplication_decisions.jsonl` |
| Passaggi selezionati per la generazione | `data/card_creation/edos_passages.jsonl` |
| Card | `data/card_creation/cards.jsonl` |
| Tentativi/scarti | `data/card_creation/generation_attempts.jsonl` |
| Indice card | `data/card_creation/indexes/cards.faiss` |
| Checkpoint e configurazione effettiva | `data/card_creation/checkpoints/` |

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
Se il PDF o i parametri cambiano rispetto al manifest, il Markdown viene
rigenerato automaticamente. I Markdown legacy senza manifest vengono conservati.

`generate` riprende automaticamente: salta card completate e passaggi rifiutati,
mentre ritenta gli output non validi. Registra ogni tentativo con risposta originale
e motivo dell'errore/scarto. Se cambiano input, prompt o modello, le card e i log
obsoleti vengono invalidati automaticamente. `generate --overwrite` sostituisce card e log
della generazione precedente; su `all` rigenera anche i Markdown.

`index --force-reindex` e `index-cards --force-reindex` forzano la ricostruzione.
Altrimenti riutilizzano l'indice compatibile e ricostruiscono quello obsoleto.
Se si cambiano i chunk, eseguire nuovamente `index` prima di `retrieve`.

## Checkpoint e ripresa automatica

Si può rilanciare lo stesso comando `all`: gli stadi completati con input e
configurazione compatibili vengono saltati **prima di caricare tokenizer o
modelli**. Il parsing mantiene il controllo per PDF e riutilizza i Markdown
esistenti, senza chiamate al servizio quando non servono nuove conversioni.

La cache verifica SHA-256 dei file di input e output, configurazione effettiva
e codice dei componenti. I manifest in `checkpoints/` distinguono `running`
da `complete`: un fallimento non viene registrato come lavoro completato.
Non sono i manifest `.parse.json` di LlamaParse, che restano separati.

Il comando pubblico `retrieve` comprende tre checkpoint indipendenti:

1. Dense retrieval: salva `edos_passages_dense.jsonl` dopo ogni categoria.
2. Reranking: salva il ranking completo in `edos_passages_ranked.jsonl` dopo
   ogni categoria. Una categoria interrotta viene ricalcolata; quelle completate
   vengono riutilizzate.
3. Deduplicazione: riutilizza il ranking completo e i vettori FAISS salvati.

La generazione conserva ogni carta e ogni tentativo con sostituzioni atomiche
dei file JSONL. Alla ripresa verifica i record e completa soltanto i passaggi
mancanti o con output non valido, senza duplicare le card già salvate.

L'invalidazione segue le dipendenze, come concordato:

| Modifica | Da ricostruire, oltre agli stadi successivi |
| --- | --- |
| Markdown, metadati bibliografici, minimo token o tokenizer | Chunk e indice paragrafi |
| Modello embedding | Chunk/tokenizer, indici e retrieval |
| Categorie/descrizioni o J | Dense retrieval |
| Modello o istruzione del reranker | Reranking |
| K, attivazione o soglia di deduplicazione | Deduplicazione |
| Modello Gemma, prompt, temperatura o budget di output | Generazione delle card |

Quindi cambiare la soglia del coseno non ricarica Qwen e cambiare Gemma non
rifà chunking, embedding o reranking. Aggiornare una libreria per correggere un
errore di caricamento non invalida da solo gli artefatti già completati.
Batch size, device, numero di retry e soglia degli avvisi sui chunk sono opzioni
operative: modificarle non elimina risultati già completati e validi.
Le modifiche al codice possono invalidare conservativamente più fasi dello
stesso componente.

Gli output invalidati vengono eliminati e ricreati, **solo tra i file registrati
della pipeline dentro `output_dir`**, senza cancellazioni ricorsive. PDF e
Markdown di input non vengono eliminati; i Markdown con manifest obsoleto
possono essere aggiornati dallo stadio di parsing. Le correzioni manuali
delle card sono conservate se schema, input e configurazione restano compatibili.
Per conservare risultati di esperimenti con configurazioni differenti, usare
un `output_dir` distinto per ciascun esperimento.

Alla prima esecuzione di questa versione, gli artefatti senza checkpoint
compatibile richiedono una verifica o ricostruzione iniziale: non è possibile
dedurre in modo affidabile tutti i vecchi parametri dalla sola presenza dei
file. Gli indici FAISS e le card legacy possono essere riutilizzati quando
i loro manifest precedenti superano le verifiche esistenti. Le esecuzioni
successive usano i nuovi checkpoint.

Con `all`, le dipendenze vengono ricostruite nell'ordine corretto. Un singolo
stadio non usa prerequisiti con checkpoint incompleto o diventato obsoleto:
in quel caso richiede di rilanciare `all` o il comando che produce i prerequisiti.
Non eseguire due job contemporaneamente nello stesso `output_dir` e non
modificare input/configurazione durante un job.

I test sintetici della cache e della ripresa non caricano modelli e sono in
`tests/test_cache.py`. Dalla cartella `src`:

```bash
python -m unittest discover -s unito_amazon/card_creation/tests
```

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
configurazione, oppure direttamente `all`. Se cambiano i passaggi selezionati,
le card e gli indici successivi vengono invalidati e rigenerati automaticamente.

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

Corpo e input completo sono contati con `chunk_tokenizer_model` (oppure il tokenizer
dell'embedding nel flusso legacy), senza token speciali. Le righe separate da un semplice a capo rimangono nello stesso
paragrafo; righe vuote e titoli delimitano i paragrafi. I titoli diventano metadati
di sezione. Viene rimossa solo la formattazione Markdown di presentazione.

### Esclusioni editoriali verificabili

`editorial_filter_enabled: true` è il default, esplicitato anche nei due JSON
di esempio. Il filtro opera sui paragrafi prima dell'aggregazione e usa solo
regole sintattiche versionate in `editorial.py` (`EDITORIAL_RULES_VERSION`).
Non consulta query, label, embedding, reranker o LLM. La versione iniziale:

- Esclude paragrafi in sezioni il cui titolo completo è `IMPRINT`,
  `DISCLAIMER AND COPYRIGHT` o la variante `DISCLAIMERANDCOPYRIGHT`, incluse
  le loro sottosezioni secondo la gerarchia Markdown.
- Sotto un titolo esatto `Contents` o `Table of contents`, esclude un paragrafo
  soltanto se ogni riga non vuota ha forma di voce seguita da numero di pagina
  arabo o romano; sono ammessi puntini e il formato `/ p.1`.
- Esclude paragrafi composti soltanto da righe `logo: ...`, un ISBN isolato,
  un avviso `© ... <anno>` su una sola riga o un credito esplicito
  `Cover photo:` / `Cover design:` / `Cover design and layout:`.
- Esclude paragrafi che iniziano con specifiche formule editoriali:
  `All rights reserved.`, `Some rights reserved.`,
  `Under the terms of this licence,` (anche `license`),
  `Cataloguing-in-Publication (CIP) data.`, `Sales, rights and licensing.`,
  `Third-party materials.` o `General disclaimers.`.

La versione `2.0` aggiunge una pulizia del layout **prima** di interpretare
le intestazioni Markdown. Rimuove righe isolate `icon: ...` e `logo: ...`, anche
quando si trovano tra righe di un paragrafo sostanziale. Riconosce marcatori
espliciti come `Page 18 ► ...` o `... ► Page 19`. Per i running title generici
richiede almeno due numeri di pagina distinti associati allo stesso titolo;
il titolo deve comparire nelle intestazioni del documento (anche su più righe,
fino a tre frammenti) o essere tutto maiuscolo. Sono gestiti titolo seguito
dal numero, numero seguito da `|`/`►` e titolo accanto a un numero su riga separata.
I frammenti di titolo su righe non Markdown devono avere forma di titolo,
secondo la regola esplicita in `layout.py`. I casi non riconosciuti restano.

La versione `2.1` riconosce anche numero e titolo separati solo da spazi,
come `12 THE STATE OF THE WORLD’S GIRLS 2020`: il numero iniziale è il
candidato numero di pagina e l'anno finale rimane parte del titolo. Restano
necessari due numeri di pagina distinti osservati per lo stesso titolo.
Una dicitura esplicita `logo:` o `publisher:` nel sorgente può confermare
il nome dell'ente in piè di pagina ripetuti, come
`50 European Institute for Gender Equality`.
I blocchi completi `<page_header>...</page_header>` e
`<page_footer>...</page_footer>` vengono rimossi come layout esplicitamente
marcato dal parser. Un titolo marcato `page_header` può identificare anche
copie senza tag, soltanto per uguaglianza normalizzata esatta e almeno due
occorrenze; le vere intestazioni Markdown restano conservate. Un blocco senza
tag di chiusura non viene usato per eliminare il resto del documento.

I titoli correnti riconosciuti vengono rimossi anche quando sono scritti come
heading Markdown, così non modificano la gerarchia reale delle sezioni. Se un
marcatore di pagina separa due frammenti del corpo, si rimuove anche il padding
di righe vuote/separatori circostante: i frammenti vengono riuniti prima del
chunking. Una vera intestazione Markdown continua a delimitare la sezione.
Non si riscrivono parole, non si completano frasi e non si ricostruiscono
automaticamente parole spezzate con trattino.

Non vengono eliminate le prime N pagine, né paragrafi sulla sola presenza di
parole chiave come `copyright`. Bibliografie, prefazioni, ringraziamenti,
tabelle e copertine senza marcatori sufficienti vengono conservati. Gli indici
con layout ambiguo, righe spezzate o HTML possono restare nel corpus.
Le regex sono euristiche trasparenti: non garantiscono classificazione semantica
perfetta e non costituiscono un miglioramento di retrieval già validato.

Ogni rimozione/esclusione viene salvata in `<output_dir>/editorial_exclusions.jsonl`
con file Markdown, SHA-256 del file, ID della fonte, regola e versione.
Le rimozioni del layout registrano numero di riga e testo originale; i titoli
correnti registrano anche firma e numeri di pagina osservati. Le esclusioni
editoriali registrano testo del paragrafo dopo la pulizia Markdown preesistente,
ID del paragrafo, sezione e pattern. I residui sotto soglia registrano corpo,
input completo, ID dei paragrafi e conteggi dei token. I Markdown di input
rimangono disponibili e non vengono modificati.
L'audit viene anche copiato in `runs/<run_id>/chunking/editorial_exclusions.jsonl`
e registrato nel manifest del run con checksum. Disabilitare il filtro mantiene
il layout e i paragrafi editoriali; la soglia minima resta comunque applicata
e i residui sotto soglia restano registrati nell'audit.

Gli ID dei paragrafi vengono assegnati dopo la pulizia del layout e prima delle
esclusioni editoriali, senza rinumerarli dopo queste ultime. I paragrafi della
stessa sezione possono essere aggregati attraverso materiale editoriale rimosso;
una rimozione non crea una sezione artificiale. I chunk superstiti
continuano a usare l'hash di fonte, ID dei paragrafi e corpo; una diversa
aggregazione può legittimamente produrre nuovi chunk ID. Tutte le combinazioni
dell'esperimento condividono lo stesso corpus filtrato.

### Titoli nel testo fornito ai modelli

Con `include_section_titles: true` (default), embedding, entrambi i reranker
e prompt dell'evidence usano un unico formatter deterministico:

```text
Section: Titolo originale > Sottotitolo originale

Corpo del chunk
```

Si usano esclusivamente le intestazioni presenti in `section_path`, rimuovendo
la loro formattazione di presentazione (`**`, backtick e tag HTML semplici).
Non si generano titoli, riassunti o informazioni mancanti. Se non ci sono titoli,
si usa soltanto il corpo. Il chunk viene materializzato con `text` uguale al
passaggio completo: encoder, reranker e prompt dell'evidence leggono quel testo,
senza anteporre nuovamente i titoli. `body_text` conserva il corpo e i titoli
originali restano nei metadati. Nei JSONL archiviati anche `text`, `chunk_text` e
`model_input_text` sono uguali all'input del passaggio. `token_count` conta
il testo completo e `body_token_count` conta il corpo. I backend controllano separatamente la lunghezza
dell'input completo, compresi titoli, query e prompt propri del modello.
`include_section_titles: false` ricostruisce i chunk con input costituito dal
solo corpo; le vecchie librerie con testo incompatibile richiedono il rechunking.

Entrambe le opzioni sono incluse nelle firme di cache degli stadi interessati.
I precedenti indici/ranking incompatibili vengono ricostruiti e gli archivi
dei run precedenti rimangono separati. Dopo questa modifica usare un nuovo
`run_id`, ad esempio `confronto-02`, per non mescolare corpus o rappresentazioni
diversi nello stesso esperimento. I test sintattici e i casi negativi sono in
`tests/test_editorial_context.py`; l'aggiornamento della matrice verifica anche
il testo con i titoli e la presenza dell'audit. Questi test sono stati aggiunti
senza eseguirli sul PC locale, come richiesto.

I paragrafi brevi consecutivi vengono uniti in ordine fino a raggiungere almeno
100 token con la configurazione predefinita. Se il successivo paragrafo è già lungo, il gruppo breve viene unito
a quel paragrafo intero. Un residuo breve a fine sezione viene aggiunto al chunk
precedente della stessa sezione. Se l'intera sezione resta sotto il minimo configurato (100 token per default),
viene esclusa dal corpus di retrieval con regola `below_minimum_tokens`,
conservandone il testo nell'audit. Non si uniscono sezioni diverse soltanto per
raggiungere la soglia e non si usano i titoli per rendere eleggibile un corpo
troppo breve. Ogni chunk pubblicato soddisfa il minimo sia nel corpo sia
nell'input completo, secondo il tokenizer comune di chunking. La versione
precedente conservava i residui sotto soglia: questo spiega i chunk di tre token
anche con `min_chunk_tokens: 200`. I nuovi test sono in `tests/test_chunk_layout.py`.

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

Il prompt è in `prompts.py`: richiede una spiegazione della pertinenza alla label
e una sintesi qualitativa autonoma, non un counter-argument già formulato.
Il modello restituisce soltanto l'oggetto intermedio
`{"reasoning": "...", "evidence": "..."}`, senza envelope `card`, citazioni
o metadati. Il codice aggiunge automaticamente i metadati e salva il passaggio
sorgente ripulito in `chunk.body_text` e il passaggio completo di titoli in
`chunk.text`. Lo stesso oggetto `chunk`
contiene `chunk_id`, `section_path` (sezione/sottosezione/... in ordine),
`paragraph_ids`, `token_count` e `body_token_count`. I paragrafi aggregati appartengono allo stesso
percorso di intestazioni Markdown; non si aggregano sezioni diverse. I separatori
tra paragrafi restano nel corpo, salvo le interruzioni di layout riconosciute
e riunite meccanicamente.

Ogni riga di `cards.jsonl` è un oggetto JSON validato tramite lo schema Pydantic
in `card_schema.py` (versione `3.0`). Contiene `reasoning`, `evidence`, fonte, chunk, label,
descrizione, ranking dense/reranker, modello e identificativi stabili.
Il campo `prompt` conserva `system` e `user` completi, inclusi label, definizione
e passaggio sorgente. In caso di retry, `user` contiene il prompt corretto
effettivamente usato per ottenere la carta. Il prompt è salvato per tracciabilità
e non entra nel testo indicizzato o nel contesto del card-RAG.
Il sampling è attivo con `generation_temperature: 0.7`. Il valore `0` disattiva
il sampling (greedy). Sono consentiti `generation_retries: 5` retry di correzione
degli output JSON non validi: al massimo 6 tentativi per carta, incluso il primo.
I due campi devono essere stringhe non vuote; campi extra
non sono ammessi. Il nuovo contratto non prevede una risposta di rifiuto.

La validazione automatica garantisce struttura e provenienza del testo sorgente,
**non** la fedeltà semantica della sintesi né la pertinenza alla label.
Non vengono salvati `status`, `validation` o `retrieval_keywords` nelle carte.
La chiave `generation_key` resta solo nel log tecnico dei tentativi; la ripresa
associa i lavori alle carte tramite `card_id`. Non sono garantite esattamente
10 card per label: si selezionano
al massimo 10 passaggi dopo la deduplicazione; gli output non validi vengono
registrati e ritentati. Il cambio di prompt/schema rigenera le carte e il loro
indice, mantenendo chunk, embedding, retrieval, reranking e deduplicazione.
Gli schemi delle vecchie carte `1.0` rimangono in `schema.py` per riferimento:
non vengono più usati dal generatore o dal card-RAG. Non viene effettuata una
conversione automatica da `argument` a `evidence`, perché hanno scopi diversi.

## Counter-speech con retrieval sulle card

In `src/config.py`, aggiungere `"cards"` a `RAG_RETRIEVERS` e poi utilizzare
il normale `src/main.py`. Ad esempio, per eseguire soltanto il card-RAG:

```python
RUN_WITHOUT_RAG = False
RAG_RETRIEVERS = ("cards",)
```

Per il confronto controllato tra zero-shot, chunk-RAG 4B e card-RAG 4B, usare
`RUN_WITHOUT_RAG = True` e `RAG_RETRIEVERS = ("qwen3-emb-4b", "cards")`.
Questa è ora la configurazione predefinita; vedere `src/README_EXPERIMENT.md`
per i subset da 100 messaggi di EDOS e MultiCONAN e i confronti pairwise.
I risultati delle card hanno strategia
`rag_cards`; ogni evidence item include i metadati della card e la sua fonte.

Il retrieval confronta l'hate speech soltanto con **tipo di sessismo e sintesi
`evidence`**, usando Qwen3-Embedding-4B e FAISS. Il testo è composto dai campi
`Sexism type` ed `Evidence`; `reasoning` resta nella carta, ma non viene
indicizzato né inserito nel contesto del generatore. I metadati della fonte
restano nella carta, ma non nel testo indicizzato né nel contesto del generatore. Il passaggio
completo resta nella libreria per verifica ma non viene indicizzato né inserito
nel contesto del counter-speech. Non viene usata la label gold del messaggio.

Il prompt di counter-speech rimane `RAGCounterSpeechPrompt`, generico rispetto
al tipo di evidence; recupera le migliori tre card, senza diversificazione.
Servono almeno tre card utilizzabili, controllate prima di caricare il decoder.

`CARD_LIBRARY_PATH`, `CARD_INDEX_PATH`, `CARD_EMBEDDING_MODEL` e
`CARD_EMBEDDING_BATCH_SIZE` sono configurabili in `src/config.py`. Se si cambia
`output_dir` nella creazione, aggiornare questi percorsi. Non è più previsto
un filtro sullo stato di approvazione delle carte.

La pipeline genera le card e il loro indice; l'esecuzione del counter-speech
rimane un passaggio esplicito attraverso `main.py`.
