# Revisione dei chunk per parole — 4 novembre 2026

Valutazione: il corpus contiene passaggi utili e generalmente compatti, ma richiede una revisione della pulizia e della soglia minima prima di essere considerato definitivo per la nuova annotazione.

Sono stati effettuati controlli strutturali sull'intero corpus e una lettura qualitativa di un campione deterministico distribuito sui 18 documenti, integrato con casi segnalati. Non è una valutazione semantica esaustiva né una verifica della fedeltà rispetto all'impaginazione dei PDF. Le segnalazioni automatiche sono candidati da esaminare, non percentuali di errore.

| Proprietà | Valore |
|---|---:|
| Documenti | 18 |
| Chunk | 3.805 |
| Parole nel corpo: minimo / mediana / P95 / massimo | 100 / 140 / 262 / 1.199 |
| Chunk con al massimo 250 parole | 3.572 (93,88%) |
| Chunk oltre 500 parole | 17 |
| Chunk contenenti tabelle HTML | 424 |
| Chunk con tag table iniziali/finali sbilanciati | 0 |
| Sezioni/blocchi esclusi sotto 100 parole | 639 |
| Di cui da 80 a 99 parole | 93 |

## Cosa funziona

I chunk seguono la struttura e conservano molti passaggi comprensibili e pertinenti al dominio. Tutti hanno un percorso di sezione. Le tabelle conservano una struttura HTML e non risultano spezzate in base al bilanciamento dei tag table; questo controllo non dimostra la correttezza delle celle rispetto al PDF.

Esempi nel file `chunks.jsonl` (numero di riga):

- **1148, en_doc_12.md, 141 parole**: “How is cyber stalking gendered?” conserva un paragrafo tematicamente coerente con riferimenti agli studi. È un buon esempio di unità utilizzabile come contesto.
- **1953, en_doc_31.md, 225 parole**: “For images” aggrega consigli ed esempi sugli stereotipi visivi in un blocco coerente. Il valore per il task dipende dalla distinzione tra guida prescrittiva ed evidenza fattuale richiesta agli annotatori.
- **997, en_doc_11.md, 200 parole**: tabella con tipi di violenza digitale, definizioni e prevalenza. La struttura è conservata, ma le note a/b/c e il contesto delle statistiche vanno mantenuti rintracciabili: non ho validato tali dati rispetto alle fonti originali.

## Problemi prioritari

### 1. Contenuto editoriale e bibliografico ancora indicizzabile

225 chunk si trovano sotto intestazioni bibliografiche/note riconosciute con corrispondenza stretta (5,91% del corpus). Non sono automaticamente tutti inutili: alcune note contengono informazioni. Le letture mostrano però elenchi di citazioni che non costituiscono di per sé evidenze autonome.

Sono presenti anche 8 chunk sotto intestazioni Contents e 13 sotto Acknowledgements. Non suggerisco di eliminare indiscriminatamente prefazioni, sommari o tutte le note.

- **Riga 2373, en_doc_4.md**: sei voci bibliografiche sono aggregate come un chunk di 106 parole sotto “Existing victim support measures”. L'intestazione tematica non rende il contenuto un'evidenza sostanziale.
- **Riga 1705, en_doc_29.md**: una tabella dell'indice rimane indicizzabile; contiene molti termini tematici ma rimanda a pagine, senza sviluppare i contenuti.
- **Riga 1, en_doc_1.md**: testo di licenza, citazione suggerita e disclaimer aggregati in 176 parole.
- **Riga 976, en_doc_10.md**: il chunk massimo, di 1.199 parole, contiene un elenco di organizzazioni e siti WAVE. Non è rappresentativo dei normali paragrafi narrativi.

### 2. La soglia minima elimina anche contenuti utili

Le 639 esclusioni `below_minimum_words` contengono complessivamente 28.561 parole, ma includono sia contenuto sostanziale sia titoli, elenchi e metadati. Non è corretto classificare tutte queste esclusioni come perdita di informazione utile.

Due controesempi concreti nel file `editorial_exclusions.jsonl`:

- **Riga 1382, en_doc_28.md, 83 parole**: “Avoiding stereotypes: Avoid gendered stereotypes as descriptive terms”. Spiega come l'uso del femminile come sinonimo di debole o inefficace costituisca linguaggio sessista. È breve ma autonomo e pertinente.
- **Riga 2327, en_doc_31.md, 74 parole**: “Definition of key terms”. Definisce sexist, gender-biased, gender-neutral e gender-inclusive language. Anche questo blocco è escluso soltanto per la dimensione.

L'evidenza suggerisce di usare 100 parole come obiettivo di aggregazione anziché come filtro assoluto per sezioni brevi autonome. In alternativa, confrontare una soglia inferiore e documentare le differenze, evitando aggregazioni tra sezioni non correlate soltanto per raggiungere un minimo.

### 3. Residui PDF e confini poco autonomi

- **Riga 2787, en_doc_5.md**: il chunk inizia con “21”, il nome del dipartimento editore e una continuazione di frase (“violence rather than…”). L'aggregazione per paragrafi non risolve automaticamente una segmentazione difettosa del parsing.
- **Riga 1695, en_doc_28.md**: rimane `NO_CONTENT_HERE` in coda.
- Quattro chunk contengono marcatori espliciti di parsing residui; quattro contengono archi di grafi Mermaid. Questi numeri non rappresentano tutti i possibili residui di impaginazione.
- **Riga 17, en_doc_1.md**: contiene `R --> Q` e altre relazioni del diagramma, seguite da etichette. Non è un normale paragrafo narrativo.
- Note, URL e footer sono in alcuni casi mescolati al corpo. Le note dovrebbero essere associate al passaggio come riferimenti, separando quelle puramente bibliografiche senza cancellare dati o spiegazioni sostanziali.

## Raccomandazione

Conservare il criterio strutturale e il conteggio indipendente dal tokenizer. Prima del ranking/annotazione definitivi, dare priorità a: (1) pulizia dei residui e riconoscimento di indici/bibliografie; (2) conservazione dei blocchi brevi autonomi; (3) ricomposizione delle frasi interrotte e gestione delle note. La lunghezza massima è una questione secondaria nel corpus osservato: 17 chunk superano 500 parole e alcuni sono elenchi di scarso valore per il task.

La qualità della segmentazione non dimostra la pertinenza alle categorie EDOS: quella richiede la valutazione dei passaggi recuperati. Questa revisione non ha modificato i chunk, il codice o le esclusioni. Gli hash, le segnalazioni e il campione sono salvati in `quality_review.json`.
