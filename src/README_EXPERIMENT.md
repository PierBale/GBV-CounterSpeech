# Confronto zero-shot, chunk-RAG e card-RAG

La configurazione corrente in `src/config.py` abilita soltanto:

- `without_rag`: stesso decoder, senza evidence;
- `rag_qwen3_emb_4b`: retrieval sui chunk con Qwen3-Embedding-4B;
- `rag_cards`: retrieval sulle carte con Qwen3-Embedding-4B.

BM25 rimane disponibile ma disattivato; la baseline 0.6B non è abilitata.
Decoder e parametri di generazione del counter-speech restano gli stessi fra
le strategie. I parametri Gemma della **creazione delle carte** sono separati.

## Dataset e sottoinsiemi

`DATASET = "both"`, `LIMIT = 100`, `DATASET_SAMPLE_SEED = 42`.
Il limite vale **per dataset**, non sul totale concatenato:

- EDOS: messaggi sessisti del test set (`EDOS_SPLIT = "test"`);
- MultiCONAN: messaggi con target `WOMEN`, senza inventare uno split ufficiale.

Prima del campionamento vengono eliminati messaggi identici, ignorando soltanto
gli spazi iniziali/finali. In particolare, più reference MultiCONAN per lo stesso
messaggio non diventano più item di test. La selezione è casuale con seed fisso,
non stratificata per label EDOS; viene conservata in
`data/generated/subset_both.json`. Se un dataset ha meno di 100 messaggi unici
eleggibili, vengono usati quelli disponibili e il numero effettivo viene stampato.
Tutti i decoder e tutti i sistemi usano gli stessi item selezionati.

## Corpus condiviso e retrieval

Le baseline sui chunk leggono `CHUNK_LIBRARY_PATH`, per default
`data/card_creation/chunks.jsonl`: sono gli stessi chunk, senza nuovo chunking,
da cui sono state generate le carte. Il programma controlla che ogni carta
rimandi a un chunk della libreria con testo, sezioni, paragrafi e fonte identici.
Carte o chunk non compatibili causano un errore prima di caricare il decoder.

Chunk-RAG 4B e card-RAG usano lo stesso modello, la stessa istruzione di query,
FAISS con similarità coseno e `top_k=3`. Nessun routing tramite label gold,
reranking o diversificazione durante il retrieval dei messaggi. Le due strategie
condividono in memoria il modello embedding, ma hanno indici distinti:

- chunk: `data/indexes/chunk_rag_qwen3_4b.faiss`;
- carte: `data/card_creation/indexes/cards.faiss`.

Il primo indice non sovrascrive quello del retrieval per categoria usato nella
creazione delle carte. I record RAG riportano checksum del corpus, embedding
model, istruzione, tipo di unità e assenza di reranking nei `metadata`.

Il corpus sorgente è condiviso, ma le collezioni non hanno necessariamente la
stessa cardinalità: il chunk-RAG cerca su tutti i chunk, il card-RAG sulle carte
derivate dai passaggi selezionati. Il confronto misura quindi il sistema di
selezione/sintesi delle carte nel suo insieme, non la sola parafrasi a parità
di passaggi candidati. Restano diversi anche lunghezza e formato delle evidence.

## Lunghezza del contesto

Ogni risposta contiene `context_length`, con:

- numero di evidence (`3` per i RAG, `0` per zero-shot);
- caratteri e parole del contesto formattato effettivamente inserito nel prompt;
- token calcolati con il tokenizer testuale del decoder, senza special token;
- identificativo del tokenizer e ambito della misura.

Sono inclusi i prefissi `Evidence 1`, ecc. e l'escaping del testo, ma non i tag
esterni `<evidence>`, il messaggio di odio o il system prompt. Non si impone un
budget comune né si tronca il contesto per pareggiare le lunghezze. Se un backend
non espone il tokenizer, i token del contesto non vuoto sono `null`, non una stima.
Zero-shot ha lunghezze pari a zero.

## Esecuzione

Preparare prima `chunks.jsonl` e `cards.jsonl` tramite la pipeline delle carte.
Non occorre riparsificare i PDF per questo esperimento. Dalla cartella `src`:

```bash
python main.py
python judge.py
```

`main.py` salva, per esempio,
`data/generated/llama-3.1-8b_both_counterspeech.jsonl`.
**Attenzione:** la generazione dei counter-speech sostituisce il file di output
dello stesso decoder/dataset; non aggiunge una ripresa automatica di `main.py`.
Il judge invece riutilizza i giudizi già presenti per coppie e testi identici.

## Confronti pairwise

Il judge costruisce tutte le coppie entro lo stesso messaggio, dataset/split e
decoder, senza confrontare con il gold MultiCONAN:

1. Zero-shot vs chunk-RAG 4B;
2. Zero-shot vs card-RAG 4B;
3. Chunk-RAG 4B vs card-RAG 4B, marcato `primary_comparison: true`.

Con 100 item per dataset: 200 messaggi, 600 counter-speech e **600 confronti per
decoder e judge**. Riabilitando BM25, diventano quattro sistemi e sei coppie per
messaggio. Output duplicati o gruppi incompleti vengono segnalati, non ignorati.

I riepiloghi `.summary.json` sotto `data/judged` riportano vittorie, pareggi,
sconfitte e tassi di vittoria, sia aggregati sia separati per dataset e coppia.
Le lunghezze del contesto di entrambi i sistemi vengono conservate nei record
dei confronti, ma non mostrate al judge.

Il judge usa i marcatori ufficiali di JudgeLM (`[Question]`, le due risposte
Assistant, `[System]`, `### Response:`), con i criteri factuality, number of facts,
relevance e specificity in quest'ordine. Il modello restituisce il formato
nativo previsto da JudgeLM: esattamente due punteggi sulla prima riga, separati
da uno spazio, e una spiegazione nelle righe successive. Non gli viene richiesto
un JSON e non occorrono librerie aggiuntive per vincolare il formato.

Il parser legge soltanto i due punteggi della prima riga, accettando anche
decimali, spazi multipli e la virgola come separatore, come `parse_score` nel
codice ufficiale. Entrambi devono essere finiti e compresi fra 1 e 10. Il
vincitore viene calcolato dal codice: vince il punteggio più alto; è `tie` solo
se i due punteggi validi sono identici (soglia zero). La parola "tie" nella
spiegazione non determina l'esito. Una riga con tre o più punteggi è invalida:
non si prendono arbitrariamente i primi due valori. A differenza del fallback
`[-1, -1]` nel codice ufficiale, un errore di parsing non diventa un pareggio.

Punteggi, `winner` (`response_1`, `response_2` o `tie`) e spiegazione (`reasoning`)
vengono comunque salvati come campi JSON in `pairwise_judge`, insieme al testo
originale in `raw_response`. Il primo tentativo usa decoding deterministico;
se il parsing fallisce, si eseguono al massimo `JUDGE_MAX_RETRIES = 5` nuovi
tentativi con lo stesso prompt, sampling attivo e `JUDGE_RETRY_TEMPERATURE = 0.7`.
Se ancora invalido, l'esito resta `null` con `parse_error` ed è escluso dai validi.

Gli output nuovi hanno il suffisso `.native-scores-v1.jsonl`: i giudizi precedenti
vengono conservati, senza essere riutilizzati da questo protocollo. Non occorre
rigenerare i counter-speech. Per rieseguire solo il judge, da `src`:

```bash
python judge.py
```

Questa implementazione non aggiunge
inversione dell'ordine A/B, verifiche fattuali sulle fonti o valutazione umana:
non erano parte delle modifiche richieste. Il giudizio pairwise non certifica
automaticamente la factuality delle risposte.

## Statistiche per dataset e tabelle LaTeX

Dalla cartella `src`, dopo il judge, calcolare le metriche automatiche (se non
sono ancora disponibili) e poi esportare le statistiche:

```bash
python metrics.py
python stats.py --judge-paths ../data/judged/llama-3.1-8b_both_counterspeech.pairwise-judged.judgelm-7b.native-scores-v1.jsonl
```

Oltre ai report JSON/CSV/Markdown, viene creato automaticamente
`data/stats/judgelm_pairwise.tex` (oppure nella directory di `--output-dir`).
Il file contiene una sola tabella `table*` per ogni coppia decoder/judge, con
sezioni EDOS e MultiCONAN separate da `\midrule`, senza un totale aggregato.
EDOS e MultiCONAN non vengono sommati. Anche console e Markdown hanno sezioni
separate; il JSON contiene `by_dataset.edos` e `by_dataset.multiconan`, mentre
i CSV hanno la colonna `source` (o `Dataset` nelle tabelle di confronto).
Le righe corrispondono alle coppie osservate, incluso chunk-RAG vs card-RAG:
non vengono aggiunte baseline gold o sistemi disattivati.

Le colonne sono Baseline, System, Valid / Total, System Wins, Baseline Wins e
Ties. Ogni esito riporta conteggio e percentuale con due decimali sui **validi**;
se non ci sono giudizi validi, la percentuale è `n/a`. Il grassetto evidenzia
la percentuale di vittorie più alta tra sistemi con la stessa baseline nello
stesso blocco, senza implicare significatività statistica.

Nel preambolo LaTeX servono `\usepackage{booktabs,graphicx}`; per includere il
file nel documento usare, adattando il percorso:

```latex
\input{judgelm_pairwise.tex}
\input{automatic_metrics.tex}
```

`stats.py` aggrega giudizi già salvati e non avvia LLM o calcoli BERTScore.
Genera anche `data/stats/automatic_metrics.tex`: una tabella per decoder, solo
per MultiCONAN con reference disponibile. Le colonne sono BLEU-4, METEOR,
ROUGE-L, BERTScore-F1 e BERTScore-F1 rescaled. Distinct-1, Distinct-2 e repetition
rate restano nei dati salvati, ma non vengono mostrate nella tabella LaTeX.
Ogni cella riporta media, deviazione standard campionaria e numero di
valori validi. BLEU-4 è una media per risposta, non un punteggio corpus;
i valori non vengono moltiplicati per 100. I dati mancanti sono `n/a`, non zero.
Il CSV dettagliato conserva anche varianza e BERTScore precision/recall.

`metrics.py` esclude EDOS e i record MultiCONAN privi di reference prima del
calcolo e riutilizza le metriche già presenti. BERTScore carica un modello:
eseguire questo comando sul cluster. Distinct-n e repetition rate non richiedono
reference, ma in questo esperimento vengono riportati anch'essi solo sul
sottoinsieme MultiCONAN con reference. `stats.py` applica lo stesso filtro anche
ai vecchi file di metriche misti. Se non ci sono metriche MultiCONAN, il `.tex`
contiene un commento esplicativo e nessuna tabella con valori inventati.

`--no-save` stampa soltanto le statistiche e non scrive nemmeno il file LaTeX.
Se nella cartella sono presenti più esperimenti, selezionare quelli desiderati
con `--judge-paths` per non aggregarli involontariamente.
Analogamente, `--metric-paths` seleziona i file `.metrics.jsonl` desiderati.
I vecchi report non più prodotti (ad esempio il confronto col gold) non vengono
cancellati: per avere una cartella priva di report precedenti usare un nuovo
`--output-dir`.

## Test locali leggeri

I test sono scritti con dati fittizi e modelli mock, senza download di pesi:

```bash
python -m unittest discover -s unito_amazon/evaluation/tests -v
```

Non sono stati eseguiti durante l'implementazione, come richiesto.
