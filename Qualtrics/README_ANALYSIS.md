# Analisi delle annotazioni di retrieval

Aprire `retrieval_analysis.ipynb` per consultare risultati, grafici e spiegazioni.
Il notebook contiene già gli output e può essere rieseguito dalla root del progetto
o dalla cartella `Qualtrics`.

Per rigenerare l'analisi senza Jupyter:

```powershell
python -m pip install -r Qualtrics/requirements-analysis.txt
python Qualtrics/analyze_retrieval.py
python -m unittest discover -s Qualtrics/tests -v
```

Gli input predefiniti sono `Best-retrieval-results.csv`, il relativo question map,
il QSF della survey e i ranking del run `confronto-02`. Il QSF serve a verificare
che i codici esportati 1/2/3 corrispondano ai giudizi visualizzati 0/1/2.
I percorsi possono essere cambiati con `--results`, `--mapping`, `--survey`,
`--run-dir` e `--out-dir` (eseguire `--help` per i dettagli).

I risultati sono salvati in `analysis/`:

- `report.md`: risultati, metodo, limiti e riferimenti alla letteratura.
- `model_ranking.csv`: ranking macro sulle query e intervalli al 95%.
- `agreement.csv` e `agreement_pairwise.csv`: accordo globale, per etichetta e per coppia.
  La metrica principale è alpha di Krippendorff ordinale in tutti e tre i casi,
  con IC 95% bootstrap degli item. Per coppia si usano solo gli item valutati da
  entrambi e si ricampionano insieme i due voti; il kappa quadratico resta una diagnostica.
- `metrics_by_label.csv`, `label_patterns.csv`: prestazioni e pattern per etichetta.
- `ordering_common_pool.csv`: correlazione tra score e giudizi sugli stessi item.
- `paired_comparisons.csv`: confronti appaiati e correzione di Holm per metrica.
- `aggregation_sensitivity.csv`, `annotator_model_ranking.csv`: stabilità rispetto
  a media, mediana, soft label e ai singoli annotatori. La prima tabella include
  nDCG lineare/esponenziale, rank, IC 95% e correlazioni di ordinamento.
- `aggregation_metrics_by_label.csv`: tutte le tre aggregazioni per query e pipeline.
- `aggregation_common_query_ranking.csv`: confronto nDCG sulle stesse query valide
  per tutte le aggregazioni. La mediana rende non definito nDCG per la 3.3:
  questo confronto usa 10 query, gli score umani medi ne usano sempre 11.
- `aggregation_paired_comparisons.csv`: test tra pipeline per ciascuna aggregazione,
  con Holm per aggregazione/metrica e anche sull'intera famiglia dei confronti.
- `annotations_long.csv`, `item_consensus.csv`, `pipeline_selections.csv`:
  dati analitici con attribuzione dei passaggi condivisi a ogni configurazione.
- `annotators.csv`, `analysis_audit.json`: filtri sulle risposte e hash degli input.
- Cinque grafici SVG vettoriali: ranking, pattern per etichetta, accordo per etichetta,
  confronto tra aggregazioni e pattern per etichetta delle tre aggregazioni.

Gli output usano pseudonimi degli annotatori; non riportano indirizzi IP, nomi,
email o ID originali delle risposte. Nessun input viene modificato.

La metrica principale è nDCG@3 con gain lineare, normalizzato sullo stesso pool
annotato per query; la rilevanza media mostra la qualità senza considerare il rank.
Kendall tau-b serve a studiare l'ordine degli score e gestisce i pari merito.
Il confronto su item comuni usa l'intersezione dei candidati annotati presenti
in tutti i file `reranking_full`: non considera come irrilevanti gli item mancanti.
Le correlazioni su terne proprie sono esportate separatamente e risentono
della differente selezione dei candidati.

Le soft label conservano `p_relevance_0/1/2` in `item_consensus.csv`. Con gain
lineare, `E[Y] = p1 + 2*p2` coincide con la media aritmetica. Con gain esponenziale
si confronta `2^mean−1`, `2^median−1` ed `E[2^Y−1] = p1 + 3*p2` per le soft label.
Gli ultimi sono gain derivati, riscalati su 0–2; lo score umano delle soft label
è sempre il valore atteso dei voti su 0–2. Il confronto soft esponenziale è
nDCG dei gain attesi, non la media degli nDCG dei tre annotatori.
L'agreement di Krippendorff usa sempre i voti originali e non cambia tra aggregazioni.

Con una sola query per etichetta e tre annotatori, i pattern per etichetta sono
descrittivi. Gli intervalli condizionano sugli annotatori osservati e il ranking
non dimostra la superiorità generalizzabile di un modello.
