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
  nDCG@3 classico lineare, rank, IC 95% e correlazioni di ordinamento.
- `aggregation_metrics_by_label.csv`: tutte le tre aggregazioni per query e pipeline.
- `aggregation_common_query_ranking.csv`: confronto nDCG sulle stesse query valide
  per tutte le aggregazioni. Seguendo scikit-learn, l'ideale nullo della 3.3 con
  mediana ha nDCG 0: tutte le 11 query entrano ora in tutte le aggregazioni.
- `aggregation_paired_comparisons.csv`: test tra pipeline per ciascuna aggregazione,
  con Holm per aggregazione/metrica e anche sull'intera famiglia dei confronti.
- `annotations_long.csv`, `item_consensus.csv`, `pipeline_selections.csv`:
  dati analitici con attribuzione dei passaggi condivisi a ogni configurazione.
- `annotators.csv`, `analysis_audit.json`: filtri sulle risposte e hash degli input.
- `ndcg_sklearn_verification.json`: verifica delle chiamate dirette a scikit-learn
  su tutte le query e conferma che i risultati lineari rimangono invariati.
- Cinque grafici SVG vettoriali: ranking, pattern per etichetta, accordo per etichetta,
  confronto tra aggregazioni e pattern per etichetta delle tre aggregazioni.

Gli output usano pseudonimi degli annotatori; non riportano indirizzi IP, nomi,
email o ID originali delle risposte. Nessun input viene modificato.

La metrica principale è nDCG@3 con gain lineare, normalizzato sullo stesso pool
annotato per query; la rilevanza media mostra la qualità senza considerare il rank.
Il calcolo usa `sklearn.metrics.ndcg_score`, `k=3`, `ignore_ties=False`; DCG usa
`sklearn.metrics.dcg_score`. Versione della libreria e impostazioni sono nell'audit.
Per query si mantengono tutti gli item annotati con i loro gain; i punteggi d'ordine
3, 2, 1 identificano i tre item selezionati nel rank originale e gli altri item
hanno punteggio d'ordine 0, sotto il cutoff. Il match è per `question_id`.
Scikit-learn assegna nDCG 0 quando l'ideale è tutto zero: tale query viene inclusa
nella media, al posto della precedente convenzione n.d./esclusione.
Riferimento: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ndcg_score.html
Kendall tau-b serve a studiare l'ordine degli score e gestisce i pari merito.
Il confronto su item comuni usa l'intersezione dei candidati annotati presenti
in tutti i file `reranking_full`: non considera come irrilevanti gli item mancanti.
Le correlazioni su terne proprie sono esportate separatamente e risentono
della differente selezione dei candidati.

Le soft label conservano `p_relevance_0/1/2` in `item_consensus.csv`.
`E[Y] = p1 + 2*p2` coincide con la media aritmetica: con nDCG lineare nativo
di scikit-learn, media e soft label producono gli stessi risultati.
Tutti gli score umani restano su 0–2 e non vengono trasformati.
L'agreement di Krippendorff usa sempre i voti originali e non cambia tra aggregazioni.

Per nDCG si valuta il ranking finale estratto: eventuali score grezzi uguali
sono risolti dai rank salvati. Kendall e Spearman usano invece gli score grezzi
con i loro pari merito. Alpha globale, per etichetta e per coppia usano
`import krippendorff` e `krippendorff.alpha`, con livello ordinale (principale)
o nominale (diagnostica). La matrice viene trasposta in annotatori × item;
i mancanti restano NaN. L'adattatore locale gestisce solo formato e campioni
degeneri, senza implementare la formula. Ogni replica bootstrap richiama la
libreria. La versione installata non fornisce gli IC, quindi il ricampionamento
degli item è gestito dallo script. Kappa, nDCG e DCG usano scikit-learn.

Con una sola query per etichetta e tre annotatori, i pattern per etichetta sono
descrittivi. Gli intervalli condizionano sugli annotatori osservati e il ranking
non dimostra la superiorità generalizzabile di un modello.
