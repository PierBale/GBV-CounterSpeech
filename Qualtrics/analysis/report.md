# Valutazione umana del retrieval

3 annotatori, 101 coppie query–passaggio uniche,
11 query su 11 etichette EDOS e
132 selezioni top-3 (4 configurazioni). Run: confronto-02.

## Risultati principali

- L'accordo corretto per il caso è basso (alpha ordinale 0.297):
  il ranking va interpretato come esplorativo e dipende da giudizi con disaccordo sostanziale.
- **bge-m3__bge-reranker-v2-m3** ha la migliore media di rilevanza (1.141/2)
  e il migliore nDCG@3 (0.759) con il consenso medio.
  Nessuna differenza tra sistemi è significativa al 5% dopo Holm.
- L'accordo più alto è per **2.3 dehumanising attacks & overt sexual objectification**
  (alpha 0.652). Il pool meno rilevante è quello di
  **3.3 backhanded gendered compliments** (0.111/2).
- Il ranking cambia passando dal consenso medio a quello mediano; consultare la tabella di sensibilità. La classifica dei singoli annotatori è esportata separatamente.

## Accordo prima dell'aggregazione

Alpha di Krippendorff **ordinale**: **0.297**,
IC 95% [0.166, 0.419].
Accordo esatto tra coppie di annotatori: 48.5%;
accordo unanime: 28.7%;
disaccordi estremi 0 contro 2: 19.1%.
Non vengono duplicate le annotazioni dei passaggi condivisi.
Il calcolo usa `import krippendorff` e `krippendorff.alpha` (versione 0.8.2),
con `level_of_measurement="ordinal"`; il controllo nominale usa `"nominal"`.
La matrice viene trasposta nel formato richiesto dalla libreria: annotatori × item.
Ogni replica bootstrap richiama la stessa funzione della libreria; la versione
installata non espone una funzione per gli intervalli, quindi il ricampionamento
degli item è gestito dallo script. Campioni senza variazione restano non definiti.

### Accordo per coppia di annotatori

La misura principale anche per le coppie è **alpha di Krippendorff ordinale**,
calcolato sui soli item valutati da entrambi gli annotatori, mantenendo i voti umani 0/1/2.
Gli IC 95% sono bootstrap degli item con i due voti ricampionati insieme.
Il kappa quadratico resta nel CSV come diagnostica aggiuntiva.
Alpha ordinale e kappa quadratico usano distanze e modelli di accordo atteso
diversi: i loro valori non sono direttamente intercambiabili.
L'alpha globale è stimato congiuntamente sui tre annotatori, non è la media degli alpha per coppia.

| annotator_a | annotator_b | n_items | alpha_ordinal | alpha_ci_low | alpha_ci_high | exact_agreement |
| --- | --- | --- | --- | --- | --- | --- |
| ann_03ecaf62 | ann_1fb83fbd | 101 | 0.233 | 0.033 | 0.413 | 0.455 |
| ann_03ecaf62 | ann_948ed9de | 101 | 0.221 | 0.026 | 0.403 | 0.455 |
| ann_1fb83fbd | ann_948ed9de | 101 | 0.401 | 0.218 | 0.565 | 0.545 |

## Ranking delle configurazioni

Ordine principale per nDCG@3 sul pool comune annotato di ogni query.
La rilevanza media è una seconda classifica, indipendente dalla posizione.
La frazione `fully_relevant_fraction` è la quota di singoli giudizi uguali a 2,
non la precisione calcolata su una soglia del consenso.

| rank_ndcg | rank_relevance | pipeline_id | mean_relevance | ndcg_pool_at3 | ndcg_order_at3 | kendall_tau_b_top3 | kendall_tau_b_common |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | bge-m3__bge-reranker-v2-m3 | 1.141 | 0.759 | 0.942 | 0.018 | 0.266 |
| 2 | 2 | qwen3-embedding-8b__qwen3-reranker-8b | 1.071 | 0.677 | 0.919 | 0.164 | 0.074 |
| 3 | 3 | bge-m3__qwen3-reranker-8b | 1.000 | 0.616 | 0.916 | 0.331 | 0.059 |
| 4 | 4 | qwen3-embedding-8b__bge-reranker-v2-m3 | 0.990 | 0.603 | 0.822 | 0.037 | 0.266 |

Gli intervalli delle metriche e il numero di query valide sono in `model_ranking.csv`.
I confronti appaiati sono in `paired_comparisons.csv`: delta A−B, IC 95%, test
esatto a inversione dei segni e correzione di Holm **separata per metrica**.
Il test presuppone scambiabilità dei due sistemi sotto l'ipotesi nulla;
con sole 11 query la potenza statistica è limitata.

| metric | pipeline_a | pipeline_b | delta_a_minus_b | delta_ci_low | delta_ci_high | p_holm |
| --- | --- | --- | --- | --- | --- | --- |
| ndcg_pool_at3 | bge-m3__bge-reranker-v2-m3 | bge-m3__qwen3-reranker-8b | 0.143 | -0.031 | 0.299 | 0.708 |
| ndcg_pool_at3 | bge-m3__bge-reranker-v2-m3 | qwen3-embedding-8b__bge-reranker-v2-m3 | 0.156 | 0.040 | 0.294 | 0.176 |
| ndcg_pool_at3 | bge-m3__bge-reranker-v2-m3 | qwen3-embedding-8b__qwen3-reranker-8b | 0.082 | -0.102 | 0.286 | 1.000 |
| ndcg_pool_at3 | bge-m3__qwen3-reranker-8b | qwen3-embedding-8b__bge-reranker-v2-m3 | 0.013 | -0.182 | 0.216 | 1.000 |
| ndcg_pool_at3 | bge-m3__qwen3-reranker-8b | qwen3-embedding-8b__qwen3-reranker-8b | -0.061 | -0.184 | 0.061 | 1.000 |
| ndcg_pool_at3 | qwen3-embedding-8b__bge-reranker-v2-m3 | qwen3-embedding-8b__qwen3-reranker-8b | -0.074 | -0.243 | 0.123 | 1.000 |
| mean_relevance | bge-m3__bge-reranker-v2-m3 | bge-m3__qwen3-reranker-8b | 0.141 | -0.141 | 0.374 | 1.000 |
| mean_relevance | bge-m3__bge-reranker-v2-m3 | qwen3-embedding-8b__bge-reranker-v2-m3 | 0.152 | 0.010 | 0.283 | 0.562 |
| mean_relevance | bge-m3__bge-reranker-v2-m3 | qwen3-embedding-8b__qwen3-reranker-8b | 0.071 | -0.212 | 0.394 | 1.000 |
| mean_relevance | bge-m3__qwen3-reranker-8b | qwen3-embedding-8b__bge-reranker-v2-m3 | 0.010 | -0.283 | 0.354 | 1.000 |
| mean_relevance | bge-m3__qwen3-reranker-8b | qwen3-embedding-8b__qwen3-reranker-8b | -0.071 | -0.293 | 0.162 | 1.000 |
| mean_relevance | qwen3-embedding-8b__bge-reranker-v2-m3 | qwen3-embedding-8b__qwen3-reranker-8b | -0.081 | -0.374 | 0.263 | 1.000 |

## Pattern per etichetta

`label_patterns.csv` riassume l'accordo sugli item unici, la rilevanza dei pool
e i vincitori per rilevanza/nDCG. I pari merito sono mantenuti.
`metrics_by_label.csv` contiene ogni configurazione × etichetta e la posizione
relativa del sistema; `patterns_by_label.svg` consente il confronto visivo.

| edos_label | n_items | alpha_ordinal | pool_mean_relevance | best_ndcg | best_relevance |
| --- | --- | --- | --- | --- | --- |
| 1.1 threats of harm | 11 | 0.374 | 1.303 | qwen3-embedding-8b__qwen3-reranker-8b | bge-m3__qwen3-reranker-8b; qwen3-embedding-8b__qwen3-reranker-8b |
| 1.2 incitement and encouragement of harm | 8 | 0.152 | 1.625 | qwen3-embedding-8b__qwen3-reranker-8b | bge-m3__bge-reranker-v2-m3; qwen3-embedding-8b__bge-reranker-v2-m3 |
| 2.1 descriptive attacks | 10 | -0.069 | 0.833 | bge-m3__bge-reranker-v2-m3 | bge-m3__bge-reranker-v2-m3 |
| 2.2 aggressive and emotive attacks | 6 | 0.089 | 1.111 | qwen3-embedding-8b__qwen3-reranker-8b | qwen3-embedding-8b__qwen3-reranker-8b |
| 2.3 dehumanising attacks & overt sexual objectification | 12 | 0.652 | 1.250 | bge-m3__qwen3-reranker-8b | bge-m3__qwen3-reranker-8b |
| 3.1 casual use of gendered slurs, profanities, and insults | 9 | 0.161 | 0.852 | bge-m3__bge-reranker-v2-m3 | bge-m3__bge-reranker-v2-m3 |
| 3.2 immutable gender differences and gender stereotypes | 8 | 0.223 | 1.042 | bge-m3__bge-reranker-v2-m3 | bge-m3__bge-reranker-v2-m3 |
| 3.3 backhanded gendered compliments | 9 | -0.083 | 0.111 | bge-m3__bge-reranker-v2-m3 | bge-m3__bge-reranker-v2-m3 |
| 3.4 condescending explanations or unwelcome advice | 10 | -0.055 | 1.000 | qwen3-embedding-8b__qwen3-reranker-8b | qwen3-embedding-8b__qwen3-reranker-8b |
| 4.1 supporting mistreatment of individual women | 8 | 0.269 | 1.417 | qwen3-embedding-8b__qwen3-reranker-8b | qwen3-embedding-8b__bge-reranker-v2-m3; qwen3-embedding-8b__qwen3-reranker-8b |
| 4.2 supporting systemic discrimination against women as a group | 10 | -0.144 | 0.867 | bge-m3__bge-reranker-v2-m3 | bge-m3__bge-reranker-v2-m3 |

## Sensibilità al consenso

Il confronto usa le stesse selezioni e lo stesso pool annotato per ogni query.
La media e la mediana producono uno score scalare su 0–2. Le **soft label**
conservano le frequenze empiriche `p0`, `p1`, `p2`, esportate in `item_consensus.csv`.
La loro rilevanza attesa è `p1 + 2*p2`: coincide esattamente con la media dei voti.
Quindi, con gain lineare, media e soft label danno gli stessi score e nDCG.

Si usa solo il **gain lineare nativo di scikit-learn**, senza trasformare i voti:
`ndcg_score(y_true, y_score, k=3)`. `y_true` contiene media, mediana o valore atteso
delle soft label degli item del pool. Non viene calcolata la media degli nDCG
dei singoli annotatori. Il denominatore è comune alle quattro pipeline entro
ogni query e variante. Media e soft label sono due rappresentazioni degli stessi
score attesi e non due misure indipendenti in questo confronto.

| aggregation | pipeline_id | mean_relevance | rank_ndcg | ndcg_pool_at3 |
| --- | --- | --- | --- | --- |
| mean | bge-m3__bge-reranker-v2-m3 | 1.141 | 1 | 0.759 |
| mean | qwen3-embedding-8b__qwen3-reranker-8b | 1.071 | 2 | 0.677 |
| mean | bge-m3__qwen3-reranker-8b | 1.000 | 3 | 0.616 |
| mean | qwen3-embedding-8b__bge-reranker-v2-m3 | 0.990 | 4 | 0.603 |
| median | bge-m3__bge-reranker-v2-m3 | 1.212 | 1 | 0.679 |
| median | qwen3-embedding-8b__qwen3-reranker-8b | 1.061 | 2 | 0.610 |
| median | qwen3-embedding-8b__bge-reranker-v2-m3 | 1.061 | 3 | 0.578 |
| median | bge-m3__qwen3-reranker-8b | 0.939 | 4 | 0.510 |
| soft | bge-m3__bge-reranker-v2-m3 | 1.141 | 1 | 0.759 |
| soft | qwen3-embedding-8b__qwen3-reranker-8b | 1.071 | 2 | 0.677 |
| soft | bge-m3__qwen3-reranker-8b | 1.000 | 3 | 0.616 |
| soft | qwen3-embedding-8b__bge-reranker-v2-m3 | 0.990 | 4 | 0.603 |

**Convenzione scikit-learn e supporto comune.** Con la mediana, l'etichetta 3.3
ha ideale tutto zero. `sklearn.metrics.ndcg_score` restituisce **0** in questo caso;
si adotta direttamente la convenzione della libreria, al posto del precedente n.d.
Tutte le aggregazioni includono quindi le stesse
**11 query**, inclusa la 3.3:

| aggregation | pipeline_id | n_queries | rank_ndcg | ndcg_pool_at3 |
| --- | --- | --- | --- | --- |
| mean | bge-m3__bge-reranker-v2-m3 | 11 | 1 | 0.759 |
| mean | qwen3-embedding-8b__qwen3-reranker-8b | 11 | 2 | 0.677 |
| mean | bge-m3__qwen3-reranker-8b | 11 | 3 | 0.616 |
| mean | qwen3-embedding-8b__bge-reranker-v2-m3 | 11 | 4 | 0.603 |
| median | bge-m3__bge-reranker-v2-m3 | 11 | 1 | 0.679 |
| median | qwen3-embedding-8b__qwen3-reranker-8b | 11 | 2 | 0.610 |
| median | qwen3-embedding-8b__bge-reranker-v2-m3 | 11 | 3 | 0.578 |
| median | bge-m3__qwen3-reranker-8b | 11 | 4 | 0.510 |
| soft | bge-m3__bge-reranker-v2-m3 | 11 | 1 | 0.759 |
| soft | qwen3-embedding-8b__qwen3-reranker-8b | 11 | 2 | 0.677 |
| soft | bge-m3__qwen3-reranker-8b | 11 | 3 | 0.616 |
| soft | qwen3-embedding-8b__bge-reranker-v2-m3 | 11 | 4 | 0.603 |

Questi risultati e i loro IC sono in `aggregation_common_query_ranking.csv`.
Nel grafico di confronto, score umano e nDCG usano ora tutte le 11 query.
L'ideale nullo ha nDCG zero anche per la metrica condizionata all'ordine dei tre item.
Le correlazioni non definite restano invece mancanti.

`aggregation_sensitivity.csv` contiene anche IC 95%, query valide e Kendall tau-b.
`aggregation_metrics_by_label.csv` mostra ogni aggregazione × query × pipeline.
`aggregation_comparison.svg` e `aggregation_patterns_by_label.svg` mostrano il confronto.
`aggregation_paired_comparisons.csv` contiene i test appaiati per score umano e
nDCG classico: `p_holm` corregge le sei coppie entro aggregazione e metrica;
`p_holm_all_configs` corregge l'intera famiglia di 36 confronti.
Con questa correzione complessiva, i confronti con p < 0,05 sono
0.

`annotator_model_ranking.csv` mostra inoltre la classifica separata per annotatore,
senza attribuire a nessuno il ruolo di gold. L'agreement si calcola sui voti originali
e non cambia passando da media a mediana o soft label.

## Metodo e limiti

- Dal QSF si ricava la corrispondenza tra codici Qualtrics 1/2/3 e rilevanza 0/1/2.
  Le due righe di metadati, le anteprime e le risposte non terminate sono escluse.
  Eventuali giudizi mancanti restano mancanti; non sono convertiti in zero.
- Un item è una **coppia query–chunk**, perché lo stesso chunk può essere rilevante
  per un'etichetta e irrilevante per un'altra. Per i sistemi, `source_matches`
  restituisce ogni item condiviso a tutti i sistemi e ai rispettivi rank originali.
- Alpha ordinale usa distanze basate sulle frequenze marginali delle categorie.
  L'alpha nominale e l'accordo esatto sono diagnostiche aggiuntive.
- Consenso principale: media dei voti 0/1/2. Questo assegna utilità lineare alla scala.
  DCG@3 = somma di `relevance_i / log2(i+1)`, i=1..3.
  nDCG@3 divide DCG per l'ideale dei **migliori tre item nell'unione annotata**
  dei top-3 di tutte le configurazioni della stessa query: denominatore uguale
  per tutti i sistemi, senza applicare trasformazioni ai voti.
- Implementazione: `sklearn.metrics.ndcg_score` (scikit-learn 1.7.2),
  `k=3`, `ignore_ties=False`. Per query, `y_true` contiene i gain di **tutti** gli
  item annotati, allineati con `y_score` per `question_id`. I tre item selezionati
  ricevono score d'ordine 3, 2, 1 secondo il rank originale; gli altri ricevono 0,
  sotto il cutoff, conservando i propri gain umani per il calcolo dell'ideale.
  I pari merito sotto il cutoff non cambiano il top-3. DCG usa `sklearn.metrics.dcg_score`.
  Si valuta il ranking finale estratto: eventuali score grezzi uguali nei tre item
  sono risolti secondo il rank esportato, anziché mediando i possibili ordinamenti.
- `ndcg_order_at3` usa come ideale solo i tre item scelti dallo stesso sistema:
  misura l'ordine condizionatamente alla selezione, può essere alto anche con
  passaggi poco utili. Se l'ideale è tutto zero, nDCG vale 0 secondo scikit-learn
  ed entra nella media; il numero di valori validi è sempre esportato.
- Kendall tau-b confronta **score del reranker e consenso umano nella stessa query**,
  con score alto = rilevanza alta, gestendo i pari merito. Spearman rho è esportato
  come controllo. Le correlazioni non sono definite con meno di due item o valori
  costanti; non sono sostituite da zero e non si concatenano score di query diverse.
- `kendall_tau_b_top3` valuta i tre item propri di ogni sistema. Per un confronto
  dell'ordine sullo **stesso insieme di item**, `kendall_tau_b_common` usa i soli item
  annotati presenti in `reranking_full` di tutte e quattro le configurazioni.
  L'intersezione contiene 39 item; copertura e query utilizzabili
  sono in `ordering_common_pool.csv`. Il confronto resta condizionato a questo sottoinsieme.
- Le metriche globali sono medie macro delle query, con uguale peso per etichetta
  in questo dataset. IC dei modelli: bootstrap delle query; confronti: bootstrap
  appaiato. IC alpha: bootstrap degli item, globale e per etichetta.
  10000 repliche, seed 20261103. Inferenza condizionata ai tre annotatori osservati;
  non stima la variabilità di una popolazione di annotatori.
- C'è **una sola query per etichetta**. I pattern per etichetta sono descrittivi,
  non prove di differenze generalizzabili. L'alpha globale può anche riflettere
  differenze tra etichette: va letto insieme agli alpha per etichetta.
- Passaggi riutilizzati tra etichette e documenti comuni introducono dipendenze;
  gli IC basati su item/query sono approssimazioni. I sistemi condividono anche
  alcune selezioni: non vanno trattate come campioni indipendenti.
- Solo i top-3 riuniti sono annotati. Non si possono stimare recall sull'intero
  corpus, nDCG@10 o qualità di tutto il ranking senza ulteriori giudizi.
  Il pool è costruito dai sistemi valutati: l'ideale è relativo a quel pool.
- Si confrontano **pipeline embedding + reranker**, non modelli isolati:
  l'effetto dei due componenti non si identifica causalmente da questo campione.

## Riferimenti

- Krippendorff, *Computing Krippendorff's Alpha-Reliability*: https://www.asc.upenn.edu/sites/default/files/2021-03/Computing%20Krippendorff%27s%20Alpha-Reliability.pdf
- Castro (2017), Fast Krippendorff, libreria utilizzata: https://github.com/pln-fing-udelar/fast-krippendorff
- Järvelin & Kekäläinen (2002), *Cumulated Gain-based Evaluation of IR Techniques*,
  ACM TOIS, DOI 10.1145/582415.582418: https://researchportal.tuni.fi/en/publications/cumulated-gain-based-evaluation-of-ir-techniques/
- SciPy, `kendalltau`, formula tau-b e gestione dei pari merito: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kendalltau.html
- scikit-learn, implementazione nDCG: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ndcg_score.html
- Fornaciari et al. (2021), soft label come distribuzioni dei giudizi: https://aclanthology.org/2021.naacl-main.204/

## Riproduzione

```powershell
python Qualtrics/analyze_retrieval.py --bootstrap 10000 --seed 20261103
python -m unittest discover -s Qualtrics/tests -v
```

Input e hash SHA-256 sono in `analysis_audit.json`. I dati originali non sono modificati.
