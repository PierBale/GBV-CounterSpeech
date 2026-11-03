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
| 4 | 4 | qwen3-embedding-8b__bge-reranker-v2-m3 | 0.990 | 0.603 | 0.904 | 0.037 | 0.266 |

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

Per studiare l'effetto della distribuzione si aggiunge il gain esponenziale:

- media: `g = 2^mean − 1`;
- mediana: `g = 2^median − 1`;
- soft label: `g = E[2^Y − 1] = p1 + 3*p2`.

È una scelta esplicita di utilità per questa analisi: `E[g(Y)]` non coincide con
`g(E[Y])`. Il gain esponenziale atteso può favorire item polarizzati a parità di media;
non è automaticamente una stima più corretta della rilevanza. I gain sono riscalati
con il fattore 2/3 su 0–2 (nDCG è invariato); restano utilità derivate, distinte
dai voti umani. Lo score umano riportato per soft label è sempre `E[Y]` su 0–2.

Per ciascuna variante nDCG divide DCG dei gain per l'ideale ottenuto ordinando
gli stessi gain nel pool. È **nDCG dei gain attesi**, non la media degli nDCG
dei singoli annotatori né un'aspettativa del rapporto su possibili etichette.
Il denominatore è comune alle quattro pipeline entro ogni query e variante.

| aggregation | pipeline_id | mean_relevance | rank_ndcg | ndcg_pool_at3 | rank_ndcg_exponential | ndcg_pool_at3_exponential |
| --- | --- | --- | --- | --- | --- | --- |
| mean | bge-m3__bge-reranker-v2-m3 | 1.141 | 1 | 0.759 | 1 | 0.701 |
| mean | qwen3-embedding-8b__qwen3-reranker-8b | 1.071 | 2 | 0.677 | 2 | 0.621 |
| mean | bge-m3__qwen3-reranker-8b | 1.000 | 3 | 0.616 | 3 | 0.548 |
| mean | qwen3-embedding-8b__bge-reranker-v2-m3 | 0.990 | 4 | 0.603 | 4 | 0.543 |
| median | bge-m3__bge-reranker-v2-m3 | 1.212 | 1 | 0.747 | 1 | 0.721 |
| median | qwen3-embedding-8b__qwen3-reranker-8b | 1.061 | 2 | 0.672 | 2 | 0.647 |
| median | qwen3-embedding-8b__bge-reranker-v2-m3 | 1.061 | 3 | 0.636 | 3 | 0.600 |
| median | bge-m3__qwen3-reranker-8b | 0.939 | 4 | 0.561 | 4 | 0.507 |
| soft | bge-m3__bge-reranker-v2-m3 | 1.141 | 1 | 0.759 | 1 | 0.740 |
| soft | qwen3-embedding-8b__qwen3-reranker-8b | 1.071 | 2 | 0.677 | 2 | 0.659 |
| soft | bge-m3__qwen3-reranker-8b | 1.000 | 3 | 0.616 | 3 | 0.589 |
| soft | qwen3-embedding-8b__bge-reranker-v2-m3 | 0.990 | 4 | 0.603 | 4 | 0.572 |

**Query valide e confronto a parità di supporto.** Con la mediana, l'etichetta 3.3
ha ideale tutto zero: il suo nDCG non è definito. Le medie disponibili sopra usano
11 query per media/soft e 10 per mediana; non sono un confronto sullo stesso supporto.
Il confronto di nDCG fra aggregazioni usa quindi le
**10 query valide in tutte le varianti**:

| aggregation | pipeline_id | n_queries | rank_ndcg | ndcg_pool_at3 | rank_ndcg_exponential | ndcg_pool_at3_exponential |
| --- | --- | --- | --- | --- | --- | --- |
| mean | bge-m3__bge-reranker-v2-m3 | 10 | 1 | 0.765 | 1 | 0.701 |
| mean | qwen3-embedding-8b__qwen3-reranker-8b | 10 | 2 | 0.721 | 2 | 0.660 |
| mean | qwen3-embedding-8b__bge-reranker-v2-m3 | 10 | 3 | 0.663 | 3 | 0.598 |
| mean | bge-m3__qwen3-reranker-8b | 10 | 4 | 0.631 | 4 | 0.556 |
| median | bge-m3__bge-reranker-v2-m3 | 10 | 1 | 0.747 | 1 | 0.721 |
| median | qwen3-embedding-8b__qwen3-reranker-8b | 10 | 2 | 0.672 | 2 | 0.647 |
| median | qwen3-embedding-8b__bge-reranker-v2-m3 | 10 | 3 | 0.636 | 3 | 0.600 |
| median | bge-m3__qwen3-reranker-8b | 10 | 4 | 0.561 | 4 | 0.507 |
| soft | bge-m3__bge-reranker-v2-m3 | 10 | 1 | 0.765 | 1 | 0.744 |
| soft | qwen3-embedding-8b__qwen3-reranker-8b | 10 | 2 | 0.721 | 2 | 0.701 |
| soft | qwen3-embedding-8b__bge-reranker-v2-m3 | 10 | 3 | 0.663 | 3 | 0.630 |
| soft | bge-m3__qwen3-reranker-8b | 10 | 4 | 0.631 | 4 | 0.601 |

Questi risultati e i loro IC sono in `aggregation_common_query_ranking.csv`.
Nel grafico di confronto, il pannello score umano usa tutte le 11 query,
mentre i pannelli nDCG usano le query valide comuni. Non si sostituisce con zero
il nDCG non definito. Le tabelle per etichetta mostrano esplicitamente i mancanti.

`aggregation_sensitivity.csv` contiene anche IC 95%, query valide e Kendall tau-b.
`aggregation_metrics_by_label.csv` mostra ogni aggregazione × query × pipeline.
`aggregation_comparison.svg` e `aggregation_patterns_by_label.svg` mostrano il confronto.
`aggregation_paired_comparisons.csv` contiene i test appaiati per score umano e
i due nDCG: `p_holm` corregge le sei coppie entro aggregazione e metrica;
`p_holm_all_configs` corregge l'intera famiglia di 54 confronti.
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
  per tutti i sistemi. `ndcg_pool_at3_exponential` usa `2^relevance−1` come sensibilità.
- `ndcg_order_at3` usa come ideale solo i tre item scelti dallo stesso sistema:
  misura l'ordine condizionatamente alla selezione, può essere alto anche con
  passaggi poco utili. Se l'ideale è tutto zero, nDCG è non definito ed escluso
  dalla media; il numero di valori validi è sempre esportato.
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
- Järvelin & Kekäläinen (2002), *Cumulated Gain-based Evaluation of IR Techniques*,
  ACM TOIS, DOI 10.1145/582415.582418: https://researchportal.tuni.fi/en/publications/cumulated-gain-based-evaluation-of-ir-techniques/
- SciPy, `kendalltau`, formula tau-b e gestione dei pari merito: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kendalltau.html
- Fornaciari et al. (2021), soft label come distribuzioni dei giudizi: https://aclanthology.org/2021.naacl-main.204/

## Riproduzione

```powershell
python Qualtrics/analyze_retrieval.py --bootstrap 10000 --seed 20261103
python -m unittest discover -s Qualtrics/tests -v
```

Input e hash SHA-256 sono in `analysis_audit.json`. I dati originali non sono modificati.
