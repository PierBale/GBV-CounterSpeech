"""Build and execute the review notebook using ordinary Python cells.

The notebook itself can also be rerun in Jupyter. Requires IPython for images.
"""

import contextlib
import io
import json
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
cells = []


def markdown(source):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)})


def code(source):
    cells.append({"cell_type": "code", "metadata": {}, "source": source.splitlines(keepends=True),
                  "execution_count": None, "outputs": []})


markdown("""# Accordo e confronto delle configurazioni di retrieval

Analisi della survey **Best-retrieval-results**, run **confronto-02**.

**Tutti gli score umani e le loro medie sono su scala 0–2:**
0 = non rilevante, 1 = parzialmente rilevante, 2 = rilevante.
La decodifica dei codici Qualtrics 1/2/3 usa le scelte effettive del QSF.
Alpha e Kendall sono indici statistici distinti dai voti umani e mantengono le scale standard.

Le domande condivise si contano una volta nell'accordo e si attribuiscono a tutte
le pipeline pertinenti per valutare il retrieval. Eseguibile dalla root o da `Qualtrics`.
""")
code("""from pathlib import Path
import sys
import importlib
import pandas as pd
from IPython.display import SVG, Markdown

ROOT = next(p for p in (Path.cwd(), *Path.cwd().parents)
            if (p / 'Qualtrics/analyze_retrieval.py').is_file())
sys.path.insert(0, str(ROOT / 'Qualtrics'))
import analyze_retrieval
importlib.reload(analyze_retrieval)
pd.set_option('display.precision', 3)
pd.set_option('display.max_colwidth', 90)
OUT = ROOT / 'Qualtrics/analysis'
analysis = analyze_retrieval.run_analysis(out_dir=OUT, bootstrap=10000, seed=20261103)
audit = analysis['audit']
print(f"Annotatori: {audit['n_annotators']} | Item: {audit['n_items']} | Query: {audit['n_queries']} | Etichette: {audit['n_labels']}")
print(f"Selezioni top-3: {audit['n_selections']} | Voti mancanti: {audit['n_missing_ratings']}")
annotations = pd.read_csv(OUT / 'annotations_long.csv')
assert annotations.relevance.between(0, 2).all()
print('Distribuzione score umani (0-2):')
print(annotations.relevance.value_counts().sort_index().rename_axis('Score umano').to_string())
""")
markdown("""## Accordo tra annotatori

**Alpha di Krippendorff ordinale** considera l'ordine 0 < 1 < 2, più annotatori
e voti mancanti. Si calcola prima di aggregare i voti. Gli IC al 95% sono bootstrap
degli item, condizionatamente agli annotatori osservati. La misura principale per
ogni coppia è ancora **alpha ordinale**, con IC bootstrap sui voti ricampionati insieme.
L'alpha globale non è la media degli alpha per coppia. Alpha nominale e kappa
quadratico per coppia restano diagnostiche aggiuntive; l'accordo esatto è una proporzione.
Un alpha negativo indica disaccordo maggiore di quello atteso per caso, non un voto umano negativo.
Il calcolo usa **`import krippendorff`**, con
`krippendorff.alpha(reliability_data=matrix.T, level_of_measurement='ordinal')`.
La libreria richiede annotatori sulle righe e item sulle colonne. I voti mancanti
rimangono NaN; gli item condivisi compaiono una volta. Anche ogni replica bootstrap
usa la libreria. Campioni senza variazione non hanno alpha stimabile.
""")
code("""agreement = analysis['agreement']
display(agreement.loc[agreement.scope == 'global', [
    'n_items', 'n_ratings', 'alpha_ordinal', 'alpha_ci_low', 'alpha_ci_high',
    'alpha_nominal', 'pairwise_exact_agreement', 'all_raters_exact_agreement',
    'pairwise_extreme_disagreement']])
display(analysis['agreement_pairs'][[
    'annotator_a', 'annotator_b', 'n_items', 'alpha_ordinal',
    'alpha_ci_low', 'alpha_ci_high', 'exact_agreement']])
display(SVG(filename=str(OUT / 'agreement_by_label.svg')))
""")
markdown("""## Ranking delle configurazioni

La **rilevanza umana media (0–2)** misura la qualità dei tre passaggi senza considerare l'ordine.
**nDCG@3** considera anche la posizione: gain lineare e sconto `1/log2(rank+1)`.
Il denominatore ideale usa i migliori tre item del pool annotato della query,
uguale per tutti i sistemi. Si usa esclusivamente il gain lineare nativo della libreria.
Le metriche globali sono medie macro delle query; gli IC sono bootstrap delle query.
Il calcolo usa **`sklearn.metrics.ndcg_score`**, con `k=3`, e `dcg_score` per DCG.
Gli item sono allineati per `question_id` sull'intero pool annotato. Gli score
d'ordine 3, 2, 1 ricostruiscono il top-3 originale; gli altri item stanno sotto
il cutoff con score d'ordine 0, mantenendo i loro giudizi umani nell'ideale.
Non si applicano trasformazioni agli score umani prima del calcolo.
Si valuta l'ordine finale estratto: eventuali score grezzi uguali sono risolti
dal rank salvato, senza mediare i possibili ordinamenti a pari merito.
Con ideale tutto zero la libreria restituisce **0**, che viene incluso nella media.
""")
code("""ranking = analysis['summary']
display(ranking[[
    'rank_ndcg', 'rank_relevance', 'pipeline_id', 'mean_relevance',
    'mean_relevance_ci_low', 'mean_relevance_ci_high', 'ndcg_pool_at3',
    'ndcg_pool_at3_ci_low', 'ndcg_pool_at3_ci_high'
]].rename(columns={'mean_relevance': 'Rilevanza umana media (0-2)',
                   'mean_relevance_ci_low': 'Rilevanza IC95 inferiore (0-2)',
                   'mean_relevance_ci_high': 'Rilevanza IC95 superiore (0-2)'}))
display(SVG(filename=str(OUT / 'model_ranking.svg')))
""")
markdown("""## Chi ordina meglio in base agli score?

**nDCG@k** è la metrica classica per ranking con rilevanza graduata; qui `k=3`.
**Kendall tau-b** confronta direttamente l'ordine degli score del reranker con
quello dei giudizi umani e gestisce i pari merito. Score alto = rilevanza alta.
Si calcola per query, poi si media: non si concatenano score di query diverse.

- `kendall_tau_b_common`: usa gli **stessi item annotati**, presenti nei candidati di tutte le configurazioni.
- `kendall_tau_b_top3`: usa i tre item propri di ciascun sistema; le selezioni possono differire.
- `ndcg_order_at3`: ideale sui tre item propri, per studiare l'ordine condizionatamente alla selezione.

Le correlazioni non definite restano mancanti; il numero di query valide è mostrato.
Per nDCG, un ideale tutto zero produce 0 secondo la convenzione di scikit-learn.
L'intersezione comune è un sottoinsieme del pool annotato: non è un test dell'intero ranking.
Un nDCG dell'ordine alto può coesistere con passaggi poco rilevanti o tutti con lo stesso giudizio.
""")
code("""display(ranking[[
    'pipeline_id', 'ndcg_order_at3', 'ndcg_order_at3_n',
    'kendall_tau_b_top3', 'kendall_tau_b_top3_n',
    'kendall_tau_b_common', 'kendall_tau_b_common_n',
    'kendall_tau_b_common_ci_low', 'kendall_tau_b_common_ci_high'
]])
common = pd.read_csv(OUT / 'ordering_common_pool.csv')
print('Item comuni:', common.groupby('query_id').n_common.first().sum())
display(common.pivot(index='edos_label', columns='pipeline_id', values='kendall_tau_b_common'))
""")
markdown("""## Pattern per etichetta EDOS

La heatmap confronta score umano medio su scala 0–2 e nDCG@3. La tabella mostra
accordo sugli item unici, rilevanza del pool e vincitori per etichetta, mantenendo i pari merito.
**Una sola query per etichetta:** le differenze sono descrittive.
Un nDCG alto può comparire anche in una categoria con passaggi quasi tutti irrilevanti:
va letto insieme allo score umano assoluto.
""")
code("""display(SVG(filename=str(OUT / 'patterns_by_label.svg')))
display(analysis['label_patterns'][[
    'edos_label', 'n_items', 'alpha_ordinal', 'alpha_ci_low', 'alpha_ci_high',
    'pool_mean_relevance', 'best_ndcg', 'best_relevance'
]].rename(columns={'pool_mean_relevance': 'Rilevanza umana media del pool (0-2)'}))
""")
markdown("""## Stabilità del ranking e confronti appaiati

Confronto tra **media, mediana e soft label**. Le soft label sono le frequenze
dei voti sulle classi 0, 1 e 2. Il loro score umano `E[Y] = p1 + 2*p2` coincide
con la media: con gain lineare producono quindi gli stessi risultati.

Si usa solo il nDCG classico lineare di scikit-learn, senza trasformazioni degli score.
Media e soft label danno quindi gli stessi valori: non sono due misure indipendenti.
Lo score umano rimane sempre su **0–2**; gli IC sono su query e i ricampionamenti
sono uguali tra le tre scelte di aggregazione.
Con la mediana l'etichetta 3.3 ha ideale tutto zero: ora `ndcg_score` assegna **0**,
in sostituzione del precedente n.d. Il confronto usa quindi **tutte le 11 query**
per tutte le aggregazioni, sia per gli score umani sia per nDCG.
I test tra sistemi usano le stesse query, inversione esatta dei segni e correzione
di Holm: `p_holm` entro aggregazione e metrica, `p_holm_all_configs` su tutti
i confronti delle tre configurazioni. Gli IC dei delta sono bootstrap appaiati.
Con poche query e accordo basso il ranking resta esplorativo.
""")
code("""comparison = analysis['aggregation_comparison']
display(comparison[['aggregation', 'pipeline_id', 'mean_relevance', 'rank_ndcg',
                    'ndcg_pool_at3', 'ndcg_pool_at3_n',
                    'kendall_tau_b_common']].rename(
    columns={'mean_relevance': 'Score umano (0-2)'}))
display(SVG(filename=str(OUT / 'aggregation_comparison.svg')))
display(SVG(filename=str(OUT / 'aggregation_patterns_by_label.svg')))
matched = analysis['aggregation_matched']
print('nDCG@3 classico sulle stesse query per tutte le aggregazioni:')
display(matched.pivot(index='pipeline_id', columns='aggregation',
                         values='ndcg_pool_at3'))
display(matched[['aggregation', 'pipeline_id', 'n_queries', 'rank_ndcg', 'ndcg_pool_at3',
                    'ndcg_pool_at3_ci_low', 'ndcg_pool_at3_ci_high']])
display(pd.read_csv(OUT / 'annotator_model_ranking.csv').sort_values(
    ['annotator', 'rank_ndcg']).rename(columns={'mean_relevance': 'Rilevanza umana (0-2)'}))
display(analysis['paired'][[
    'metric', 'pipeline_a', 'pipeline_b', 'delta_a_minus_b',
    'delta_ci_low', 'delta_ci_high', 'wins_a', 'ties', 'wins_b', 'p_holm']])
""")
markdown("""## Lettura dei risultati

La sintesi seguente è ricalcolata dai dati. Tutti i voti umani sono su scala 0–2;
i delta tra sistemi sono differenze, che possono essere negative.
""")
code("""best = ranking.iloc[0]
global_agreement = agreement.loc[agreement.scope == 'global'].iloc[0]
labels = analysis['label_patterns']
highest_agreement = labels.sort_values('alpha_ordinal', ascending=False).iloc[0]
lowest_relevance = labels.sort_values('pool_mean_relevance').iloc[0]
print(f"Primo per nDCG: {best.pipeline_id}; nDCG@3 = {best.ndcg_pool_at3:.3f}; score umano = {best.mean_relevance:.3f}/2")
print(f"Alpha ordinale globale: {global_agreement.alpha_ordinal:.3f}, IC95 [{global_agreement.alpha_ci_low:.3f}, {global_agreement.alpha_ci_high:.3f}]")
print(f"Confronti significativi al 5% dopo Holm: {(analysis['paired'].p_holm < .05).sum()}")
print(f"Accordo più alto: {highest_agreement.edos_label}, alpha = {highest_agreement.alpha_ordinal:.3f}")
print(f"Pool meno rilevante: {lowest_relevance.edos_label}, score umano = {lowest_relevance.pool_mean_relevance:.3f}/2")
""")
markdown("""## Metodo, limiti e riferimenti

Il pool contiene l'unione dei top-3: non permette recall sull'intero corpus né nDCG@10.
Passaggi e documenti riutilizzati creano dipendenze; gli IC su item/query sono approssimazioni.
Si confrontano pipeline embedding + reranker e l'inferenza condiziona sui tre annotatori osservati.
Il ranking non dimostra una superiorità generalizzabile dei singoli modelli.

- [Krippendorff, Computing Alpha-Reliability](https://www.asc.upenn.edu/sites/default/files/2021-03/Computing%20Krippendorff%27s%20Alpha-Reliability.pdf)
- [Castro (2017), Fast Krippendorff: libreria utilizzata](https://github.com/pln-fing-udelar/fast-krippendorff)
- [Järvelin e Kekäläinen, 2002, Cumulated Gain-based Evaluation of IR Techniques](https://researchportal.tuni.fi/en/publications/cumulated-gain-based-evaluation-of-ir-techniques/)
- [SciPy: Kendall tau-b e pari merito](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kendalltau.html)
- [scikit-learn: implementazione nDCG](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ndcg_score.html)
- [Fornaciari et al. (2021): soft label come distribuzioni dei giudizi](https://aclanthology.org/2021.naacl-main.204/)

Report completo, audit e CSV: `analysis/`. Script: `analyze_retrieval.py`.
Seed 20261103, 10.000 repliche bootstrap. Test: `python -m unittest discover -s Qualtrics/tests -v` dalla root.
""")


def main():
    namespace = {"__name__": "__main__"}
    active_outputs = []

    def capture_display(*objects):
        for obj in objects:
            if hasattr(obj, "_repr_svg_"):
                data = {"image/svg+xml": obj._repr_svg_(), "text/plain": "<SVG>"}
            elif isinstance(obj, pd.DataFrame):
                data = {"text/plain": obj.to_string(), "text/html": obj.to_html()}
            else:
                data = {"text/plain": str(obj)}
            active_outputs.append({"output_type": "display_data", "metadata": {}, "data": data})

    namespace["display"] = capture_display
    execution = 0
    for i, cell in enumerate(cells):
        cell["id"] = f"retrieval-analysis-{i:02d}"
        if cell["cell_type"] != "code":
            continue
        execution += 1
        active_outputs = cell["outputs"]
        captured = io.StringIO()
        with contextlib.redirect_stdout(captured):
            exec(compile("".join(cell["source"]), f"<notebook cell {execution}>", "exec"), namespace)
        if captured.getvalue():
            active_outputs.insert(0, {"output_type": "stream", "name": "stdout", "text": captured.getvalue()})
        cell["execution_count"] = execution
    notebook = {
        "cells": cells, "nbformat": 4, "nbformat_minor": 5,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python", "file_extension": ".py"}},
    }
    path = HERE / "retrieval_analysis.ipynb"
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Saved and executed {execution} code cells: {path}")


if __name__ == "__main__":
    main()
