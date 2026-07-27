# Checkpoint ufficiali Khan et al. — EDOS Task C

Questa pipeline usa direttamente i cinque checkpoint pubblicati dagli autori di
Khan et al. (ACL 2025). Non riaddestra alcun modello e non esegue Task A o Task
B.

## Cosa implementa

Per Task C, il paper usa:

| Chiave | Checkpoint Hugging Face | Ruolo | Macro F1 riportato |
|---|---|---|---:|
| `deberta9` | `sahrishkhan/edos-deberta-9-c-model` | voto base | 0.5571 |
| `dtfn3` | `sahrishkhan/edos-roberta-deberta-3-c-model` | voto base | 0.5352 |
| `dtfn7` | `sahrishkhan/edos-roberta-deberta-7-c-model` | voto base | 0.5385 |
| `dtfn8` | `sahrishkhan/edos-roberta-deberta-8-c-model` | voto base | 0.5344 |
| `mistral` | `sahrishkhan/edos-mistral-c-model` | fallback | 0.5120 |

I primi quattro modelli effettuano un hard vote. Se più classi condividono il
numero massimo di voti, viene scelta la predizione di Mistral-7B. Il paper
riporta macro F1 `0.6018` per questo M7-FE.

I tre DTFN concatenano il token CLS di DeBERTa-v3-Large e RoBERTa-Large e
applicano un classificatore lineare a 11 classi. Mistral ricrea la
configurazione QLoRA degli autori: 4 bit NF4, `r=16`, `alpha=8`, dropout `0.05`
e target `q_proj`, `k_proj`, `v_proj`, `o_proj`.

## Requisiti del server

- Linux con GPU CUDA; Mistral usa `bitsandbytes` in 4 bit.
- Almeno circa 70 GB liberi tra checkpoint e cache dei modelli base.
- Preferibilmente almeno 32 GB di RAM. I `.pth` pubblicati contengono anche
  stato di optimizer e scheduler: i soli cinque file occupano circa 37.8 GB.
- Python 3.10 o 3.11.

Installazione:

```bash
cd "EDOS Classifier"
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-official.txt
```

## Download

Il downloader usa revisioni Hugging Face fissate, così un futuro aggiornamento
dei repository non cambia silenziosamente i risultati:

```bash
python scripts/11_download_official_task_c.py \
  --output-dir models/official_task_c
```

È possibile scaricare un checkpoint alla volta:

```bash
python scripts/11_download_official_task_c.py \
  --output-dir models/official_task_c \
  --models deberta9
```

## Pipeline completa

Dopo il download:

```bash
bash run_official_task_c.sh
```

Download e pipeline in un solo comando:

```bash
DOWNLOAD_MODELS=1 bash run_official_task_c.sh
```

Per SLURM o un percorso dati diverso si possono impostare le variabili:

```bash
EDOS_CSV=/path/edos_labelled_aggregated.csv \
CONAN_JSON=/path/WOMAN-Multitarget-CONAN.json \
CHECKPOINT_ROOT=/scratch/checkpoints/official_task_c \
OUTPUT_ROOT=/scratch/results/official_task_c \
DEVICE=cuda:0 \
bash run_official_task_c.sh
```

I modelli sono caricati ed eseguiti in processi separati. Non è quindi
necessario tenere contemporaneamente in GPU tutti e cinque i modelli.

## Output

Valutazione EDOS:

```text
outputs/official_task_c/edos/
  deberta9.csv
  deberta9_metrics.json
  dtfn3.csv
  dtfn3_metrics.json
  dtfn7.csv
  dtfn7_metrics.json
  dtfn8.csv
  dtfn8_metrics.json
  mistral.csv
  mistral_metrics.json
  m7fe.csv
  m7fe_metrics.json
```

Il loader seleziona esclusivamente i 970 esempi con `split=test` e
`label_sexist=sexist`, come richiesto da EDOS Task C. Il file delle metriche
contiene macro F1, weighted F1, accuracy, confusion matrix e report per classe.

Annotazione CONAN:

```text
outputs/official_task_c/conan/m7fe.csv
data/conan/WOMAN-Multitarget-CONAN_EDOS_TASK_C_M7FE.json
```

Ogni elemento conserva i campi originali e riceve soltanto:

```json
"HATE_SPEECH_EDOS_PREDICTIONS": {
  "TASK_C": {
    "label": "3.2 immutable gender differences and gender stereotypes",
    "confidence": 0.71,
    "confidence_kind": "mean_base_probability_for_majority_label",
    "method": "M7-FE official checkpoints",
    "used_mistral_fallback": false,
    "top_vote_count": 3,
    "vote_counts": {},
    "base_votes": {},
    "fallback_pred": "2.1 descriptive attacks",
    "is_predicted_not_gold": true
  }
}
```

La confidence non fa parte della regola M7-FE del paper. Quando c'è una
maggioranza è la media, sui quattro modelli base, della probabilità della label
scelta; quando viene usato il fallback è la probabilità assegnata da Mistral.
Non va interpretata come probabilità calibrata.

## Caricamento sicuro dei `.pth`

La pipeline prova prima `torch.load(weights_only=True)`. Se la versione con cui
gli autori hanno serializzato un checkpoint non è compatibile con il caricamento
sicuro, il processo si arresta. Solo dopo aver verificato che il file provenga
dal repository ufficiale si può abilitare il fallback pickle:

```bash
ALLOW_UNSAFE_CHECKPOINT_LOAD=1 bash run_official_task_c.sh
```

## Limite metodologico

Task C presuppone che l'input sia sessista e assegna sempre una delle 11
categorie. Il subset CONAN qui incluso contiene hate speech con target `WOMEN`,
ma la label aggiunta resta una predizione automatica, non un'annotazione gold.
La pipeline non effettua un filtro Task A.

Fonti:

- Paper: <https://aclanthology.org/2025.acl-long.809/>
- Codice: <https://github.com/Sahrish42/explaining_matters_sexism_detection_acl2025>
- Modelli: <https://huggingface.co/sahrishkhan/models>
