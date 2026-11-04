# Annotazione finale: context relevance

Importare ciascuno dei quattro QSF come un questionario distinto in Qualtrics.

| File | Chi lo compila | Item di context relevance |
|---|---|---:|
| `comune.qsf` | marem, arianna e marta | 29 |
| `marem.qsf` | marem | 27 |
| `arianna.qsf` | arianna | 27 |
| `marta.qsf` | marta | 27 |

Ogni annotatrice compila il comune e il proprio questionario: **56 item a testa**,
168 giudizi complessivi. I 29 item comuni sono il 26,36% dei 110 item univoci.
Nel comune si seleziona obbligatoriamente il nome dell'annotatrice all'inizio.

Il pool parte dalle posizioni **1–4 del ranking finale di ciascuna delle quattro
configurazioni per ognuna delle 11 etichette EDOS**, nel run
`edos-v2-words-clean-v4b-20261104`: 176 selezioni originali, unite in 110 item.
Un item è la coppia `(query_id, chunk_id)`. Se più configurazioni recuperano lo
stesso chunk per la stessa etichetta, si presenta una sola volta e si conservano
tutti i modelli e i rank in `source_matches`. Lo stesso chunk associato a un'altra
etichetta richiede un giudizio distinto.

I quattro questionari sono disgiunti: gli item comuni non sono ripetuti nei tre
pool individuali. Ciascuno dei 110 item appartiene a un solo QSF. Le prime quattro
posizioni di ogni configurazione sono quindi coperte dal pool complessivo dei
quattro questionari.

## Assegnazione e ordine

I 29 comuni sono campionati in proporzione alla numerosità del pool di ogni
etichetta, usando il metodo dei maggiori resti per i conteggi interi. Tutte le
etichette sono rappresentate. I restanti 81 item sono distribuiti in tre pool
da 27: per ogni etichetta, le numerosità dei pool individuali differiscono al
massimo di uno.

Le etichette sono in ordine numerico EDOS, da 1.1 a 4.2. Gli item all'interno di
ogni etichetta sono mescolati **prima della scrittura del QSF**, con seed
`20261104`. L'ordine è fisso per tutte le compilazioni dello stesso questionario:
non ci sono randomizzatori Qualtrics, né dei blocchi né delle domande.

Ogni item ha una pagina con:

- etichetta e definizione EDOS, testo del chunk e valutazione obbligatoria
  **0 = Not relevant, 1 = Partially relevant, 2 = Relevant**;
- campo note multilinea **facoltativo**, copiato dal draft originale, sulla
  stessa pagina della valutazione.

Il testo introduttivo, le scelte e la configurazione delle note provengono da
`Draft-retrieval-setting.qsf`, copiato dal pilot senza modificarlo. I testi dei
chunk e le definizioni corrispondono al run; le tabelle HTML sono mantenute.
Nell'interfaccia non sono mostrati modelli, rank o score.

## Mappe e riproducibilità

Ogni `*_question_map.csv` associa l'item al QID della valutazione, al QID e al
tag delle note, alla posizione effettiva nel questionario e a tutte le
configurazioni e posizioni originali. I tag degli item sono stabili rispetto
all'ordine e all'assegnazione. I QID Qualtrics sono locali a ogni questionario:
usare sempre anche il nome del questionario per interpretarli.

`annotation_assignment.json` raccoglie assegnazioni, conteggi per etichetta,
seed, hash SHA-256 di input, template, generatore, QSF e mappe. Le risposte future
del comune e dei tre pool vanno conservate separatamente, con il rispettivo QSF
e question map.

Per rigenerare gli stessi file dalla radice del progetto, usando solo Python
standard:

```powershell
python Qualtrics/retrieval_internal_annotation_final/build_surveys.py
python -m unittest discover -s Qualtrics/retrieval_internal_annotation_final/tests -v
```

Il generatore accetta `--seed`, `--input-dir`, `--template` e `--output-dir`.
Se i file esistenti differiscono, la sostituzione richiede `--overwrite`.
I risultati e il codice del pilot rimangono nella cartella precedente.
