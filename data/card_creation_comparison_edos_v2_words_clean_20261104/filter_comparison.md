# Revisione dei filtri editoriali — versione 3.0

Nuova esecuzione: `edos-v2-words-clean-20261104`. I file precedenti rimangono conservati. Il parsing dei 18 documenti è riutilizzato. La soglia di 100 parole e i confini delle sottosezioni restano invariati.

| Controllo | Prima | Dopo |
|---|---:|---:|
| Chunk totali | 3805 | 3579 |
| Intestazioni bibliografiche o note | 225 | 76 |
| Intestazioni Contents | 8 | 3 |
| Ringraziamenti | 13 | 0 |
| Directory di membri | 3 | 0 |
| Massimo di parole nel corpo | 1199 | 706 |

Questi conteggi descrivono le intestazioni, non sono percentuali di contenuto non pertinente. Le note esplicative non sono escluse per la sola intestazione. I tre chunk rimasti sotto Contents sono testo introduttivo sostanziale di en_doc_31: il parser non aveva riconosciuto il titolo Introduction. La tabella dell’indice è stata rimossa, mantenendo il testo seguente.

## Regole applicate

- Intestazioni esatte di ringraziamenti, comprese le sottosezioni.
- Intestazioni esplicite di elenchi di membri o directory di organizzazioni: comprende l’elenco WAVE.
- Indici riconosciuti tramite titoli e pagine, anche con righe a capo; tabelle HTML con due colonne titolo/pagina. La sola intestazione Contents non è sufficiente.
- Citazioni riconoscibili tramite autore/anno o autore/titolo/data nelle sezioni bibliografiche o nelle note numerate; URL isolati e Ibid richiedono lo stesso contesto.
- Conservazione dei blocchi misti ambigui: una citazione incollata a un punto sostanziale del documento non comporta la rimozione dell’intero paragrafo.

## Verifiche sui casi reali

- Conservata la nota di en_doc_1 che spiega che le sette strategie non sono mutuamente esclusive.
- Conservato il paragrafo 87 di en_doc_26, dove una citazione era stata attaccata a una definizione di abuso tramite immagini intime.
- Rimossi l’elenco WAVE e le sezioni esplicite di ringraziamenti.
- Conservati i tre blocchi introduttivi di en_doc_31 pur trovandosi sotto Contents.
- I blocchi di 83 e 74 parole discussi in precedenza rimangono nelle esclusioni sotto soglia: la modifica all’aggregazione tra sottosezioni è stata annullata dall’utente e non è stata applicata.

## Audit e limiti

Le nuove regole hanno identificato 952 paragrafi bibliografici. Questo numero non è equivalente al numero di chunk eliminati: il filtraggio avviene prima dell’aggregazione e può cambiare i confini dei chunk.

Le decisioni sono documentate in `editorial_exclusions.jsonl` con testo originale, ID del paragrafo e regola. Le rimozioni parziali hanno `action: trimmed` e conservano anche `removed_text` e `retained_text`. Le regole sono conservative: possono restare citazioni o indici non riconosciuti; non viene dichiarata una pulizia semantica completa. Gli hash e i conteggi sono in `filter_comparison.json`.
