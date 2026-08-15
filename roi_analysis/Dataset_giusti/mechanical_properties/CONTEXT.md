# Mechanical Properties — Contesto e Struttura

## Panoramica

Questa cartella contiene i dati delle **proprietà meccaniche** rilevate sperimentalmente
sui provini stampati in FDM. Il test consiste nell'applicare carichi crescenti sul provino
e registrare se regge (`ok`) o cede (`ROTTO`) a ogni step di peso.

L'obiettivo è correlare il **peso di rottura** (la proprietà meccanica chiave) con il
**profilo termico** misurato durante la stampa, in funzione della durata della pausa
tra i layer (`pausa_s`). Questo rappresenta la variabile target per i modelli predittivi
descritti nel `CONTEXT.md` della cartella padre.

---

## Provini Testati

I 12 provini validi coprono 5 condizioni di pausa:

| Condizione | pausa_s | Provini |
|---|---|---|
| Standard (no pausa) | 0 | `Rec-G3_std_1`, `Rec-027_std_2` |
| Pausa 10s | 10 | `Rec-022_10s_1`, `Rec-029_10s_2` |
| Pausa 30s | 30 | `Rec-025_30s_1`, `Rec-028_30s_2`, `Rec-031_30s_3`, `Rec-032_30s_4` |
| Pausa 60s | 60 | `Rec-G3_S60_1`, `Rec-026_60s_2` |
| Pausa 90s | 90 | `Rec-024_90s_1`, `Rec-030_90s_2` |

I nomi dei provini corrispondono **esattamente alle chiavi del dizionario `SPECIMENS`**
usato negli script Python del progetto, permettendo join diretti senza nessuna
trasformazione.

---

## Protocollo di Test

Il test è di tipo **scalare progressivo a rottura**:

1. Si applica un peso di 0.5 kg e si osserva se il provino regge.
2. Se regge, si aumenta al gradino successivo: 1.25 → 1.75 → 2.0 → 2.5 → 3.25 → 3.75 kg.
3. Al primo step in cui il provino cede, si registra `ROTTO` e il test si ferma.
4. Gli step successivi alla rottura non vengono eseguiti (celle assenti nei CSV, non NaN).

**Valori speciali:**
- `ok` — il provino ha retto a quel peso
- `ROTTO` — il provino è ceduto a quel peso (valore chiave)
- *(cella vuota / NaN)* — step non eseguito perché il provino era già rotto
- `/` — dato non disponibile (solo `Rec-G3_std_1` a 3.75 kg)

---

## File Presenti

```
mechanical_properties/
├── mechanical_properties_summary.csv   ← una riga per provino, peso di rottura
├── mechanical_properties_long.csv      ← formato tidy, una riga per (provino × step)
├── mechanical_properties_raw.csv       ← tabella originale wide format
└── CONTEXT.md                          ← questo file
```

### `mechanical_properties_summary.csv`
**Il file principale per modelli e visualizzazioni.**

| Colonna | Tipo | Descrizione |
|---|---|---|
| `specimen` | string | Nome provino (chiave di `SPECIMENS`) |
| `pausa_s` | int | Durata pausa tra layer in secondi |
| `peso_rottura_kg` | float | Peso a cui il provino ha ceduto (target del modello) |

```python
summary = pd.read_csv("mechanical_properties_summary.csv")
# feature: summary["pausa_s"]
# target:  summary["peso_rottura_kg"]
```

### `mechanical_properties_long.csv`
**Formato tidy — ideale per analisi statistiche e filtraggio.**
50 righe × 3 colonne. Contiene solo gli step effettivamente testati.

| Colonna | Tipo | Descrizione |
|---|---|---|
| `peso_totale_kg` | float | Peso applicato nello step (0.5 → 3.25 kg) |
| `specimen` | string | Nome provino |
| `result` | string | `"ok"` o `"ROTTO"` |

```python
long = pd.read_csv("mechanical_properties_long.csv")
rotture = long[long["result"] == "ROTTO"]
```

### `mechanical_properties_raw.csv`
**Riproduzione fedele della tabella Excel originale.** 7 righe × 13 colonne.
Una colonna per ogni provino, una riga per ogni step di peso.
Utile per ispezione visiva e verifica dei dati grezzi.

---

## Come Usare Questi Dati nel Progetto

### Join con SPECIMENS

```python
from specimens import SPECIMENS  # o dove è definito il dict
import pandas as pd

summary = pd.read_csv("Dataset_giusti/mechanical_properties/mechanical_properties_summary.csv")

for _, row in summary.iterrows():
    spec = SPECIMENS[row["specimen"]]          # match diretto per chiave
    failure_load = row["peso_rottura_kg"]
    pause = row["pausa_s"]
    layers = spec["layers_front"]              # frame dei layer termici
```

### Correlazione Termica → Meccanica

I CSV di questa cartella sono il **target** da predire a partire dalle feature
termiche estratte in `Dataset_giusti/creazione_dataset/datasets/`. Il workflow
tipico è:

1. Caricare le feature termiche (ROI mean, gradiente, ecc.) per ogni provino.
2. Fare merge su `specimen` con `mechanical_properties_summary.csv`.
3. Addestrare un modello: feature termiche → `peso_rottura_kg`.

```python
thermal = pd.read_csv(".../ROI_wide_3_7_depth_1_2.csv")
mech    = pd.read_csv(".../mechanical_properties_summary.csv")
merged  = thermal.merge(mech, on="specimen")
```

---

## Note

- `Rec-023` è assente da tutti i file: il provino non è valido per errore di registrazione (`valid: False` in `SPECIMENS`).
- I provini STD hanno molti più layer (22–29) rispetto ai provini con pausa (6 layer post-pausa), poiché non hanno interruzioni.
- La colonna `pausa_s` è già numerica e pronta per essere usata come feature continua o categoriale.
