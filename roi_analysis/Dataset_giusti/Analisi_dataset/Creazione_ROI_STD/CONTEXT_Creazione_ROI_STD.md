# CONTEXT — Creazione_ROI_STD
## Creato: 2026-06-03

---

## 1. Obiettivo

Questa cartella contiene lo script e tutti gli output necessari a costruire la **ROI di riferimento STD** (Standard ROI), cioè la rappresentazione termica di una stampa FFF sana e priva di difetti.

La ROI STD è il punto di partenza per qualsiasi analisi successiva: confronto con i provini con pausa, anomaly detection, conformal prediction.

---

## 2. Problema di partenza: il drift termico

I provini STD mostrano un **drift termico sistematico**: la temperatura media della ROI cresce linearmente con il numero di layer, da circa 103 °C (layer 1) a circa 110 °C (layer 3). Questo drift non è un difetto — è la firma fisica di un pezzo che si riscalda progressivamente durante la stampa nominale.

Se non si rimuove il drift, tutti i layer successivi al primo sembrano "più caldi" dello standard, rendendo impossibile un confronto equo tra layer diversi.

**Assunzione chiave**: il drift è un effetto additivo sulla media della distribuzione di temperatura. La variabilità (std dev) è omoschedastica — non cambia con il layer. Questa assunzione è stata verificata visivamente dal grafico `plot_linear_model.png` e sarà testata formalmente in `test_ipotesi/`.

---

## 3. Metodo: regressione lineare + mean centering

### 3.1 Modello

Si fitta una regressione lineare OLS sulla temperatura media ROI di tutti i frame STD core + roi_complete:

```
T_hat(layer) = intercept + slope * layer_index
```

I parametri (intercept, slope, r, p, std_err) sono salvati in `std_roi_data.npz` e `std_roi_summary.txt`.

### 3.2 Correzione drift

Per ogni frame STD, si calcola il drift rispetto al layer 1 e lo si sottrae:

```
drift_k = slope * (layer_index - 1)
T_corrected = T_observed - drift_k
```

Effetto: tutti i frame, indipendentemente dal layer, sono riportati al livello termico del **layer 1** (~103 °C). Questo è il livello di riferimento scelto perché corrisponde alle condizioni di inizio stampa che i provini con pausa tendono a replicare (il raffreddamento causato dalla pausa riporta il pezzo a quella temperatura).

### 3.3 Aggregazione pixel per pixel

Dopo la correzione, tutti i frame STD corretti vengono impilati in una matrice `(N_frames, N_pixels)`. Da questa si calcolano:

| Quantità | Descrizione | File output |
|---|---|---|
| `std_mean_map` | Media per pixel | `plot_std_roi_mean.png` |
| `std_std_map` | Std dev per pixel (ddof=1) | `plot_std_roi_std.png` |
| `std_max_map` | Massimo per pixel | `plot_std_roi_max.png` |
| `std_min_map` | Minimo per pixel | `plot_std_roi_min.png` |

---

## 4. Script: `build_std_roi.py`

### Input

| Parametro | Valore default | Descrizione |
|---|---|---|
| `DATASET_NAME` | `ROI_wide_3_10_depth_1_3` | Nome del dataset CSV da usare |
| `STD_SPECIMENS` | Lista di 9 provini | Solo i provini senza difetti |
| `DATASETS_DIR` | Path Windows locale | Cartella con i CSV |
| `OUTPUT_DIR` | Stessa cartella dello script | Dove salvare tutti gli output |

Cambia solo la sezione `CONFIG` in cima allo script per usare un dataset diverso.

### Output

| File | Tipo | Contenuto |
|---|---|---|
| `std_roi_data.npz` | NumPy archive | `std_mean_map`, `std_std_map`, `std_max_map`, `std_min_map`, `regression_params`, `roi_shape`, `T_hat_layer1` |
| `std_roi_summary.txt` | Testo | Riepilogo leggibile di tutti i parametri e mappe pixel per pixel |
| `plot_linear_model.png` | PNG | Scatter temperatura media per layer + retta di regressione, un colore per provino |
| `plot_std_roi_mean.png` | PNG | Heatmap ROI media drift-corrected con colorbar e valori annotati |
| `plot_std_roi_std.png` | PNG | Heatmap std dev pixel-wise con colorbar |
| `plot_std_roi_max.png` | PNG | Heatmap temperatura massima pixel-wise con colorbar |
| `plot_std_roi_min.png` | PNG | Heatmap temperatura minima pixel-wise con colorbar |

### Come caricare `std_roi_data.npz` in altri script

```python
import numpy as np
from pathlib import Path

STD_DIR = Path(r"C:\Users\tomga\Desktop\Lab Data science\Dataset_giusti\Analisi_dataset\Creazione_ROI_STD")

data = np.load(STD_DIR / "std_roi_data.npz")

std_mean_map = data["std_mean_map"]      # shape (ROI_H, ROI_W)
std_std_map  = data["std_std_map"]
std_max_map  = data["std_max_map"]
std_min_map  = data["std_min_map"]

params = data["regression_params"]       # [intercept, slope, r, p, std_err]
T_hat_layer1 = float(data["T_hat_layer1"])  # reference temperature level
ROI_H, ROI_W = data["roi_shape"]
```

---

## 5. Provini STD usati

| Provino | Pausa |
|---|---|
| Rec-022_10s_1 | 10s |
| Rec-024_90s_1 | 90s |
| Rec-025_30s_1 | 30s |
| Rec-026_60s_2 | 60s |
| Rec-028_30s_2 | 30s |
| Rec-029_10s_2 | 10s |
| Rec-030_90s_2 | 90s |
| Rec-031_30s_3 | 30s |
| Rec-032_30s_4 | 30s |

Nota: i provini G3 (`Rec-G3_*`) **non** sono inclusi perché usano geometria di acquisizione diversa (angolo camera differente) e non sono confrontabili con gli STD sul piano assoluto dei pixel.

---

## 6. Assunzioni e limiti

| Assunzione | Status | Come verificarla |
|---|---|---|
| Drift lineare sulla media | Verificata visivamente (r ≈ 0.96) | `plot_linear_model.png` |
| Omoschedasticità (varianza costante per layer) | Da verificare formalmente | Test di Levene/Bartlett in `test_ipotesi/` |
| Residui i.i.d. dopo correzione | Da verificare | Ljung-Box sui residui per layer |
| Drift solo sulla media, non sulla forma della distribuzione | Assunzione forte, ragionevole con drift lineare piccolo | Analisi distribuzione pixel per layer |

---

## 7. Passo successivo

Una volta creata la ROI STD, il confronto con i provini con pausa avviene in:
`Analisi_dataset/Anomaly_Detection/` (da creare)

Lì si userà `std_roi_data.npz` come reference per costruire la banda di normalità e identificare frame anomali nei provini con pausa.
