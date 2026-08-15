"""
plot_temp_binned.py
-------------------
Stessa logica di plot_temp_vs_time.py, ma invece di plottare ogni singolo frame
la temperatura viene mediata su finestre temporali di ampiezza BIN_SIZE secondi.
Ogni punto viene posizionato al centro della finestra (es. finestra 10-20s -> x=15s).

Riduce il rumore frame-by-frame e rende piu' leggibile il confronto tra provini.

Output:
    Analisi_dataset/<DATASET_NAME>/BinnedMean_<METRIC>_vs_time.png
"""

import sys
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from pathlib import Path

# ---------------------------------------------------------------------------
# Aggiunge la cartella creazione_dataset al path per importare frame_selector
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent

CREAZIONE_DS = None
for parent in Path(__file__).resolve().parents:
    candidate = parent / "creazione_dataset"
    if candidate.exists() and candidate.is_dir():
        CREAZIONE_DS = candidate
        break

if CREAZIONE_DS is None:
    raise FileNotFoundError("Cartella 'creazione_dataset' non trovata.")

CONFIG_DIR = None
for parent in Path(__file__).resolve().parents:
    if (parent / "config.py").exists():
        CONFIG_DIR = parent
        break

if CONFIG_DIR is None:
    raise FileNotFoundError("config.py non trovato.")

sys.path.insert(0, str(CONFIG_DIR))
sys.path.insert(0, str(CREAZIONE_DS))

from frame_selector import SPECIMENS


# ===========================================================================
#  CONFIG  -  modifica solo questa sezione
# ===========================================================================

# --- Dataset di input ---
DATASET_NAME = "ROI_wide_2_6_depth_1_2"

# --- Metrica da plottare ---
# "mean"  -> media di roi_mean_C dentro ogni bin
# "max"   -> media di roi_max_C  dentro ogni bin
# "min"   -> media di roi_min_C  dentro ogni bin
PLOT_METRIC = "mean"

# --- Ampiezza bin in secondi ---
BIN_SIZE = 2  # es. 2.0 -> finestre da 2s, punto al centro

# --- Asse X ---
# "seconds" -> tempo in secondi dal restart
# "frames"  -> frame relativi al restart
X_AXIS = "seconds"

# --- Filtro validita' ROI ---
ROI_MIN_VALID_FRAC = 1.0

# --- Provini standard ---
INCLUDE_STD_SPECIMENS = False

# ===========================================================================
#  FINE CONFIG
# ===========================================================================


# ---------------------------------------------------------------------------
# Path derivati
# ---------------------------------------------------------------------------
DATASETS_DIR = CREAZIONE_DS / "datasets"
OUTPUT_BASE  = SCRIPT_DIR

dataset_path = DATASETS_DIR / f"{DATASET_NAME}.csv"
output_dir   = OUTPUT_BASE / DATASET_NAME
output_dir.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Mapping metrica
# ---------------------------------------------------------------------------
METRIC_CONFIG = {
    "mean": {"col": "roi_mean_C", "label": "Media bin di roi_mean_C [\u00b0C]", "file_prefix": "BinnedMean_Mean"},
    "max":  {"col": "roi_max_C",  "label": "Media bin di roi_max_C [\u00b0C]",  "file_prefix": "BinnedMean_Max"},
    "min":  {"col": "roi_min_C",  "label": "Media bin di roi_min_C [\u00b0C]",  "file_prefix": "BinnedMean_Min"},
}

if PLOT_METRIC not in METRIC_CONFIG:
    raise ValueError(f"PLOT_METRIC deve essere 'mean', 'max' o 'min'. Ricevuto: '{PLOT_METRIC}'")

metric_col   = METRIC_CONFIG[PLOT_METRIC]["col"]
metric_label = METRIC_CONFIG[PLOT_METRIC]["label"]
file_prefix  = METRIC_CONFIG[PLOT_METRIC]["file_prefix"]
output_fname = f"{file_prefix}_vs_time_bin{int(BIN_SIZE)}s.png"

x_label = "Tempo dal restart [s]" if X_AXIS == "seconds" else "Frame dal restart"


# ---------------------------------------------------------------------------
# Palette colori
# ---------------------------------------------------------------------------
PAUSE_PALETTES = {
    "10s":  ["#922b21", "#e74c3c"],
    "30s":  ["#1e8449", "#27ae60", "#58d68d", "#a9dfbf"],
    "60s":  ["#1a5276", "#2980b9", "#7fb3d3"],
    "90s":  ["#6c3483", "#8e44ad", "#c39bd3"],
    "std":  ["#717d7e", "#aab7b8"],
}

ALWAYS_EXCLUDE = {"Rec-023"}
STD_SPECIMENS  = {"Rec-027_std_2", "Rec-G3_std_1"}  # fix: era Rec-027_2


# ---------------------------------------------------------------------------
# Funzioni
# ---------------------------------------------------------------------------
def get_pause_group(specimen_name: str) -> str:
    name = specimen_name.lower()
    for pause in ["10s", "30s", "60s", "90s"]:
        if pause in name:
            return pause
    match = re.search(r's(\d+)', name)
    if match:
        candidate = f"{match.group(1)}s"
        if candidate in PAUSE_PALETTES:
            return candidate
    return "std"


def bin_timeseries(x: np.ndarray, y: np.ndarray, bin_size: float):
    """
    Raggruppa (x, y) in finestre di ampiezza bin_size.
    Restituisce (x_centers, y_means, y_stds) dove:
      - x_centers: centro di ogni finestra
      - y_means:   media di y dentro la finestra
      - y_stds:    deviazione standard di y dentro la finestra
    Le finestre con meno di 1 punto vengono scartate.
    I gap tra layer (salti > bin_size * 3) vengono preservati come NaN.
    """
    if len(x) == 0:
        return np.array([]), np.array([]), np.array([])

    x_min = np.floor(x.min() / bin_size) * bin_size
    x_max = np.ceil(x.max()  / bin_size) * bin_size
    edges = np.arange(x_min, x_max + bin_size, bin_size)

    x_centers, y_means, y_stds = [], [], []

    for i in range(len(edges) - 1):
        left, right = edges[i], edges[i + 1]
        mask = (x >= left) & (x < right)
        if mask.sum() == 0:
            continue
        center = (left + right) / 2.0
        x_centers.append(center)
        y_means.append(np.mean(y[mask]))
        y_stds.append(np.std(y[mask]))

    return np.array(x_centers), np.array(y_means), np.array(y_stds)


def insert_gaps(x_centers: np.ndarray, y_means: np.ndarray,
                y_stds: np.ndarray, gap_threshold: float):
    """
    Inserisce NaN dove la distanza tra due x_centers consecutivi
    supera gap_threshold, cosi' la linea si interrompe tra i layer.
    """
    if len(x_centers) < 2:
        return x_centers, y_means, y_stds

    x_out, y_out, s_out = [x_centers[0]], [y_means[0]], [y_stds[0]]
    for i in range(1, len(x_centers)):
        if x_centers[i] - x_centers[i - 1] > gap_threshold:
            x_out.append(np.nan)
            y_out.append(np.nan)
            s_out.append(np.nan)
        x_out.append(x_centers[i])
        y_out.append(y_means[i])
        s_out.append(y_stds[i])

    return np.array(x_out), np.array(y_out), np.array(s_out)


# ---------------------------------------------------------------------------
# Carica e filtra dataset
# ---------------------------------------------------------------------------
print(f"[INFO] Carico dataset: {dataset_path}")
df = pd.read_csv(dataset_path)
df = df[df["frame_type"] == "core"].copy()
df = df[df["roi_valid_frac"] >= ROI_MIN_VALID_FRAC].copy()
print(f"[INFO] Righe dopo filtri: {len(df)}")


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(14, 7))

all_specimens  = df["specimen"].unique()
skipped        = []
pause_counters = {k: 0 for k in PAUSE_PALETTES}
legend_handles = []

for specimen in sorted(all_specimens):

    if specimen in ALWAYS_EXCLUDE:
        skipped.append((specimen, "invalido"))
        continue
    if specimen in STD_SPECIMENS and not INCLUDE_STD_SPECIMENS:
        skipped.append((specimen, "standard - escluso da CONFIG"))
        continue

    spec_info = SPECIMENS.get(specimen)
    if spec_info is None:
        skipped.append((specimen, "non trovato in SPECIMENS"))
        continue

    restart_frame = spec_info.get("restart_frame")
    if restart_frame is None:
        skipped.append((specimen, "restart_frame None"))
        continue

    df_spec = df[df["specimen"] == specimen].copy()
    if df_spec.empty:
        skipped.append((specimen, "nessuna riga dopo i filtri"))
        continue

    # Asse X relativo al restart
    if X_AXIS == "seconds":
        df_spec["x"] = (df_spec["frame_idx"] - restart_frame) / 3.0
    else:
        df_spec["x"] = df_spec["frame_idx"] - restart_frame

    df_spec = df_spec.sort_values("x")
    x_raw = df_spec["x"].values
    y_raw = df_spec[metric_col].values

    # Binning
    x_centers, y_means, y_stds = bin_timeseries(x_raw, y_raw, BIN_SIZE)
    if len(x_centers) == 0:
        skipped.append((specimen, "nessun bin valido"))
        continue

    # Inserisci gap tra layer (threshold = 3 * bin_size)
    gap_thr = BIN_SIZE * 3
    x_plot, y_plot, s_plot = insert_gaps(x_centers, y_means, y_stds, gap_thr)

    # Colore
    pause_group  = get_pause_group(specimen)
    palette      = PAUSE_PALETTES.get(pause_group, PAUSE_PALETTES["std"])
    idx_in_group = pause_counters[pause_group]
    color        = palette[idx_in_group % len(palette)]
    pause_counters[pause_group] += 1

    # Linea + scatter (solo sui punti reali, non sui NaN)
    ax.plot(x_plot, y_plot, color=color, linewidth=1.5, alpha=0.8)

    mask_real = ~np.isnan(x_plot)
    ax.scatter(x_plot[mask_real], y_plot[mask_real],
               color=color, s=30, zorder=5, alpha=0.95)

    # Banda di std
    seg_x, seg_y, seg_s = [], [], []
    for xi, yi, si in zip(x_plot, y_plot, s_plot):
        if np.isnan(xi):
            if len(seg_x) > 1:
                ax.fill_between(seg_x,
                                np.array(seg_y) - np.array(seg_s),
                                np.array(seg_y) + np.array(seg_s),
                                color=color, alpha=0.12, linewidth=0)
            seg_x, seg_y, seg_s = [], [], []
        else:
            seg_x.append(xi)
            seg_y.append(yi)
            seg_s.append(si)
    if len(seg_x) > 1:
        ax.fill_between(seg_x,
                        np.array(seg_y) - np.array(seg_s),
                        np.array(seg_y) + np.array(seg_s),
                        color=color, alpha=0.12, linewidth=0)

    handle = mlines.Line2D(
        [], [], color=color, marker='o', markersize=5,
        linewidth=1.5, label=specimen
    )
    legend_handles.append(handle)


# ---------------------------------------------------------------------------
# Log provini saltati
# ---------------------------------------------------------------------------
if skipped:
    print("\n[INFO] Provini saltati:")
    for name, reason in skipped:
        print(f"  - {name}: {reason}")


# ---------------------------------------------------------------------------
# Legenda ordinata per gruppo
# ---------------------------------------------------------------------------
ordered_handles = []
for pause in ["10s", "30s", "60s", "90s", "std"]:
    if pause == "std" and not INCLUDE_STD_SPECIMENS:
        continue
    group_handles = [
        h for h in legend_handles
        if get_pause_group(h.get_label()) == pause
    ]
    if group_handles:
        sep = mlines.Line2D([], [], color="none",
                            label=f"\u25a0 Pausa {pause}", linewidth=0)
        ordered_handles.append(sep)
        ordered_handles.extend(group_handles)


# ---------------------------------------------------------------------------
# Formattazione
# ---------------------------------------------------------------------------
title_metric = {"mean": "media", "max": "massima", "min": "minima"}[PLOT_METRIC]
ax.set_title(
    f"Temperatura {title_metric} ROI (media per bin {BIN_SIZE}s) vs tempo dal restart\n"
    f"Dataset: {DATASET_NAME}  |  Filtro ROI: valid_frac \u2265 {ROI_MIN_VALID_FRAC}",
    fontsize=12, pad=12
)
ax.set_xlabel(x_label, fontsize=11)
ax.set_ylabel(metric_label, fontsize=11)
ax.grid(True, linestyle="--", alpha=0.4, linewidth=0.7)
ax.spines[["top", "right"]].set_visible(False)

ax.legend(
    handles=ordered_handles,
    loc="upper right",
    fontsize=8,
    framealpha=0.85,
    edgecolor="#cccccc",
    ncol=1,
)

plt.tight_layout()


# ---------------------------------------------------------------------------
# Salvataggio
# ---------------------------------------------------------------------------
out_path = output_dir / output_fname
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\n[OK] Grafico salvato in: {out_path}")
plt.show()
