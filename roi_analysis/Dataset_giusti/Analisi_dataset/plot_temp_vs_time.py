"""
plot_temp_vs_time.py
--------------------
Visualizza la temperatura della ROI nel tempo per tutti i provini,
su un grafico scatter+linea con colori per famiglia di pausa.

L'asse X e' il tempo in secondi (o frame) a partire dal restart_frame
del provino, cosi' che t=0 corrisponde all'istante in cui la stampa riprende.
I gap inter-layer sono visibili come zone vuote sulla linea.

Output:
    Analisi_dataset/<DATASET_NAME>/<METRIC>_Temp_vs_time[_smooth].png
"""

import sys
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from pathlib import Path
from scipy.interpolate import make_interp_spline

# ---------------------------------------------------------------------------
# Aggiunge la cartella creazione_dataset al path per importare frame_selector
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent

# Risale l'albero per trovare la cartella creazione_dataset
CREAZIONE_DS = None
for parent in Path(__file__).resolve().parents:
    candidate = parent / "creazione_dataset"
    if candidate.exists() and candidate.is_dir():
        CREAZIONE_DS = candidate
        break

if CREAZIONE_DS is None:
    raise FileNotFoundError(
        "Cartella 'creazione_dataset' non trovata risalendo dai parent di questo script."
    )

# Trova la cartella che contiene config.py (serve a frame_selector)
CONFIG_DIR = None
for parent in Path(__file__).resolve().parents:
    if (parent / "config.py").exists():
        CONFIG_DIR = parent
        break

if CONFIG_DIR is None:
    raise FileNotFoundError(
        "config.py non trovato risalendo dai parent di questo script."
    )

sys.path.insert(0, str(CONFIG_DIR))
sys.path.insert(0, str(CREAZIONE_DS))

from frame_selector import SPECIMENS


# ===========================================================================
#  CONFIG  -  modifica solo questa sezione
# ===========================================================================

# --- Dataset di input ---
DATASET_NAME = "ROI_wide_2_6_depth_1_2"

# --- Metrica da plottare ---
# "mean"  -> roi_mean_C   -> nome output: Mean_Temp_vs_time
# "max"   -> roi_max_C    -> nome output: Max_Temp_vs_time
# "min"   -> roi_min_C    -> nome output: Min_Temp_vs_time
PLOT_METRIC = "min"

# --- Asse X ---
# "seconds" -> tempo in secondi dal restart (frame / 3.0)
# "frames"  -> indice frame relativo al restart
X_AXIS = "seconds"

# --- Stile linea ---
# "linear" -> linea spezzata che connette i punti nell'ordine
# "smooth" -> spline cubica (curva morbida tra i punti)
LINE_STYLE = "linear"

# --- Filtro validita' ROI ---
# Valore minimo di roi_valid_frac per includere un'osservazione [0.0 - 1.0]
# Usa 1.0 per includere solo osservazioni con la ROI completamente dentro il pezzo
# (equivale a roi_complete == True)
ROI_MIN_VALID_FRAC = 1.0

# --- Provini standard (senza pausa) ---
# True  -> include Rec-027_std_2 e Rec-G3_std_1 (se presenti nel dataset)
# False -> li esclude dal plot
INCLUDE_STD_SPECIMENS = False

# ===========================================================================
#  FINE CONFIG
# ===========================================================================


# ---------------------------------------------------------------------------
# Path derivati automaticamente dalla CONFIG
# ---------------------------------------------------------------------------
DATASETS_DIR = CREAZIONE_DS / "datasets"
OUTPUT_BASE  = SCRIPT_DIR

dataset_path = DATASETS_DIR / f"{DATASET_NAME}.csv"
output_dir   = OUTPUT_BASE / DATASET_NAME
output_dir.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Mapping metrica -> colonna CSV e label
# ---------------------------------------------------------------------------
METRIC_CONFIG = {
    "mean": {"col": "roi_mean_C", "label": "Temperatura media ROI [\u00b0C]", "file_prefix": "Mean_Temp"},
    "max":  {"col": "roi_max_C",  "label": "Temperatura massima ROI [\u00b0C]", "file_prefix": "Max_Temp"},
    "min":  {"col": "roi_min_C",  "label": "Temperatura minima ROI [\u00b0C]",  "file_prefix": "Min_Temp"},
}

if PLOT_METRIC not in METRIC_CONFIG:
    raise ValueError(f"PLOT_METRIC deve essere 'mean', 'max' o 'min'. Ricevuto: '{PLOT_METRIC}'")

metric_col    = METRIC_CONFIG[PLOT_METRIC]["col"]
metric_label  = METRIC_CONFIG[PLOT_METRIC]["label"]
file_prefix   = METRIC_CONFIG[PLOT_METRIC]["file_prefix"]
line_suffix   = "_smooth" if LINE_STYLE == "smooth" else ""
output_fname  = f"{file_prefix}_vs_time{line_suffix}.png"

x_label = "Tempo dal restart [s]" if X_AXIS == "seconds" else "Frame dal restart"


# ---------------------------------------------------------------------------
# Palette colori per famiglia di pausa
# Ogni famiglia ha tonalita' diverse per i replicati (dal piu' scuro al piu' chiaro)
# ---------------------------------------------------------------------------
PAUSE_PALETTES = {
    "10s":  ["#922b21", "#e74c3c"],                           # rossi
    "30s":  ["#1e8449", "#27ae60", "#58d68d", "#a9dfbf"],    # verdi
    "60s":  ["#1a5276", "#2980b9", "#7fb3d3"],               # blu
    "90s":  ["#6c3483", "#8e44ad", "#c39bd3"],               # viola
    "std":  ["#717d7e", "#aab7b8"],                           # grigi (standard)
}

# Provini esclusi sempre
ALWAYS_EXCLUDE = {"Rec-023"}

# Provini standard (senza pausa e senza restart_frame)
STD_SPECIMENS = {"Rec-027_std_2", "Rec-G3_std_1"}


# ---------------------------------------------------------------------------
# Funzione: estrae la durata della pausa dal nome del provino
# ---------------------------------------------------------------------------
def get_pause_group(specimen_name: str) -> str:
    """
    Restituisce la famiglia di pausa (es. '10s', '30s', '60s', '90s', 'std')
    estraendola dal nome del provino.

    Gestisce entrambe le convenzioni di naming:
      - standard:   Rec-026_60s_2  -> pattern "<N>s"
      - gruppo G3:  Rec-G3_S60_1   -> pattern "S<N>" (es. "s60" dopo lower())
    """
    name = specimen_name.lower()

    # Convenzione standard: "10s", "30s", "60s", "90s"
    for pause in ["10s", "30s", "60s", "90s"]:
        if pause in name:
            return pause

    # Convenzione G3: "S60", "S30" ecc. -> regex cerca "s" seguito da cifre
    match = re.search(r's(\d+)', name)
    if match:
        candidate = f"{match.group(1)}s"
        if candidate in PAUSE_PALETTES:
            return candidate

    return "std"


# ---------------------------------------------------------------------------
# Funzione: smooth con spline cubica
# ---------------------------------------------------------------------------
def smooth_line(x: np.ndarray, y: np.ndarray, n_points: int = 300):
    """
    Restituisce (x_smooth, y_smooth) interpolati con spline cubica.
    Se i punti sono meno di 4, restituisce i dati originali.
    """
    if len(x) < 4:
        return x, y
    x_new = np.linspace(x.min(), x.max(), n_points)
    spline = make_interp_spline(x, y, k=3)
    y_new  = spline(x_new)
    return x_new, y_new


# ---------------------------------------------------------------------------
# Carica e filtra il dataset
# ---------------------------------------------------------------------------
print(f"[INFO] Carico dataset: {dataset_path}")
df = pd.read_csv(dataset_path)

# Solo frame core
df = df[df["frame_type"] == "core"].copy()

# Filtro validita' ROI
df = df[df["roi_valid_frac"] >= ROI_MIN_VALID_FRAC].copy()

print(f"[INFO] Righe dopo filtro core + roi_valid_frac >= {ROI_MIN_VALID_FRAC}: {len(df)}")


# ---------------------------------------------------------------------------
# Costruisci il plot
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(14, 7))

all_specimens = df["specimen"].unique()
skipped = []

# Conta i replicati per famiglia di pausa (per assegnare tonalita' diverse)
pause_counters = {k: 0 for k in PAUSE_PALETTES}

legend_handles = []

for specimen in sorted(all_specimens):

    # Esclusioni
    if specimen in ALWAYS_EXCLUDE:
        skipped.append((specimen, "invalido"))
        continue
    if specimen in STD_SPECIMENS and not INCLUDE_STD_SPECIMENS:
        skipped.append((specimen, "standard - escluso da CONFIG"))
        continue

    # Verifica restart_frame
    spec_info = SPECIMENS.get(specimen)
    if spec_info is None:
        skipped.append((specimen, "non trovato in SPECIMENS"))
        continue

    restart_frame = spec_info.get("restart_frame")
    if restart_frame is None:
        skipped.append((specimen, "restart_frame None (standard)"))
        continue

    # Dati del provino
    df_spec = df[df["specimen"] == specimen].copy()
    if df_spec.empty:
        skipped.append((specimen, "nessuna riga dopo i filtri"))
        continue

    # Asse X: tempo relativo al restart_frame
    if X_AXIS == "seconds":
        df_spec["x"] = (df_spec["frame_idx"] - restart_frame) / 3.0
    else:
        df_spec["x"] = df_spec["frame_idx"] - restart_frame

    df_spec = df_spec.sort_values("x")
    x_vals = df_spec["x"].values
    y_vals = df_spec[metric_col].values

    # Colore dalla palette della famiglia
    pause_group = get_pause_group(specimen)
    palette = PAUSE_PALETTES.get(pause_group, PAUSE_PALETTES["std"])
    idx_in_group = pause_counters[pause_group]
    color = palette[idx_in_group % len(palette)]
    pause_counters[pause_group] += 1

    # Plot linea
    if LINE_STYLE == "smooth":
        if len(x_vals) > 1:
            diffs = np.diff(x_vals)
            gap_threshold = np.median(diffs) * 10
            split_indices = np.where(diffs > gap_threshold)[0] + 1
            segments = np.split(np.arange(len(x_vals)), split_indices)
        else:
            segments = [np.arange(len(x_vals))]

        first_segment = True
        for seg in segments:
            if len(seg) == 0:
                continue
            xs, ys = smooth_line(x_vals[seg], y_vals[seg])
            ax.plot(xs, ys, color=color, linewidth=1.2, alpha=0.7,
                    label=specimen if first_segment else "_nolegend_")
            first_segment = False
    else:
        if len(x_vals) > 1:
            diffs = np.diff(x_vals)
            gap_threshold = np.median(diffs) * 10
            x_plot = [x_vals[0]]
            y_plot = [y_vals[0]]
            for i in range(1, len(x_vals)):
                if diffs[i - 1] > gap_threshold:
                    x_plot.append(np.nan)
                    y_plot.append(np.nan)
                x_plot.append(x_vals[i])
                y_plot.append(y_vals[i])
        else:
            x_plot, y_plot = x_vals.tolist(), y_vals.tolist()
        ax.plot(x_plot, y_plot, color=color, linewidth=1.2, alpha=0.7)

    # Scatter punti sopra la linea
    ax.scatter(x_vals, y_vals, color=color, s=18, zorder=5, alpha=0.9)

    # Handle legenda
    handle = mlines.Line2D(
        [], [], color=color, marker='o', markersize=5,
        linewidth=1.5, label=specimen
    )
    legend_handles.append(handle)


# ---------------------------------------------------------------------------
# Stampa a console i provini saltati
# ---------------------------------------------------------------------------
if skipped:
    print("\n[INFO] Provini saltati:")
    for name, reason in skipped:
        print(f"  - {name}: {reason}")


# ---------------------------------------------------------------------------
# Legenda ordinata per gruppo di pausa
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
        sep = mlines.Line2D(
            [], [], color="none", label=f"\u25a0 Pausa {pause}",
            linewidth=0
        )
        ordered_handles.append(sep)
        ordered_handles.extend(group_handles)


# ---------------------------------------------------------------------------
# Formattazione grafico
# ---------------------------------------------------------------------------
title_metric = {"mean": "media", "max": "massima", "min": "minima"}[PLOT_METRIC]
ax.set_title(
    f"Temperatura {title_metric} ROI vs tempo dal restart\n"
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
