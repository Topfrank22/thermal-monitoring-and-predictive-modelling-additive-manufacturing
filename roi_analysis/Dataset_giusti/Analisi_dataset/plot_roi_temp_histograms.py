"""
plot_roi_temp_histograms.py
----------------------------
Distribuzione della temperatura minima della ROI per ogni provino.

Legge i dataset CSV (roi_min_C = pixel piu' freddo della ROI per frame)
e produce istogrammi/KDE per capire il range termico coperto dalla ROI.

NOTA: questi grafici mostrano la distribuzione DENTRO la ROI, NON la
temperatura ambiente. Per la stima della temperatura ambiente dall'intera
immagine usa plot_ambient_temp.py.

Output (Istogrammi_Temperature_ROI/):
  Un grafico per dataset con la distribuzione di roi_min_C per provino.

USO:
    python plot_roi_temp_histograms.py
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from pathlib import Path
from scipy.stats import gaussian_kde

# ---------------------------------------------------------------------------
# Path setup
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


# ===========================================================================
#  CONFIG
# ===========================================================================

DATASETS = [
    "ROI_wide_2_6_depth_1_2",
    "ROI_wide_3_7_depth_1_2",
    "ROI_wide_3_10_depth_1_3",
]

INCLUDE_STD_SPECIMENS = True
BIN_WIDTH = 0.5
X_RANGE = None
SUMMARY_PERCENTILES = [1, 5, 10, 25, 50]

# ===========================================================================

DATASETS_DIR = CREAZIONE_DS / "datasets"
OUTPUT_BASE  = SCRIPT_DIR / "Istogrammi_Temperature_ROI"
OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

ALWAYS_EXCLUDE = {"Rec-023"}
STD_SPECIMENS  = {"Rec-027_std_2", "Rec-G3_std_1"}

PAUSE_COLORS = {
    "10s": "#e74c3c",
    "30s": "#27ae60",
    "60s": "#2980b9",
    "90s": "#8e44ad",
    "std": "#7f8c8d",
}


def get_pause_group(name: str) -> str:
    low = name.lower()
    for pause in ["10s", "30s", "60s", "90s"]:
        if pause in low:
            return pause
    return "std"


for dataset_name in DATASETS:
    dataset_path = DATASETS_DIR / f"{dataset_name}.csv"
    if not dataset_path.exists():
        print(f"[WARN] Dataset non trovato: {dataset_path}  -> skip")
        continue

    print(f"\n{'=' * 60}")
    print(f"  Dataset: {dataset_name}")
    print(f"{'=' * 60}")

    df = pd.read_csv(dataset_path)
    specimens = sorted(df["specimen"].unique())

    specimen_data = {}
    for spec in specimens:
        if spec in ALWAYS_EXCLUDE:
            continue
        if spec in STD_SPECIMENS and not INCLUDE_STD_SPECIMENS:
            continue
        vals = df[df["specimen"] == spec]["roi_min_C"].dropna().values
        if len(vals) > 0:
            specimen_data[spec] = vals

    if not specimen_data:
        print("  [WARN] Nessun provino valido, skip.")
        continue

    print(f"\n  Percentili di roi_min_C per provino:")
    header = f"  {'Provino':<25}" + "".join(f" P{p:02d}" for p in SUMMARY_PERCENTILES)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for spec, vals in specimen_data.items():
        row = f"  {spec:<25}"
        for p in SUMMARY_PERCENTILES:
            row += f"  {np.percentile(vals, p):>5.1f}"
        print(row)

    fig, ax = plt.subplots(figsize=(13, 7))
    legend_handles = []
    all_mins = np.concatenate(list(specimen_data.values()))
    x_lo = (X_RANGE[0] if X_RANGE else np.floor(all_mins.min()) - 1.0)
    x_hi = (X_RANGE[1] if X_RANGE else np.ceil(all_mins.max()) + 1.0)
    bins  = np.arange(x_lo, x_hi + BIN_WIDTH, BIN_WIDTH)
    x_kde = np.linspace(x_lo, x_hi, 600)

    for spec, vals in specimen_data.items():
        pause_group = get_pause_group(spec)
        color = PAUSE_COLORS.get(pause_group, "#555555")
        ax.hist(vals, bins=bins, density=True,
                alpha=0.18, color=color, edgecolor="none")
        if len(vals) > 5:
            try:
                kde = gaussian_kde(vals, bw_method="scott")
                ax.plot(x_kde, kde(x_kde), color=color, linewidth=1.8,
                        alpha=0.85)
            except Exception:
                pass
        handle = mlines.Line2D([], [], color=color, linewidth=2.0, label=spec)
        legend_handles.append(handle)

    ax.set_xlim(x_lo, x_hi)
    ax.set_xlabel("roi_min_C  [\u00b0C]  -  temperatura minima nella ROI per frame", fontsize=11)
    ax.set_ylabel(f"Densita'  (bin = {BIN_WIDTH}\u00b0C)", fontsize=11)
    ax.set_title(
        f"Distribuzione temperatura minima ROI (tutti i frame)\n"
        f"Dataset: {dataset_name}",
        fontsize=12, pad=12
    )
    ax.grid(True, linestyle="--", alpha=0.35, linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(handles=legend_handles, loc="upper right", fontsize=8,
              framealpha=0.85, edgecolor="#cccccc")
    plt.tight_layout()
    out_path = OUTPUT_BASE / f"{dataset_name}_roi_temp_hist.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  [OK] Grafico salvato in: {out_path}")
    plt.close(fig)

print("\nDone. Controlla Istogrammi_Temperature_ROI/")
