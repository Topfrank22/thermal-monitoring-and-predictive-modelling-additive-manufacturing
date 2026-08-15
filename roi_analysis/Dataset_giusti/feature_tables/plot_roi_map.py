"""
plot_roi_map.py
---------------
Genera un singolo PNG che mostra:
  - La griglia della ROI con la label di ogni pixel  (row, col)
  - La posizione relativa della punta dell'ugello rispetto alla ROI
  - L'indice pixel_id (ordine row-major) in ogni cella

La figura e' la stessa per tutti i provini che usano lo stesso DATASET_NAME
(la geometria ROI e' fissa per dataset).

Output:
    Analisi_dataset/<DATASET_NAME>/grafici_extrapolation_pixel/
        ROI_map_<DATASET_NAME>.png
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
from pathlib import Path

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

ANALISI_DIR = SCRIPT_DIR.parent / "Analisi_dataset"

sys.path.insert(0, str(CONFIG_DIR))
sys.path.insert(0, str(CREAZIONE_DS))

# ===========================================================================
#  CONFIG
# ===========================================================================

DATASET_NAME = "ROI_wide_2_6_depth_1_2"  # <--- modifica qui per cambiare ROI

# ===========================================================================
#  FINE CONFIG
# ===========================================================================

DATASETS_DIR = CREAZIONE_DS / "datasets"
dataset_path = DATASETS_DIR / f"{DATASET_NAME}.csv"

OUTPUT_DIR = ANALISI_DIR / DATASET_NAME / "grafici_extrapolation_pixel"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Carica dataset e inferisce geometria ROI
# ---------------------------------------------------------------------------
print(f"[INFO] Carico dataset: {dataset_path}")
df_full = pd.read_csv(dataset_path)

# Usa solo frame core con ROI completa per avere coordinate stabili
df = df_full[
    (df_full["frame_type"] == "core") &
    (df_full["roi_valid_frac"] >= 1.0)
].copy()

if df.empty:
    raise ValueError("Nessuna riga dopo filtro core + roi_valid_frac >= 1.0")

# Leggi coordinate ROI dalla prima riga disponibile
row0 = df.iloc[0]
roi_c0 = int(row0["roi_c0"])
roi_c1 = int(row0["roi_c1"])
roi_r0 = int(row0["roi_r0"])
roi_r1 = int(row0["roi_r1"])

# Range INCLUSIVO: n = fine - inizio + 1
n_cols = roi_c1 - roi_c0 + 1
n_rows = roi_r1 - roi_r0 + 1
n_pixels = n_rows * n_cols

print(f"[INFO] ROI assoluta: col [{roi_c0}, {roi_c1}], row [{roi_r0}, {roi_r1}]")
print(f"[INFO] Shape ROI: {n_rows} righe x {n_cols} colonne = {n_pixels} pixel")

# Posizione media della punta ugello in coordinate ASSOLUTE
nozzle_x_abs = df["tip_x"].median()
nozzle_y_abs = df["tip_y"].median()

# Posizione ugello in coordinate RELATIVE alla ROI
# roi_c0 corrisponde a col 0 nella ROI, roi_r0 corrisponde a row 0
nozzle_col_rel = nozzle_x_abs - roi_c0   # offset in pixel dalla colonna sinistra ROI
nozzle_row_rel = nozzle_y_abs - roi_r0   # offset in pixel dalla riga top ROI

print(f"[INFO] Punta ugello (assoluta): ({nozzle_x_abs:.1f}, {nozzle_y_abs:.1f}) px")
print(f"[INFO] Punta ugello (relativa ROI): col={nozzle_col_rel:.1f}, row={nozzle_row_rel:.1f} px")

# ---------------------------------------------------------------------------
# Colormap: stesso schema degli extrap plots (tab20, indicizzato per pixel_id)
# ---------------------------------------------------------------------------
cmap = cm.get_cmap("tab20", n_pixels)

# ---------------------------------------------------------------------------
# Figura
# ---------------------------------------------------------------------------
# Dimensioni figura: ogni cella e' CELL_SIZE x CELL_SIZE inches
CELL_SIZE = 1.6
fig_w = max(n_cols * CELL_SIZE + 3.0, 7.0)
fig_h = max(n_rows * CELL_SIZE + 3.0, 5.0)

fig, ax = plt.subplots(figsize=(fig_w, fig_h))
ax.set_aspect("equal")

# Sfondo grigio chiaro per la zona fuori ROI
ax.set_facecolor("#f5f5f5")

# ---------------------------------------------------------------------------
# Disegna griglia pixel
# ---------------------------------------------------------------------------
for r in range(n_rows):
    for c in range(n_cols):
        px_id = r * n_cols + c
        color = cmap(px_id)

        # Cella: origine in (c, n_rows - 1 - r) per avere row 0 in alto
        x0 = c
        y0 = n_rows - 1 - r

        rect = mpatches.FancyBboxPatch(
            (x0 + 0.04, y0 + 0.04),
            0.92, 0.92,
            boxstyle="round,pad=0.04",
            linewidth=1.5,
            edgecolor="white",
            facecolor=color,
            alpha=0.85,
            zorder=2,
        )
        ax.add_patch(rect)

        # Label principale: (row, col)
        ax.text(
            x0 + 0.5, y0 + 0.62,
            f"px({r},{c})",
            ha="center", va="center",
            fontsize=11, fontweight="bold",
            color="white",
            zorder=4,
        )
        # Sottolabel: pixel_id
        ax.text(
            x0 + 0.5, y0 + 0.30,
            f"id={px_id}",
            ha="center", va="center",
            fontsize=8.5, color="white",
            alpha=0.90,
            zorder=4,
        )

# ---------------------------------------------------------------------------
# Bordo esterno ROI
# ---------------------------------------------------------------------------
outer = mpatches.FancyBboxPatch(
    (0, 0), n_cols, n_rows,
    boxstyle="square,pad=0",
    linewidth=2.5,
    edgecolor="#333333",
    facecolor="none",
    zorder=5,
)
ax.add_patch(outer)

# ---------------------------------------------------------------------------
# Posizione ugello
# ---------------------------------------------------------------------------
# Converti coordinate relative (pixel) in coordinate asse (colonne/righe)
# Asse x = colonne (0..n_cols), asse y = righe invertite (0..n_rows, 0=bottom=riga n_rows-1)
nozzle_ax_x = nozzle_col_rel          # gia' in unita' colonne
nozzle_ax_y = n_rows - nozzle_row_rel  # inverti: row 0 ROI e' in alto (y = n_rows)

# Disegna freccia + punto
ax.plot(
    nozzle_ax_x, nozzle_ax_y,
    marker="v",          # triangolo verso il basso = ugello che scende
    markersize=18,
    color="#c0392b",
    markeredgecolor="white",
    markeredgewidth=1.5,
    zorder=8,
    label=f"Punta ugello\n(col={nozzle_col_rel:.1f}, row={nozzle_row_rel:.1f} px dalla ROI)",
    clip_on=False,
)

# Linea tratteggiata verticale dall'ugello alla ROI (se ugello e' sopra)
if nozzle_ax_y > n_rows:
    ax.plot(
        [nozzle_ax_x, nozzle_ax_x],
        [n_rows, nozzle_ax_y],
        color="#c0392b", linewidth=1.2, linestyle="--", alpha=0.6, zorder=7, clip_on=False
    )
elif nozzle_ax_y < 0:
    ax.plot(
        [nozzle_ax_x, nozzle_ax_x],
        [nozzle_ax_y, 0],
        color="#c0392b", linewidth=1.2, linestyle="--", alpha=0.6, zorder=7, clip_on=False
    )

# Annotazione
offset_y = 0.35 if nozzle_ax_y >= n_rows - 0.2 else -0.35
ax.annotate(
    f"Ugello\n({nozzle_col_rel:+.1f}, {nozzle_row_rel:+.1f}) px",
    xy=(nozzle_ax_x, nozzle_ax_y),
    xytext=(nozzle_ax_x + 0.5, nozzle_ax_y + (1.0 if nozzle_ax_y >= n_rows else -1.0)),
    fontsize=9,
    color="#c0392b",
    arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.2),
    zorder=9,
    clip_on=False,
)

# ---------------------------------------------------------------------------
# Assi e label
# ---------------------------------------------------------------------------
ax.set_xlim(-0.5, n_cols + 0.5)
ax.set_ylim(-0.5, n_rows + 1.2)  # spazio sopra per l'ugello

# Tick colonne (asse x)
ax.set_xticks(np.arange(n_cols) + 0.5)
ax.set_xticklabels([f"col {roi_c0 + c}\n(roi col {c})" for c in range(n_cols)], fontsize=8)

# Tick righe (asse y, invertito: row 0 ROI = in alto)
ax.set_yticks(np.arange(n_rows) + 0.5)
ax.set_yticklabels([f"row {roi_r0 + (n_rows-1-r)}\n(roi row {n_rows-1-r})" for r in range(n_rows)], fontsize=8)

ax.tick_params(axis="both", length=0)

for spine in ax.spines.values():
    spine.set_visible(False)

# Griglia sottile tra celle
for c in range(n_cols + 1):
    ax.axvline(c, color="#cccccc", linewidth=0.5, zorder=1)
for r in range(n_rows + 1):
    ax.axhline(r, color="#cccccc", linewidth=0.5, zorder=1)

ax.set_title(
    f"Mappa ROI  —  {DATASET_NAME}\n"
    f"Shape: {n_rows} righe x {n_cols} colonne = {n_pixels} pixel  |  "
    f"ROI assoluta: col [{roi_c0},{roi_c1}], row [{roi_r0},{roi_r1}]",
    fontsize=12, pad=14,
)

# Legenda ugello
ax.legend(loc="upper left", fontsize=9, framealpha=0.85, edgecolor="#cccccc")

# Nota convenzione
fig.text(
    0.5, 0.01,
    "Convenzione: pixel_id = row * n_cols + col  (row-major, row 0 = top ROI)",
    ha="center", fontsize=8, color="#888888",
)

plt.tight_layout(rect=[0, 0.03, 1, 1])

out_path = OUTPUT_DIR / f"ROI_map_{DATASET_NAME}.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)

print(f"[OK] ROI map salvata: {out_path}")
