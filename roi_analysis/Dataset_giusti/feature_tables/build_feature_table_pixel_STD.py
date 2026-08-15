"""
build_feature_table_pixel_STD.py
---------------------------------
Feature table A LIVELLO DI PIXEL per i provini STANDARD (senza pausa).

Differenze rispetto a build_feature_table_pixel.py:
  - Processa SOLO i provini standard: Rec-027_std_2 (29 layer), Rec-G3_std_1 (22 layer).
  - Non esiste restart_frame => origine temporale definita come:
        t = 0  al centro del PRIMO layer annotato di CIASCUN provino
      in modo che entrambi i provini partano dallo stesso secondo s=0.
      I frame_idx assoluti NON sono confrontabili tra provini diversi
      (riprese separate, offset di frame arbitrario): confrontare direttamente
      i frame_idx di Rec-027 con quelli di Rec-G3 sarebbe un errore.
      La scala temporale RELATIVA (t_rel = frame_idx.mean()/3 - t0_ref) e'
      quella corretta per il confronto.
  - Il fit esponenziale NON viene fatto pixel per pixel (troppo instabile);
    i parametri T0/A/alpha sono disponibili a livello ROI aggregato
    in build_feature_table_roi_STD.py.
  - Itera su tutti i layer disponibili (22-29) per costruire T1_px..TN_px.
  - Salva ANCHE T1_first6_px..T6_first6_px (prime 6 temperature) per
    comparabilita' col dataset normale (provini con pausa, 6 layer).
  - Calcola feature derivate per pixel:
      delta_T_px  = T_last_px - T1_px   (escursione sull'intera stampa)
      delta_T6_px = T6_px - T1_px       (escursione sui primi 6 layer, confrontabile)
      T_mean_px   = media su tutti i layer
      T_std_px    = std su tutti i layer (variabilita' temporale)
      T_max_px    = temperatura massima raggiunta
      T_min_px    = temperatura minima registrata
  - Una riga per ogni (specimen, pixel_id).

Output: feature_tables/output/feature_table_pixel_STD_<DATASET_NAME>.csv

Schema colonne:
  specimen | n_layers | pixel_row | pixel_col | pixel_id |
  T1_px..TN_px | T1_first6_px..T6_first6_px |
  delta_T_px | delta_T6_px |
  T_mean_px | T_std_px | T_max_px | T_min_px
"""

import re
import sys
import numpy as np
import pandas as pd
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

sys.path.insert(0, str(CONFIG_DIR))
sys.path.insert(0, str(CREAZIONE_DS))

from frame_selector import SPECIMENS  # noqa: E402

# ===========================================================================
#  CONFIG
# ===========================================================================

DATASET_NAME = "ROI_wide_3_10_depth_1_4"  # <--- modifica qui per cambiare ROI config

# Durata media layer [s]. None = stima automatica dai dati STD
DT_LAYER_S = None

ALWAYS_EXCLUDE = {"Rec-023"}
STD_SPECIMENS  = {"Rec-027_std_2", "Rec-G3_std_1"}

# ===========================================================================
#  FINE CONFIG
# ===========================================================================

DATASETS_DIR = CREAZIONE_DS / "datasets"
OUTPUT_DIR   = SCRIPT_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

dataset_path = DATASETS_DIR / f"{DATASET_NAME}.csv"

# Pattern: colonne T<intero>_px (es. T1_px, T22_px) — esclude T_mean_px, T_std_px ecc.
_T_RAW_PX_RE = re.compile(r'^T(\d+)_px$')

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_roi_pixels(raw_str, n_expected=None):
    """Parsa roi_pixels_raw. Adatta lunghezza con padding/troncamento graceful."""
    try:
        vals = np.array([float(v) for v in str(raw_str).split(",") if v.strip()])
        if len(vals) == 0:
            return None
        if n_expected is not None and len(vals) != n_expected:
            if abs(len(vals) - n_expected) > max(2, n_expected * 0.2):
                return None
            if len(vals) < n_expected:
                padded = np.full(n_expected, np.nan)
                padded[:len(vals)] = vals
                return padded
            else:
                return vals[:n_expected]
        return vals
    except Exception:
        return None


def detect_roi_shape(df_sample):
    """Inferisce (n_rows, n_cols) della ROI. Range inclusivo: n = fine - inizio + 1."""
    roi_cols = {"roi_c0", "roi_c1", "roi_r0", "roi_r1"}
    if roi_cols.issubset(df_sample.columns):
        row = df_sample.iloc[0]
        n_cols = int(row["roi_c1"] - row["roi_c0"] + 1)
        n_rows = int(row["roi_r1"] - row["roi_r0"] + 1)
        if n_cols > 0 and n_rows > 0:
            return n_rows, n_cols
    for raw in df_sample["roi_pixels_raw"].dropna():
        vals = parse_roi_pixels(raw)
        if vals is not None:
            return 1, len(vals)
    return None, None


# ---------------------------------------------------------------------------
# Carica dataset
# ---------------------------------------------------------------------------
print(f"[INFO] Carico dataset: {dataset_path}")
df_full = pd.read_csv(dataset_path)

df = df_full[
    (df_full["frame_type"] == "core") &
    (df_full["roi_valid_frac"] >= 1.0)
].copy()
print(f"[INFO] Righe dopo filtro (core, roi_complete): {len(df)}")

# Filtra solo provini STD presenti nel dataset
df_STD_only = df[df["specimen"].isin(STD_SPECIMENS)].copy()
if df_STD_only.empty:
    raise ValueError(
        f"Nessun provino STD trovato nel dataset. "
        f"Verifica che {dataset_path} contenga {STD_SPECIMENS}. "
        f"Provini nel file: {df['specimen'].unique().tolist()}"
    )
print(f"[INFO] Provini STD trovati: {sorted(df_STD_only['specimen'].unique())}")

n_rows_roi, n_cols_roi = detect_roi_shape(df_STD_only)
if n_rows_roi is None:
    raise ValueError("Impossibile inferire shape ROI dai dati STD.")
n_pixels_total = n_rows_roi * n_cols_roi
print(f"[INFO] Shape ROI: {n_rows_roi} x {n_cols_roi} = {n_pixels_total} pixel")

# ---------------------------------------------------------------------------
# Stima dt_layer dai provini STD
#
# Poiche' t0_ref si cancella nella differenza, dt_layer puo' essere stimato
# direttamente come differenza dei centri frame (in secondi assoluti) tra
# layer consecutivi dello stesso provino. Non serve sottrarre t0_ref qui.
# ---------------------------------------------------------------------------
if DT_LAYER_S is not None:
    dt_layer = float(DT_LAYER_S)
    print(f"[INFO] dt_layer (override): {dt_layer:.3f} s")
else:
    dt_estimates = []
    for spec_name in STD_SPECIMENS:
        if spec_name in ALWAYS_EXCLUDE:
            continue
        spec_info    = SPECIMENS.get(spec_name, {})
        layers_front = spec_info.get("layers_front")
        if layers_front is None or len(layers_front) < 2:
            continue
        df_spec = df_STD_only[df_STD_only["specimen"] == spec_name]
        if df_spec.empty:
            continue
        # Centri frame in secondi assoluti (t0_ref si cancella nella diff)
        t_means = []
        for fs, fe in layers_front:
            mask = (df_spec["frame_idx"] >= fs) & (df_spec["frame_idx"] <= fe)
            dl   = df_spec[mask]
            if not dl.empty:
                t_means.append(dl["frame_idx"].mean() / 3.0)
        if len(t_means) >= 2:
            dt_estimates.extend(np.diff(sorted(t_means)).tolist())
    dt_layer = float(np.median(dt_estimates)) if dt_estimates else (1.0 / 3.0)
    print(f"[INFO] dt_layer stimato dai provini STD: {dt_layer:.3f} s")

# ---------------------------------------------------------------------------
# Loop provini STD => loop pixel
#
# SEMANTICA TEMPORALE:
#   t = 0  corrisponde al centro del PRIMO layer annotato di QUEL provino.
#   I frame_idx assoluti di Rec-027 e Rec-G3 non sono confrontabili tra loro
#   (riprese separate, offset arbitrario). Entrambi i provini partono da s=0
#   e avanzano di ~dt_layer secondi per layer => scale comparabili.
#
#   t0_ref = frame_idx.mean() / 3.0  del layer 1 (in secondi assoluti)
#   t_rel  = frame_idx.mean() / 3.0  di ogni layer - t0_ref
#         => layer 1 => t_rel = 0
#            layer 2 => t_rel ~ dt_layer
#            layer k => t_rel ~ (k-1) * dt_layer
# ---------------------------------------------------------------------------
rows = []

for specimen in sorted(STD_SPECIMENS):
    if specimen in ALWAYS_EXCLUDE:
        continue

    spec_info    = SPECIMENS.get(specimen)
    if spec_info is None:
        print(f"  [SKIP] {specimen}: non in SPECIMENS")
        continue

    layers_front = spec_info.get("layers_front")
    if layers_front is None or len(layers_front) == 0:
        print(f"  [SKIP] {specimen}: layers_front mancanti")
        continue

    df_spec = df_STD_only[df_STD_only["specimen"] == specimen]
    if df_spec.empty:
        print(f"  [SKIP] {specimen}: nessuna riga dopo filtri")
        continue

    # --- Origine temporale: centro frame_idx del primo layer => t=0 ---
    fs0, fe0 = layers_front[0]
    mask0    = (df_spec["frame_idx"] >= fs0) & (df_spec["frame_idx"] <= fe0)
    dl0      = df_spec[mask0]
    if dl0.empty:
        print(f"  [SKIP] {specimen}: primo layer vuoto nei dati filtrati")
        continue
    t0_ref = dl0["frame_idx"].mean() / 3.0

    n_layers_spec = len(layers_front)

    T_px_layer = np.full((n_pixels_total, n_layers_spec), np.nan)
    t_layer    = np.full(n_layers_spec, np.nan)

    for k_idx, (fs, fe) in enumerate(layers_front):
        mask   = (df_spec["frame_idx"] >= fs) & (df_spec["frame_idx"] <= fe)
        df_lyr = df_spec[mask]
        if df_lyr.empty:
            continue

        t_layer[k_idx] = df_lyr["frame_idx"].mean() / 3.0 - t0_ref

        pixel_frames = []
        for raw in df_lyr["roi_pixels_raw"].dropna():
            vals = parse_roi_pixels(raw, n_pixels_total)
            if vals is not None:
                pixel_frames.append(vals)

        if not pixel_frames:
            continue

        px_matrix = np.array(pixel_frames)
        T_px_layer[:, k_idx] = np.nanmean(px_matrix, axis=0)

    valid_layers = ~np.isnan(t_layer)
    n_valid = int(valid_layers.sum())
    print(f"  [INFO] {specimen}: {n_valid}/{n_layers_spec} layer validi")

    if n_valid < 1:
        print(f"  [SKIP] {specimen}: nessun layer valido")
        continue

    valid_idxs = np.where(valid_layers)[0]

    for px_id in range(n_pixels_total):
        row_px = px_id // n_cols_roi
        col_px = px_id % n_cols_roi

        row = {
            "specimen":  specimen,
            "n_layers":  n_valid,
            "pixel_row": row_px,
            "pixel_col": col_px,
            "pixel_id":  px_id,
        }

        # T1_px..TN_px su tutti i layer disponibili
        for k_idx in range(n_layers_spec):
            v = T_px_layer[px_id, k_idx]
            row[f"T{k_idx+1}_px"] = float(v) if not np.isnan(v) else None

        # T1_first6_px..T6_first6_px: prime 6 temperature (compatibilita' col dataset con pausa)
        first_6_idxs = valid_idxs[:6]
        for i, k_idx in enumerate(first_6_idxs):
            v = T_px_layer[px_id, k_idx]
            row[f"T{i+1}_first6_px"] = float(v) if not np.isnan(v) else None
        for i in range(len(first_6_idxs), 6):
            row[f"T{i+1}_first6_px"] = None

        # Feature derivate
        t1_val  = T_px_layer[px_id, valid_idxs[0]]  if n_valid >= 1 else np.nan
        t6_val  = T_px_layer[px_id, valid_idxs[5]]  if n_valid >= 6 else np.nan
        t_last  = T_px_layer[px_id, valid_idxs[-1]] if n_valid >= 1 else np.nan

        all_valid_temps = T_px_layer[px_id, valid_idxs]
        all_valid_temps = all_valid_temps[~np.isnan(all_valid_temps)]

        row["delta_T_px"]  = (float(t_last - t1_val)
                               if not (np.isnan(t_last) or np.isnan(t1_val)) else None)
        row["delta_T6_px"] = (float(t6_val - t1_val)
                               if not (np.isnan(t6_val) or np.isnan(t1_val)) else None)
        row["T_mean_px"]   = float(np.mean(all_valid_temps))  if len(all_valid_temps) > 0 else None
        row["T_std_px"]    = (float(np.std(all_valid_temps, ddof=1))
                               if len(all_valid_temps) > 1 else None)
        row["T_max_px"]    = float(np.max(all_valid_temps)) if len(all_valid_temps) > 0 else None
        row["T_min_px"]    = float(np.min(all_valid_temps)) if len(all_valid_temps) > 0 else None

        rows.append(row)

    print(f"  [OK]   {specimen:<35}  {n_valid} layer  {n_pixels_total} pixel/provino")

if not rows:
    print("[ERROR] Nessuna riga prodotta. Controlla dataset e SPECIMENS.")
    sys.exit(1)

feat_px = pd.DataFrame(rows)

# ---------------------------------------------------------------------------
# Riordina colonne: identita' -> geometria pixel -> T raw -> T_first6 -> feature derivate
#
# NOTA: _T_RAW_PX_RE filtra solo T<intero>_px (es. T1_px, T29_px).
# Esclude correttamente T_mean_px, T_std_px, T_max_px, T_min_px.
# ---------------------------------------------------------------------------
id_cols    = ["specimen", "n_layers", "pixel_row", "pixel_col", "pixel_id"]
t_raw_cols = sorted(
    [c for c in feat_px.columns if _T_RAW_PX_RE.match(c)],
    key=lambda c: int(_T_RAW_PX_RE.match(c).group(1))
)
t6_cols    = sorted(
    [c for c in feat_px.columns if "first6" in c],
    key=lambda c: int(c[1:c.index("_first6")])
)
feat_cols  = ["delta_T_px", "delta_T6_px", "T_mean_px", "T_std_px", "T_max_px", "T_min_px"]

feat_px = feat_px[id_cols + t_raw_cols + t6_cols + feat_cols]

# ---------------------------------------------------------------------------
# Salva
# ---------------------------------------------------------------------------
out_path = OUTPUT_DIR / f"feature_table_pixel_STD_{DATASET_NAME}.csv"
feat_px.to_csv(out_path, index=False)
print(f"\n[DONE] Feature table PIXEL STD salvata: {out_path}")
print(f"  Shape: {feat_px.shape[0]} righe x {feat_px.shape[1]} colonne")
print(f"  ({feat_px['specimen'].nunique()} provini x {n_pixels_total} pixel/provino)")
print(f"\n  Colonne: {list(feat_px.columns)}")
