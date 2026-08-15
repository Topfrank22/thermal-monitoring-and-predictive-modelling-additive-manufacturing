"""
build_feature_table_pixel.py
-----------------------------
Costruisce la feature table A LIVELLO DI PIXEL (una riga per pixel per provino).

Per ogni (specimen, pixel_row, pixel_col):
  - T[k] = media temporale del pixel su tutti i frame core del layer k
  - Fitta T(t) = T0 + A*(1 - exp(-alpha*t)) con WLS
  - Estrae: T0, A, alpha, T_inf, t_star, heating_rate_t0, sigma_*, fit_ok
  - Aggiunge: T1_px..TN_px (temperature raw del pixel per layer)
  - Feature derivate per pixel:
      delta_T_px  = T_last_px - T1_px
      T_mean_px   = media su tutti i layer
      T_std_px    = std su tutti i layer (variabilita' temporale)
      T_max_px    = temperatura massima
      T_min_px    = temperatura minima
  - Merge con mechanical_properties_summary.csv => pausa_s, peso_rottura_kg (target)
  - Merge con bed_temp_end_pause.csv => bed_temp_C, bed_temp_std_C
    (costante per provino, replicata su ogni pixel dello stesso provino)

NOTA: con 6 layer e 3 parametri liberi il fit e' poco vincolato.
      Usare fit_ok e sigma_* per filtrare i risultati.
      Per analisi aggregate usare build_feature_table_roi.py (fit sulla media ROI).

Output: feature_tables/output/feature_table_pixel_<DATASET_NAME>.csv

Schema colonne:
  specimen | pausa_s | peso_rottura_kg | pixel_row | pixel_col | pixel_id |
  T1_px..TN_px |
  delta_T_px | T_mean_px | T_std_px | T_max_px | T_min_px |
  fit_ok | fit_error | T0 | A | alpha | T_inf |
  t_star | heating_rate_t0 | sigma_T0 | sigma_A | sigma_alpha |
  bed_temp_C | bed_temp_std_C
"""

import re
import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import curve_fit

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

DATASET_NAME = "ROI_wide_3_10_depth_1_4"  # <--- modifica qui per cambiare ROI

RATE_STD_PER_LAYER = 0.240
DT_LAYER_OVERRIDE  = None
SIGMA_FLOOR        = 0.1
FIT_QUALITY_THRESH = 5.0    # sigma/param > soglia => fit_ok=False

ALWAYS_EXCLUDE = {"Rec-023"}
STD_SPECIMENS  = {"Rec-027_std_2", "Rec-G3_std_1"}
INCLUDE_STD    = False

# ===========================================================================
#  FINE CONFIG
# ===========================================================================

DATASETS_DIR = CREAZIONE_DS / "datasets"
BED_CSV      = DATASETS_DIR / "bed_temp_end_pause.csv"
OUTPUT_DIR   = SCRIPT_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

dataset_path = DATASETS_DIR / f"{DATASET_NAME}.csv"

# ---------------------------------------------------------------------------
# Carica proprieta' meccaniche (target del modello)
# ---------------------------------------------------------------------------
MECH_PATH = SCRIPT_DIR.parent / "mechanical_properties" / "mechanical_properties_summary.csv"
if MECH_PATH.exists():
    df_mech = pd.read_csv(MECH_PATH)[["specimen", "pausa_s", "peso_rottura_kg"]]
    print(f"[INFO] Proprieta' meccaniche caricate: {len(df_mech)} provini")
else:
    df_mech = None
    print(f"[WARN] mechanical_properties_summary.csv non trovato: {MECH_PATH}")

# ---------------------------------------------------------------------------
# Carica bed temperature con retrocompatibilita'
# ---------------------------------------------------------------------------
def _load_bed_csv(bed_csv: Path) -> pd.DataFrame:
    empty = pd.DataFrame(columns=["specimen", "bed_temp_C", "bed_temp_std_C"])
    if not bed_csv.exists():
        print(f"[WARN] bed CSV non trovato: {bed_csv} — bed_temp sara' NaN per tutti.")
        return empty
    df = pd.read_csv(bed_csv)
    prov_col = next(
        (c for c in df.columns if c.lower() in ("specimen", "provino", "sample", "name")),
        None
    )
    if prov_col is None:
        print("[WARN] Colonna provino non trovata nel bed CSV — bed_temp sara' NaN per tutti.")
        return empty
    df = df.rename(columns={prov_col: "specimen"})
    if "bed_temp_end_pause" in df.columns and "bed_temp_C" not in df.columns:
        df = df.rename(columns={"bed_temp_end_pause": "bed_temp_C"})
        print("[INFO] bed CSV formato vecchio: 'bed_temp_end_pause' rinominata in 'bed_temp_C'.")
    if "bed_temp_std_C" not in df.columns:
        df["bed_temp_std_C"] = float("nan")
        print("[WARN] 'bed_temp_std_C' assente nel bed CSV — sara' NaN. "
              "Riesegui bed_temp_at_pause_end.py per aggiornare.")
    print(f"[INFO] Bed temperature caricate: {len(df)} provini")
    return df[["specimen", "bed_temp_C", "bed_temp_std_C"]]

df_bed = _load_bed_csv(BED_CSV)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Pattern: colonne T<intero>_px (es. T1_px, T12_px) — esclude T_mean_px, T_std_px ecc.
_T_RAW_PX_RE = re.compile(r'^T(\d+)_px$')


def fit_heating_wls_px(t_pts, y_pts, s_pts, rate_std_s):
    """
    Fit WLS per un singolo pixel.

    Parametri
    ---------
    rate_std_s : float
        Rate di riscaldamento di riferimento [degC/s] passato ESPLICITAMENTE
        (non catturato dalla closure) per evitare dipendenze da variabile globale.

    Ritorna dict con parametri o success=False.
    """
    sigma_w = np.where(s_pts > 0, s_pts, SIGMA_FLOOR)

    def model(t, T0, A, alpha):
        return T0 + A * (1.0 - np.exp(-alpha * t))

    T0_0    = float(y_pts[np.argmin(t_pts)])
    A_0     = max(float(np.max(y_pts) - np.min(y_pts)), 1.0)
    alpha_0 = 0.05

    try:
        popt, pcov = curve_fit(
            model, t_pts, y_pts,
            p0=[T0_0, A_0, alpha_0],
            sigma=sigma_w,
            absolute_sigma=True,
            bounds=([-np.inf, 0.0, 1e-6], [np.inf, np.inf, np.inf]),
            maxfev=20000,
        )
    except Exception as e:
        return {"success": False, "error": str(e)}

    T0_fit, A_fit, alpha_fit = popt
    perr = np.sqrt(np.diag(pcov))

    heating_rate_t0 = A_fit * alpha_fit

    t_star = None
    if rate_std_s > 0 and heating_rate_t0 > rate_std_s:
        ratio = rate_std_s / heating_rate_t0
        if 0 < ratio < 1:
            t_star = float(-np.log(ratio) / alpha_fit)

    fit_ok = True
    for param, sigma in zip(popt, perr):
        if abs(param) > 1e-9 and sigma / abs(param) > FIT_QUALITY_THRESH:
            fit_ok = False
            break

    return {
        "success":         True,
        "fit_ok":          fit_ok,
        "T0":              float(T0_fit),
        "A":               float(A_fit),
        "alpha":           float(alpha_fit),
        "T_inf":           float(T0_fit + A_fit),
        "t_star":          t_star,
        "heating_rate_t0": float(heating_rate_t0),
        "sigma_T0":        float(perr[0]),
        "sigma_A":         float(perr[1]),
        "sigma_alpha":     float(perr[2]),
    }


def parse_roi_pixels(raw_str, n_expected=None):
    """
    Parsa roi_pixels_raw. Se n_expected e' fornito, adatta la lunghezza
    con padding/troncamento graceful invece di rigettare.
    """
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
    """
    Inferisce (n_rows, n_cols) della ROI.
    Range INCLUSIVO: n = fine - inizio + 1.
    """
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
print(f"[INFO] Righe dopo filtro: {len(df)}")

n_rows_roi, n_cols_roi = detect_roi_shape(df)
if n_rows_roi is None:
    raise ValueError("Impossibile inferire shape ROI.")
n_pixels_total = n_rows_roi * n_cols_roi
print(f"[INFO] Shape ROI: {n_rows_roi} x {n_cols_roi} = {n_pixels_total} pixel")

# ---------------------------------------------------------------------------
# Stima dt_layer
# ---------------------------------------------------------------------------
if DT_LAYER_OVERRIDE is not None:
    dt_layer = float(DT_LAYER_OVERRIDE)
    print(f"[INFO] dt_layer (override): {dt_layer:.3f} s")
else:
    dt_estimates = []
    for spec_name, spec_info in SPECIMENS.items():
        if spec_name in ALWAYS_EXCLUDE:
            continue
        restart_frame = spec_info.get("restart_frame")
        layers_front  = spec_info.get("layers_front")
        if restart_frame is None or layers_front is None or len(layers_front) < 2:
            continue
        df_spec = df[df["specimen"] == spec_name]
        if df_spec.empty:
            continue
        t_means = []
        for fs, fe in layers_front:
            mask = (df_spec["frame_idx"] >= fs) & (df_spec["frame_idx"] <= fe)
            dl = df_spec[mask]
            if not dl.empty:
                t_means.append((dl["frame_idx"].mean() - restart_frame) / 3.0)
        if len(t_means) >= 2:
            dt_estimates.extend(np.diff(sorted(t_means)).tolist())
    dt_layer = float(np.median(dt_estimates)) if dt_estimates else (1.0 / 3.0)
    print(f"[INFO] dt_layer stimato: {dt_layer:.3f} s")

# rate_std_s calcolato UNA VOLTA e passato esplicitamente alle funzioni (no closure)
rate_std_s = RATE_STD_PER_LAYER / dt_layer
print(f"[INFO] rate_std = {RATE_STD_PER_LAYER} degC/layer => {rate_std_s:.4f} degC/s")

# ---------------------------------------------------------------------------
# Loop provini => loop pixel
# ---------------------------------------------------------------------------
rows = []

for specimen in sorted(df["specimen"].unique()):
    if specimen in ALWAYS_EXCLUDE:
        continue
    if specimen in STD_SPECIMENS and not INCLUDE_STD:
        continue

    spec_info = SPECIMENS.get(specimen)
    if spec_info is None:
        continue

    restart_frame = spec_info.get("restart_frame")
    layers_front  = spec_info.get("layers_front")
    if restart_frame is None or layers_front is None:
        continue

    df_spec = df[df["specimen"] == specimen]
    if df_spec.empty:
        continue

    n_layers_spec = len(layers_front)

    T_px_layer = np.full((n_pixels_total, n_layers_spec), np.nan)
    S_px_layer = np.full((n_pixels_total, n_layers_spec), np.nan)
    t_layer    = np.full(n_layers_spec, np.nan)

    for k_idx, (fs, fe) in enumerate(layers_front):
        mask   = (df_spec["frame_idx"] >= fs) & (df_spec["frame_idx"] <= fe)
        df_lyr = df_spec[mask]
        if df_lyr.empty:
            continue

        t_layer[k_idx] = (df_lyr["frame_idx"].mean() - restart_frame) / 3.0

        pixel_frames = []
        for raw in df_lyr["roi_pixels_raw"].dropna():
            vals = parse_roi_pixels(raw, n_pixels_total)
            if vals is not None:
                pixel_frames.append(vals)

        if not pixel_frames:
            continue

        px_matrix = np.array(pixel_frames)
        T_px_layer[:, k_idx] = np.nanmean(px_matrix, axis=0)
        S_px_layer[:, k_idx] = (
            np.nanstd(px_matrix, axis=0, ddof=1)
            if px_matrix.shape[0] > 1 else 0.0
        )

    valid_layers = ~np.isnan(t_layer)
    if valid_layers.sum() < 3:
        print(f"  [SKIP] {specimen}: solo {valid_layers.sum()} layer validi")
        continue

    t_pts_spec   = t_layer[valid_layers]
    valid_idxs   = np.where(valid_layers)[0]

    # Bed temp per questo provino (costante su tutti i pixel)
    bed_row = df_bed[df_bed["specimen"] == specimen]
    if len(bed_row) == 1:
        bed_temp_C     = float(bed_row["bed_temp_C"].iloc[0])
        bed_temp_std_C = float(bed_row["bed_temp_std_C"].iloc[0])
    else:
        bed_temp_C     = float("nan")
        bed_temp_std_C = float("nan")

    n_ok = 0
    n_fail = 0

    for px_id in range(n_pixels_total):
        row_px = px_id // n_cols_roi
        col_px = px_id % n_cols_roi

        y_px = T_px_layer[px_id, valid_layers]
        s_px = S_px_layer[px_id, valid_layers]

        row = {
            "specimen":  specimen,
            "pixel_row": row_px,
            "pixel_col": col_px,
            "pixel_id":  px_id,
        }

        # T raw per layer
        for k_idx in range(n_layers_spec):
            v = T_px_layer[px_id, k_idx]
            row[f"T{k_idx+1}_px"] = float(v) if not np.isnan(v) else None

        # --- Feature derivate per pixel ---
        valid_temps = T_px_layer[px_id, valid_idxs]
        valid_temps_clean = valid_temps[~np.isnan(valid_temps)]

        t1_val   = T_px_layer[px_id, valid_idxs[0]]  if len(valid_idxs) >= 1 else np.nan
        t_last   = T_px_layer[px_id, valid_idxs[-1]] if len(valid_idxs) >= 1 else np.nan

        row["delta_T_px"] = (
            float(t_last - t1_val)
            if not (np.isnan(t_last) or np.isnan(t1_val)) else None
        )
        row["T_mean_px"] = float(np.mean(valid_temps_clean))  if len(valid_temps_clean) > 0 else None
        row["T_std_px"]  = (
            float(np.std(valid_temps_clean, ddof=1))
            if len(valid_temps_clean) > 1 else None
        )
        row["T_max_px"]  = float(np.max(valid_temps_clean)) if len(valid_temps_clean) > 0 else None
        row["T_min_px"]  = float(np.min(valid_temps_clean)) if len(valid_temps_clean) > 0 else None

        # --- Fit esponenziale (mantenuto per analisi esplorative, usare fit_ok per filtrare) ---
        if np.all(np.isnan(y_px)):
            row.update({
                "fit_ok": False, "fit_error": "tutti NaN",
                "T0": None, "A": None, "alpha": None,
                "T_inf": None, "t_star": None, "heating_rate_t0": None,
                "sigma_T0": None, "sigma_A": None, "sigma_alpha": None,
            })
            n_fail += 1
        else:
            res = fit_heating_wls_px(t_pts_spec, y_px, s_px, rate_std_s)
            if res["success"]:
                row.update({
                    "fit_ok":          res["fit_ok"],
                    "fit_error":       None,
                    "T0":              res["T0"],
                    "A":               res["A"],
                    "alpha":           res["alpha"],
                    "T_inf":           res["T_inf"],
                    "t_star":          res["t_star"],
                    "heating_rate_t0": res["heating_rate_t0"],
                    "sigma_T0":        res["sigma_T0"],
                    "sigma_A":         res["sigma_A"],
                    "sigma_alpha":     res["sigma_alpha"],
                })
                n_ok += 1
            else:
                row.update({
                    "fit_ok": False, "fit_error": res.get("error", "unknown"),
                    "T0": None, "A": None, "alpha": None,
                    "T_inf": None, "t_star": None, "heating_rate_t0": None,
                    "sigma_T0": None, "sigma_A": None, "sigma_alpha": None,
                })
                n_fail += 1

        # Bed temp: costante per provino, replicata su ogni pixel
        row["bed_temp_C"]     = bed_temp_C     if not np.isnan(bed_temp_C)     else None
        row["bed_temp_std_C"] = bed_temp_std_C if not np.isnan(bed_temp_std_C) else None

        rows.append(row)

    bed_str = f"{bed_temp_C:.1f} C" if not np.isnan(bed_temp_C) else "NaN"
    print(
        f"  [OK]   {specimen:<35}  "
        f"pixel fit OK: {n_ok}/{n_pixels_total}  fail: {n_fail}  bed_temp={bed_str}"
    )

# ---------------------------------------------------------------------------
# Salva DataFrame grezzo
# ---------------------------------------------------------------------------
feat_px = pd.DataFrame(rows)

# ---------------------------------------------------------------------------
# Merge proprieta' meccaniche
# ---------------------------------------------------------------------------
if df_mech is not None:
    feat_px = feat_px.merge(df_mech, on="specimen", how="left")
    n_miss = feat_px["peso_rottura_kg"].isna().sum()
    if n_miss > 0:
        miss_specs = feat_px[feat_px["peso_rottura_kg"].isna()]["specimen"].unique()
        print(f"[WARN] {n_miss} righe senza peso_rottura_kg (provini: {list(miss_specs)})")
    else:
        n_provini = feat_px["specimen"].nunique()
        print(f"[INFO] Merge meccanico OK: peso_rottura_kg disponibile per tutti i {n_provini} provini")
else:
    feat_px["pausa_s"]         = None
    feat_px["peso_rottura_kg"] = None

# ---------------------------------------------------------------------------
# Riordina colonne:
# identita' -> target -> coordinate pixel -> T raw -> feature derivate ->
# fit params -> qualita' -> bed temp
#
# NOTA: _T_RAW_PX_RE filtra solo T<intero>_px (es. T1_px, T12_px).
# Esclude correttamente T_mean_px, T_std_px, T_max_px, T_min_px
# che contengono lettere tra 'T' e '_px'.
# ---------------------------------------------------------------------------
id_cols    = ["specimen", "pausa_s", "peso_rottura_kg", "pixel_row", "pixel_col", "pixel_id"]
t_raw_cols = sorted(
    [c for c in feat_px.columns if _T_RAW_PX_RE.match(c)],
    key=lambda c: int(_T_RAW_PX_RE.match(c).group(1))
)
deriv_cols = ["delta_T_px", "T_mean_px", "T_std_px", "T_max_px", "T_min_px"]
fit_cols   = ["fit_ok", "fit_error", "T0", "A", "alpha", "T_inf",
              "t_star", "heating_rate_t0",
              "sigma_T0", "sigma_A", "sigma_alpha"]
end_cols   = ["bed_temp_C", "bed_temp_std_C"]

ordered = id_cols + t_raw_cols + deriv_cols + fit_cols + end_cols
extra   = [c for c in feat_px.columns if c not in ordered]
feat_px = feat_px[ordered + extra]

# ---------------------------------------------------------------------------
# Salva
# ---------------------------------------------------------------------------
out_path = OUTPUT_DIR / f"feature_table_pixel_{DATASET_NAME}.csv"
feat_px.to_csv(out_path, index=False)
print(f"\n[DONE] Feature table pixel salvata: {out_path}")
print(f"  Shape: {feat_px.shape[0]} righe x {feat_px.shape[1]} colonne")
print(f"  ({feat_px['specimen'].nunique()} provini x {n_pixels_total} pixel/provino)")
print(f"  Fit OK: {feat_px['fit_ok'].sum()}/{len(feat_px)}")
bed_ok = feat_px["bed_temp_C"].notna().sum()
print(f"  Bed temp OK: {bed_ok}/{len(feat_px)} righe")
print(f"\n  Colonne: {list(feat_px.columns)}")
