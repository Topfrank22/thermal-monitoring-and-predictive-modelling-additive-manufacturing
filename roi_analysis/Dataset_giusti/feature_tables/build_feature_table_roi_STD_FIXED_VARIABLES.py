"""
build_feature_table_roi_STD.py
------------------------------
Variante STD di build_feature_table_roi.py.

Differenze rispetto alla versione base:
  - Processa SOLO i provini standard (STD_SPECIMENS).
  - I provini STD non hanno restart_frame => t0_ref viene calcolato come il
    centro del primo layer annotato (layers_front[0]). Ogni provino parte da
    t=0 in modo indipendente: il confronto tra provini avviene sulla scala
    relativa, non sulla scala assoluta dei frame.
  - Per il fit OLS globale i punti di entrambi i provini sono accumulati con
    l'asse t relativo (t=0 = layer 1 di quel provino). Poiche' dt_layer e'
    identico per tutti i provini STD, la scala temporale e' comparabile.
  - Produce ANCHE la riga per-provino (stessa struttura della versione base)
    in modo che il CSV sia compatibile e confrontabile.
  - Output: feature_tables/output/feature_table_roi_STD_<DATASET_NAME>.csv

Schema colonne (identico a feature_table_roi_*.csv + delta_T + hr_t1..hr_tN):
  specimen | pausa_s | peso_rottura_kg | n_layers |
  T0 | A | alpha | T_inf | t_star |
  heating_rate_t0 | hr_t1..hr_tN |
  delta_T |
  sigma_T0 | sigma_A | sigma_alpha | fit_ok |
  roi_std_spaziale_mean | roi_skew_spaziale_mean |
  bed_temp_C | bed_temp_std_C | T1..TN

Riga aggiuntiva 'STD_OLS_global' con il fit WLS globale su tutti i punti STD.
"""

import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import curve_fit
from scipy.stats import skew

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

# Rate di riscaldamento standard [degC/layer]
RATE_STD_PER_LAYER = 0.240

# Override durata media layer [s]. None = stima automatica dai dati
DT_LAYER_OVERRIDE = None

# Floor per sigma nei pesi WLS
SIGMA_FLOOR = 0.1

# Soglia per marcare fit come non affidabile (sigma/param > FIT_QUALITY_THRESH)
FIT_QUALITY_THRESH = 5.0

ALWAYS_EXCLUDE = {"Rec-023"}
STD_SPECIMENS  = {"Rec-027_std_2", "Rec-G3_std_1"}

# ===========================================================================
#  FINE CONFIG
# ===========================================================================

DATASETS_DIR = CREAZIONE_DS / "datasets"
OUTPUT_DIR   = SCRIPT_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

dataset_path = DATASETS_DIR / f"{DATASET_NAME}.csv"

# ---------------------------------------------------------------------------
# Carica proprieta' meccaniche
# ---------------------------------------------------------------------------
MECH_PATH = SCRIPT_DIR.parent / "mechanical_properties" / "mechanical_properties_summary.csv"
if MECH_PATH.exists():
    df_mech = pd.read_csv(MECH_PATH)[["specimen", "pausa_s", "peso_rottura_kg"]]
    print(f"[INFO] Proprieta' meccaniche caricate: {len(df_mech)} provini")
else:
    df_mech = None
    print(f"[WARN] mechanical_properties_summary.csv non trovato: {MECH_PATH}")

# ---------------------------------------------------------------------------
# Carica bed temperature
# ---------------------------------------------------------------------------
BED_TEMP_PATH = DATASETS_DIR / "bed_temp_end_pause.csv"
if BED_TEMP_PATH.exists():
    df_bed_raw = pd.read_csv(BED_TEMP_PATH)
    if "bed_temp_end_pause" in df_bed_raw.columns and "bed_temp_C" not in df_bed_raw.columns:
        df_bed_raw = df_bed_raw.rename(columns={"bed_temp_end_pause": "bed_temp_C"})
    keep_cols = ["provino", "bed_temp_C"] + (
        ["bed_temp_std_C"] if "bed_temp_std_C" in df_bed_raw.columns else []
    )
    df_bed = df_bed_raw[keep_cols].rename(columns={"provino": "specimen"})
    if "bed_temp_std_C" not in df_bed.columns:
        df_bed["bed_temp_std_C"] = float("nan")
    print(f"[INFO] Bed temperature caricate: {len(df_bed)} provini")
else:
    df_bed = None
    print(f"[WARN] bed_temp_end_pause.csv non trovato — bed_temp sara' NaN")

# ---------------------------------------------------------------------------
# Helper: fit esponenziale WLS
# ---------------------------------------------------------------------------

def fit_heating_wls(t_pts, y_pts, s_pts, rate_std_s):
    """
    Fitta T(t) = T0 + A*(1 - exp(-alpha*t)) con WLS.

    Parametri
    ---------
    t_pts, y_pts, s_pts : array 1-D
        Tempi [s], temperature medie ROI [degC], std intra-layer [degC].
    rate_std_s : float
        Rate di riscaldamento di riferimento [degC/s] passato ESPLICITAMENTE
        (non catturato dalla closure).

    Ritorna dict con parametri + incertezze, oppure success=False.
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
        warnings.warn(f"Fit fallito: {e}")
        return {"success": False}

    T0_fit, A_fit, alpha_fit = popt
    perr = np.sqrt(np.diag(pcov))
    sigma_T0, sigma_A, sigma_alpha = perr

    heating_rate_t0 = A_fit * alpha_fit

    # Heating rate per-layer: hr_tk = A*alpha*exp(-alpha*t_k)
    hr_per_layer = {
        k + 1: float(A_fit * alpha_fit * np.exp(-alpha_fit * t))
        for k, t in enumerate(t_pts)
    }

    # t_star: tempo per raggiungere il rate STD
    t_star = None
    if rate_std_s > 0 and heating_rate_t0 > rate_std_s:
        ratio  = rate_std_s / heating_rate_t0
        t_star = float(-np.log(ratio) / alpha_fit)

    fit_ok = True
    for param, sigma in [(T0_fit, sigma_T0), (A_fit, sigma_A), (alpha_fit, sigma_alpha)]:
        if abs(param) > 1e-9 and sigma / abs(param) > FIT_QUALITY_THRESH:
            fit_ok = False
            break

    return {
        "success":          True,
        "fit_ok":           fit_ok,
        "T0":               float(T0_fit),
        "A":                float(A_fit),
        "alpha":            float(alpha_fit),
        "T_inf":            float(T0_fit + A_fit),
        "t_star":           t_star,
        "sigma_T0":         float(sigma_T0),
        "sigma_A":          float(sigma_A),
        "sigma_alpha":      float(sigma_alpha),
        "heating_rate_t0":  float(heating_rate_t0),
        "hr_per_layer":     hr_per_layer,
    }


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

# Filtra solo provini STD presenti nel dataset
df_STD_only = df[df["specimen"].isin(STD_SPECIMENS)].copy()
if df_STD_only.empty:
    raise ValueError(
        f"Nessun provino STD trovato nel dataset. "
        f"Verifica che {dataset_path} contenga {STD_SPECIMENS}. "
        f"Provini nel file: {df['specimen'].unique().tolist()}"
    )
print(f"[INFO] Provini STD trovati: {sorted(df_STD_only['specimen'].unique())}")

# ---------------------------------------------------------------------------
# Stima dt_layer dai soli provini STD
# Per i provini STD t0 = centro del primo layer (non restart_frame).
# dt_layer = differenza media tra centri di layer consecutivi in secondi.
# ---------------------------------------------------------------------------
if DT_LAYER_OVERRIDE is not None:
    dt_layer = float(DT_LAYER_OVERRIDE)
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
        t_means = []
        for fs, fe in layers_front:
            mask = (df_spec["frame_idx"] >= fs) & (df_spec["frame_idx"] <= fe)
            dl   = df_spec[mask]
            if not dl.empty:
                t_means.append(dl["frame_idx"].mean() / 3.0)
        if len(t_means) >= 2:
            dt_estimates.extend(np.diff(sorted(t_means)).tolist())
    dt_layer = float(np.median(dt_estimates)) if dt_estimates else (1.0 / 3.0)
    print(f"[INFO] dt_layer stimato: {dt_layer:.3f} s")

# rate_std_s calcolato UNA VOLTA e passato esplicitamente alle funzioni (no closure)
rate_std_s = RATE_STD_PER_LAYER / dt_layer
print(f"[INFO] rate_std = {RATE_STD_PER_LAYER} degC/layer => {rate_std_s:.4f} degC/s")

# ---------------------------------------------------------------------------
# Loop provini STD
#
# Ogni provino ha la propria origine temporale t=0 al centro del layer 1.
# I punti global_t_all / global_y_all / global_s_all accumulano i tempi
# RELATIVI (t=0 = layer 1 di quel provino), in modo che il fit OLS globale
# veda tutti i provini allineati sulla stessa scala relativa.
# Questo e' corretto perche' dt_layer e' lo stesso per tutti i provini STD:
# la distanza temporale tra i layer k=1..N e' comparabile tra provini.
# ---------------------------------------------------------------------------
rows = []
global_t_all, global_y_all, global_s_all = [], [], []

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
    t0_ref = dl0["frame_idx"].mean() / 3.0   # secondi assoluti del layer 1

    t_pts, y_pts, s_pts = [], [], []
    T_per_layer   = {}
    std_spaziali  = []
    skew_spaziali = []

    for k_idx, (fs, fe) in enumerate(layers_front):
        mask   = (df_spec["frame_idx"] >= fs) & (df_spec["frame_idx"] <= fe)
        df_lyr = df_spec[mask]
        if df_lyr.empty:
            continue

        # t relativo: layer 1 => 0, layer 2 => dt_layer, ...
        t_rel  = df_lyr["frame_idx"].mean() / 3.0 - t0_ref
        y_mean = df_lyr["roi_mean_C"].mean()
        y_std  = df_lyr["roi_mean_C"].std(ddof=1) if len(df_lyr) > 1 else 0.0

        t_pts.append(t_rel)
        y_pts.append(y_mean)
        s_pts.append(y_std)
        T_per_layer[k_idx + 1] = y_mean

        # Accumulo per fit globale (t relativo => scala comparabile tra provini)
        global_t_all.append(t_rel)
        global_y_all.append(y_mean)
        global_s_all.append(y_std)

        all_pixels = []
        for raw in df_lyr["roi_pixels_raw"].dropna():
            try:
                vals = [float(v) for v in str(raw).split(",") if v.strip()]
                all_pixels.extend(vals)
            except Exception:
                pass
        if len(all_pixels) > 1:
            std_spaziali.append(float(np.std(all_pixels, ddof=1)))
            sk = float(skew(all_pixels)) if len(all_pixels) > 2 else 0.0
            skew_spaziali.append(sk)

    if len(t_pts) < 3:
        print(f"  [SKIP] {specimen}: solo {len(t_pts)} layer validi (min 3)")
        continue

    t_arr = np.array(t_pts)
    y_arr = np.array(y_pts)
    s_arr = np.array(s_pts)

    res = fit_heating_wls(t_arr, y_arr, s_arr, rate_std_s)
    if not res["success"]:
        print(f"  [SKIP] {specimen}: fit fallito")
        continue

    n_layers   = len(t_pts)
    max_layers = len(layers_front)

    # delta_T = T_last - T_first (sui layer validi)
    valid_Ts = [T_per_layer[k] for k in sorted(T_per_layer.keys())]
    delta_T  = float(valid_Ts[-1] - valid_Ts[0]) if len(valid_Ts) >= 2 else None

    row = {
        "specimen":               specimen,
        "n_layers":               n_layers,
        "T0":                     res["T0"],
        "A":                      res["A"],
        "alpha":                  res["alpha"],
        "T_inf":                  res["T_inf"],
        "t_star":                 res["t_star"],
        "heating_rate_t0":        res["heating_rate_t0"],
        "sigma_T0":               res["sigma_T0"],
        "sigma_A":                res["sigma_A"],
        "sigma_alpha":            res["sigma_alpha"],
        "fit_ok":                 res["fit_ok"],
        "delta_T":                delta_T,
        "roi_std_spaziale_mean":  float(np.mean(std_spaziali))  if std_spaziali  else None,
        "roi_skew_spaziale_mean": float(np.mean(skew_spaziali)) if skew_spaziali else None,
    }

    # Heating rate per-layer: hr_t1..hr_tN
    hr_dict = res["hr_per_layer"]
    for k in range(1, n_layers + 1):
        row[f"hr_t{k}"] = hr_dict.get(k, None)

    # T1..TN (temperatura media ROI per layer, dati raw)
    for k in range(1, max_layers + 1):
        row[f"T{k}"] = T_per_layer.get(k, None)

    rows.append(row)
    print(
        f"  [OK per-provino] {specimen:<25}  "
        f"T0={res['T0']:.2f}  A={res['A']:.2f}  alpha={res['alpha']:.5f}  "
        f"n_layers={n_layers}  delta_T={delta_T:.2f}  fit_ok={res['fit_ok']}"
    )

# ---------------------------------------------------------------------------
# Fit WLS globale mescolato (tutti i punti STD insieme, t relativo)
# Ogni provino contribuisce con t=0 al layer 1, t=dt_layer al layer 2, ecc.
# Il fit globale stima i parametri medi della curva di riscaldamento STD.
# ---------------------------------------------------------------------------
if len(global_t_all) >= 3:
    t_g = np.array(global_t_all)
    y_g = np.array(global_y_all)
    s_g = np.array(global_s_all)

    res_global = fit_heating_wls(t_g, y_g, s_g, rate_std_s)
    if res_global["success"]:
        delta_T_global = float(y_g[np.argmax(t_g)] - y_g[np.argmin(t_g)])
        row_global = {
            "specimen":               "STD_OLS_global",
            "n_layers":               len(t_g),
            "T0":                     res_global["T0"],
            "A":                      res_global["A"],
            "alpha":                  res_global["alpha"],
            "T_inf":                  res_global["T_inf"],
            "t_star":                 res_global["t_star"],
            "heating_rate_t0":        res_global["heating_rate_t0"],
            "sigma_T0":               res_global["sigma_T0"],
            "sigma_A":                res_global["sigma_A"],
            "sigma_alpha":            res_global["sigma_alpha"],
            "fit_ok":                 res_global["fit_ok"],
            "delta_T":                delta_T_global,
            "roi_std_spaziale_mean":  None,
            "roi_skew_spaziale_mean": None,
        }
        rows.append(row_global)
        print(
            f"\n  [OK OLS-global] STD_OLS_global  "
            f"T0={res_global['T0']:.2f}  A={res_global['A']:.2f}  "
            f"alpha={res_global['alpha']:.5f}  n_pts={len(t_g)}"
        )
    else:
        print("[WARN] Fit OLS globale fallito.")
else:
    print("[WARN] Troppo pochi punti globali per il fit WLS mescolato.")

if not rows:
    print("[ERROR] Nessuna riga prodotta. Controlla dataset e SPECIMENS.")
    sys.exit(1)

feat_roi = pd.DataFrame(rows)

# ---------------------------------------------------------------------------
# Merge proprieta' meccaniche
# I provini STD non hanno pausa => pausa_s = NaN, peso_rottura_kg presente.
# La riga STD_OLS_global non ha match => tutti NaN (comportamento atteso).
# ---------------------------------------------------------------------------
if df_mech is not None:
    feat_roi = feat_roi.merge(df_mech, on="specimen", how="left")
    print("[INFO] Merge meccanico OK")
else:
    feat_roi["pausa_s"]         = None
    feat_roi["peso_rottura_kg"] = None

# ---------------------------------------------------------------------------
# Merge bed temperature (STD non hanno pausa => NaN atteso)
# ---------------------------------------------------------------------------
if df_bed is not None:
    feat_roi = feat_roi.merge(df_bed, on="specimen", how="left")
    print("[INFO] Merge bed temp OK")
else:
    feat_roi["bed_temp_C"]     = float("nan")
    feat_roi["bed_temp_std_C"] = float("nan")

# ---------------------------------------------------------------------------
# Riordina colonne: identita' -> target -> fit params -> hr per layer ->
#                   delta_T -> qualita' -> std spaziale -> bed -> T raw
# ---------------------------------------------------------------------------
id_cols   = ["specimen", "pausa_s", "peso_rottura_kg"]
fit_cols  = ["n_layers", "T0", "A", "alpha", "T_inf", "t_star",
             "heating_rate_t0"]
hr_cols   = sorted(
    [c for c in feat_roi.columns if c.startswith("hr_t")],
    key=lambda c: int(c[4:])
)
qual_cols = ["delta_T", "sigma_T0", "sigma_A", "sigma_alpha", "fit_ok"]
stat_cols = ["roi_std_spaziale_mean", "roi_skew_spaziale_mean"]
bed_cols  = ["bed_temp_C", "bed_temp_std_C"]
t_cols    = sorted(
    [c for c in feat_roi.columns if c.startswith("T") and c[1:].isdigit()],
    key=lambda c: int(c[1:])
)

ordered = id_cols + fit_cols + hr_cols + qual_cols + stat_cols + bed_cols + t_cols
extra   = [c for c in feat_roi.columns if c not in ordered]
feat_roi = feat_roi[ordered + extra]

# ---------------------------------------------------------------------------
# Salva
# ---------------------------------------------------------------------------
out_path = OUTPUT_DIR / f"feature_table_roi_STD_{DATASET_NAME}.csv"
feat_roi.to_csv(out_path, index=False)
print(f"\n[DONE] Feature table ROI STD salvata: {out_path}")
print(f"  Shape: {feat_roi.shape[0]} righe x {feat_roi.shape[1]} colonne")
print(f"  Colonne: {list(feat_roi.columns)}")
