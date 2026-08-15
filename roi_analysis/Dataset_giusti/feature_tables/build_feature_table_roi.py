#!/usr/bin/env python3
"""
build_feature_table_roi.py
--------------------------
Costruisce la feature table A LIVELLO DI PROVINO (fit sulla media ROI).

Per ogni provino:
- Per ogni layer k: aggrega tutti i frame core con roi_complete=True
=> T[k] = media di roi_mean_C (scalare per layer)
- Fitta il modello: T(t) = T0 + A*(1 - exp(-alpha*t)) con WLS
- Estrae: T0, A, alpha, T_inf, t_star, sigma_*, heating_rate_t0
- Heating rate per-layer: hr_t1..hr_tk = A*alpha*exp(-alpha*t_k)
- delta_T = T6 - T1 (o ultimo layer - primo se <6 layer)
- Aggiunge: T1..T6 (temperature raw per layer), roi_std_mean, roi_skew_mean
- Merge con mechanical_properties_summary.csv => peso_rottura_kg (target)
- Merge con bed_temp_end_pause.csv => bed_temp_C, bed_temp_std_C
(NaN per provini std che non hanno pausa)

Output: feature_tables/output/feature_table_roi_.csv

Schema colonne:
specimen | pausa_s | peso_rottura_kg | n_layers |
T0 | A | alpha | T_inf | t_star |
heating_rate_t0 | hr_t1..hr_tN |
delta_T |
sigma_T0 | sigma_A | sigma_alpha | fit_ok |
roi_std_spaziale_mean | roi_skew_spaziale_mean |
bed_temp_C | bed_temp_std_C | T1..TN
"""

import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import curve_fit
from scipy.stats import skew

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

DATASET_NAME = 'ROI_x3-12_y2-5_offNonG31_offG30'

RATE_STD_PER_LAYER = 0.240
DT_LAYER_OVERRIDE = None
SIGMA_FLOOR = 0.1
FIT_QUALITY_THRESH = 5.0

ALWAYS_EXCLUDE = {"Rec-023"}
STD_SPECIMENS = {"Rec-027_std_2", "Rec-G3_std_1"}
INCLUDE_STD = False

DATASETS_DIR = CREAZIONE_DS / "datasets"
OUTPUT_DIR = SCRIPT_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

dataset_path = DATASETS_DIR / f"{DATASET_NAME}.csv"

MECH_PATH = SCRIPT_DIR.parent / "mechanical_properties" / "mechanical_properties_summary.csv"
if MECH_PATH.exists():
    df_mech = pd.read_csv(MECH_PATH)[["specimen", "pausa_s", "peso_rottura_kg"]]
    print(f"[INFO] Proprieta' meccaniche caricate: {len(df_mech)} provini")
else:
    df_mech = None
    print(f"[WARN] mechanical_properties_summary.csv non trovato: {MECH_PATH}")

BED_TEMP_PATH = DATASETS_DIR / "bed_temp_end_pause.csv"
if BED_TEMP_PATH.exists():
    df_bed_raw = pd.read_csv(BED_TEMP_PATH)
    if "bed_temp_end_pause" in df_bed_raw.columns and "bed_temp_C" not in df_bed_raw.columns:
        df_bed_raw = df_bed_raw.rename(columns={"bed_temp_end_pause": "bed_temp_C"})
    keep_cols = ["provino", "bed_temp_C"] + (["bed_temp_std_C"] if "bed_temp_std_C" in df_bed_raw.columns else [])
    df_bed = df_bed_raw[keep_cols].rename(columns={"provino": "specimen"})
    if "bed_temp_std_C" not in df_bed.columns:
        df_bed["bed_temp_std_C"] = float("nan")
    print(f"[INFO] Bed temperature caricate: {len(df_bed)} provini")
else:
    df_bed = None
    print(f"[WARN] bed_temp_end_pause.csv non trovato: {BED_TEMP_PATH} — bed_temp sara' NaN")

def fit_heating_wls(t_pts, y_pts, s_pts, rate_std_s):
    sigma_w = np.where(s_pts > 0, s_pts, SIGMA_FLOOR)

    def model(t, T0, A, alpha):
        return T0 + A * (1.0 - np.exp(-alpha * t))

    T0_0 = float(y_pts[np.argmin(t_pts)])
    A_0 = max(float(np.max(y_pts) - np.min(y_pts)), 1.0)
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
    hr_per_layer = {k + 1: float(A_fit * alpha_fit * np.exp(-alpha_fit * t)) for k, t in enumerate(t_pts)}

    t_star = None
    if rate_std_s > 0 and heating_rate_t0 > rate_std_s:
        ratio = rate_std_s / heating_rate_t0
        t_star = float(-np.log(ratio) / alpha_fit)

    fit_ok = True
    for param, sigma in [(T0_fit, sigma_T0), (A_fit, sigma_A), (alpha_fit, sigma_alpha)]:
        if abs(param) > 1e-9 and sigma / abs(param) > FIT_QUALITY_THRESH:
            fit_ok = False
            break

    return {
        "success": True,
        "fit_ok": fit_ok,
        "T0": float(T0_fit),
        "A": float(A_fit),
        "alpha": float(alpha_fit),
        "T_inf": float(T0_fit + A_fit),
        "t_star": t_star,
        "sigma_T0": float(sigma_T0),
        "sigma_A": float(sigma_A),
        "sigma_alpha": float(sigma_alpha),
        "heating_rate_t0": float(heating_rate_t0),
        "hr_per_layer": hr_per_layer,
    }

print(f"[INFO] Carico dataset: {dataset_path}")
df_full = pd.read_csv(dataset_path)

df = df_full[
    (df_full["frame_type"] == "core") &
    (df_full["roi_valid_frac"] >= 1.0)
].copy()
print(f"[INFO] Righe dopo filtro: {len(df)}")

if DT_LAYER_OVERRIDE is not None:
    dt_layer = float(DT_LAYER_OVERRIDE)
    print(f"[INFO] dt_layer (override): {dt_layer:.3f} s")
else:
    dt_estimates = []
    for spec_name, spec_info in SPECIMENS.items():
        if spec_name in ALWAYS_EXCLUDE:
            continue
        restart_frame = spec_info.get("restart_frame")
        layers_front = spec_info.get("layers_front")
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

rate_std_s = RATE_STD_PER_LAYER / dt_layer
print(f"[INFO] rate_std = {RATE_STD_PER_LAYER} degC/layer => {rate_std_s:.4f} degC/s")

rows = []

for specimen in sorted(df["specimen"].unique()):
    if specimen in ALWAYS_EXCLUDE:
        continue
    if specimen in STD_SPECIMENS and not INCLUDE_STD:
        continue

    spec_info = SPECIMENS.get(specimen)
    if spec_info is None:
        print(f" [SKIP] {specimen}: non in SPECIMENS")
        continue

    restart_frame = spec_info.get("restart_frame")
    layers_front = spec_info.get("layers_front")
    if restart_frame is None or layers_front is None:
        print(f" [SKIP] {specimen}: restart_frame o layers_front mancanti")
        continue

    df_spec = df[df["specimen"] == specimen]
    if df_spec.empty:
        print(f" [SKIP] {specimen}: nessuna riga dopo filtri")
        continue

    t_pts, y_pts, s_pts = [], [], []
    T_per_layer = {}
    std_spaziali = []
    skew_spaziali = []

    for k_idx, (fs, fe) in enumerate(layers_front):
        mask = (df_spec["frame_idx"] >= fs) & (df_spec["frame_idx"] <= fe)
        df_lyr = df_spec[mask]
        if df_lyr.empty:
            continue

        t_mean = (df_lyr["frame_idx"].mean() - restart_frame) / 3.0
        y_mean = df_lyr["roi_mean_C"].mean()
        y_std = df_lyr["roi_mean_C"].std(ddof=1) if len(df_lyr) > 1 else 0.0

        t_pts.append(t_mean)
        y_pts.append(y_mean)
        s_pts.append(y_std)
        T_per_layer[k_idx + 1] = y_mean

        all_pixels = []
        for raw in df_lyr["roi_pixels_raw"].dropna():
            try:
                vals = [float(v) for v in str(raw).split(",") if v.strip()]
                all_pixels.extend(vals)
            except Exception:
                pass
        if len(all_pixels) > 1:
            std_spaziali.append(float(np.std(all_pixels, ddof=1)))
            skew_spaziali.append(float(skew(all_pixels)) if len(all_pixels) > 2 else 0.0)

    if len(t_pts) < 3:
        print(f" [SKIP] {specimen}: solo {len(t_pts)} layer validi (min 3)")
        continue

    res = fit_heating_wls(np.array(t_pts), np.array(y_pts), np.array(s_pts), rate_std_s)
    if not res["success"]:
        print(f" [SKIP] {specimen}: fit fallito")
        continue

    valid_Ts = [T_per_layer[k] for k in sorted(T_per_layer.keys())]
    delta_T = float(valid_Ts[-1] - valid_Ts[0]) if len(valid_Ts) >= 2 else None

    row = {
        "specimen": specimen,
        "n_layers": len(t_pts),
        "T0": res["T0"],
        "A": res["A"],
        "alpha": res["alpha"],
        "T_inf": res["T_inf"],
        "t_star": res["t_star"],
        "heating_rate_t0": res["heating_rate_t0"],
        "sigma_T0": res["sigma_T0"],
        "sigma_A": res["sigma_A"],
        "sigma_alpha": res["sigma_alpha"],
        "fit_ok": res["fit_ok"],
        "delta_T": delta_T,
        "roi_std_spaziale_mean": float(np.mean(std_spaziali)) if std_spaziali else None,
        "roi_skew_spaziale_mean": float(np.mean(skew_spaziali)) if skew_spaziali else None,
    }

    for k in range(1, len(t_pts) + 1):
        row[f"hr_t{k}"] = res["hr_per_layer"].get(k, None)

    for k in range(1, len(layers_front) + 1):
        row[f"T{k}"] = T_per_layer.get(k, None)

    rows.append(row)
    print(f" [OK] {specimen:<25} T0={res['T0']:.2f} A={res['A']:.2f} alpha={res['alpha']:.5f} delta_T={delta_T:.2f} fit_ok={res['fit_ok']}")

feat_roi = pd.DataFrame(rows)

if df_mech is not None:
    feat_roi = feat_roi.merge(df_mech, on="specimen", how="left")
    n_miss = feat_roi["peso_rottura_kg"].isna().sum()
    if n_miss > 0:
        miss_specs = feat_roi[feat_roi["peso_rottura_kg"].isna()]["specimen"].unique()
        print(f"[WARN] {n_miss} provini senza peso_rottura_kg nel CSV meccanico: {list(miss_specs)}")
else:
    feat_roi["pausa_s"] = None
    feat_roi["peso_rottura_kg"] = None

if df_bed is not None:
    feat_roi = feat_roi.merge(df_bed, on="specimen", how="left")
else:
    feat_roi["bed_temp_C"] = float("nan")
    feat_roi["bed_temp_std_C"] = float("nan")

id_cols = ["specimen", "pausa_s", "peso_rottura_kg"]
fit_cols = ["n_layers", "T0", "A", "alpha", "T_inf", "t_star", "heating_rate_t0"]
hr_cols = sorted([c for c in feat_roi.columns if c.startswith("hr_t")], key=lambda c: int(c[4:]))
qual_cols = ["delta_T", "sigma_T0", "sigma_A", "sigma_alpha", "fit_ok"]
stat_cols = ["roi_std_spaziale_mean", "roi_skew_spaziale_mean"]
bed_cols = ["bed_temp_C", "bed_temp_std_C"]
t_cols = sorted([c for c in feat_roi.columns if c.startswith("T") and c[1:].isdigit()], key=lambda c: int(c[1:]))

ordered = id_cols + fit_cols + hr_cols + qual_cols + stat_cols + bed_cols + t_cols
extra = [c for c in feat_roi.columns if c not in ordered]
feat_roi = feat_roi[ordered + extra]

out_path = OUTPUT_DIR / f"feature_table_roi_{DATASET_NAME}.csv"
feat_roi.to_csv(out_path, index=False)
print(f"\n[DONE] Feature table ROI salvata: {out_path}")
print(f" Shape: {feat_roi.shape[0]} provini x {feat_roi.shape[1]} colonne")
print(f" Colonne: {list(feat_roi.columns)}")