"""
build_feature_table.py
======================
Per ogni provino e dataset CSV di ROI, produce DUE output CSV:

1. feature_table_roi_<DATASET_NAME>.csv  (una riga per provino — livello ROI)
2. feature_table_px_<DATASET_NAME>.csv   (una riga per (provino, pixel) — livello pixel)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MODELLO DI RISCALDAMENTO (applicato alla media ROI):
  T(t) = T0 + A * (1 - exp(-alpha * t))

  T0      = temperatura al restart stimata dal fit [°C]
  A       = escursione termica verso il regime [°C]
  alpha   = heating rate [1/s]
  T_inf   = T0 + A   (temperatura asintotica)
  heating_rate_t0 = A * alpha   [°C/s]  (derivata del modello a t=0)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
COLONNE OUTPUT — livello ROI (feature_table_roi_*):
  specimen          : nome provino
  pause_group       : famiglia di pausa (10s / 30s / 60s / 90s / std)
  T1..T6            : temperatura media ROI al layer k [°C]
  T0_roi            : T0 stimato dal fit sulla media ROI [°C]
  A_roi             : A dal fit ROI [°C]
  alpha_roi         : alpha dal fit ROI [1/s]
  T_inf_roi         : T_inf = T0 + A dal fit ROI [°C]
  heating_rate_t0   : A*alpha (derivata a t=0) [°C/s]
  sigma_T0          : incertezza T0 (1-sigma dal fit)
  sigma_A           : incertezza A
  sigma_alpha       : incertezza alpha
  roi_std_mean      : std spaziale media della ROI su tutti i layer [°C]
  n_layers          : numero di layer usati nel fit
  fit_success       : bool — True se il fit ha converso

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
COLONNE OUTPUT — livello pixel (feature_table_px_*):
  specimen          : nome provino
  pause_group       : famiglia di pausa
  pixel_row         : riga pixel relativa alla ROI (0-based)
  pixel_col         : colonna pixel relativa alla ROI (0-based)
  T1..T6            : temperatura media del pixel al layer k [°C]
  delta_T1..delta_T6: T_k - T_1 per il pixel (escursione rispetto al primo layer)
  T_mean_across_layers : media del pixel su tutti i layer [°C]
  T_std_across_layers  : std temporale del pixel su tutti i layer [°C]

  NOTA: Il fit esponenziale NON viene eseguito pixel per pixel (6 punti, 3
  parametri liberi → instabile). Le feature di fit sono nella tabella ROI.
  I pixel danno la distribuzione spaziale intra-ROI.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PREREQUISITI COLONNE CSV DI INPUT:
  specimen, frame_idx, frame_type, layer_index, roi_valid_frac,
  roi_mean_C, roi_std_C,
  roi_pixels_raw  (stringa CSV flat di float, un valore per pixel della ROI)
  La colonna roi_pixels_raw è OPZIONALE: se assente, la tabella pixel
  viene saltata e viene prodotta solo la tabella ROI.
"""

import sys
import re
import warnings
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from scipy.optimize import curve_fit

# ---------------------------------------------------------------------------
# Path setup — stessa logica degli altri script del progetto
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
#  CONFIG — modifica solo questa sezione
# ===========================================================================

DATASET_NAME = "ROI_wide_3_10_depth_1_4"

# Soglia minima fraction valida della ROI
ROI_MIN_VALID_FRAC = 1.0

# Floor per la sigma nei pesi WLS (evita divisione per zero)
SIGMA_FLOOR = 0.1

# Rate di riscaldamento standard [°C/layer] — usato per calcolare t*
RATE_STD_PER_LAYER = 0.240

# Override durata layer in secondi (None = stima automatica)
DT_LAYER_OVERRIDE = None

# Provini da escludere sempre
ALWAYS_EXCLUDE = {"Rec-023"}
STD_SPECIMENS  = {"Rec-027_std_2", "Rec-G3_std_1"}

# Se True, include i provini STD nella feature table
INCLUDE_STD_SPECIMENS = False

# Se True, genera i grafici di estrapolazione per ogni provino
SAVE_PLOTS = True

# ===========================================================================
#  FINE CONFIG
# ===========================================================================


# ---------------------------------------------------------------------------
# Path derivati
# ---------------------------------------------------------------------------
DATASETS_DIR = CREAZIONE_DS / "datasets"
OUTPUT_DIR   = SCRIPT_DIR / "output" / DATASET_NAME
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR    = OUTPUT_DIR / "plots"
if SAVE_PLOTS:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

dataset_path = DATASETS_DIR / f"{DATASET_NAME}.csv"

PAUSE_PALETTES = {
    "10s":  ["#922b21", "#e74c3c"],
    "30s":  ["#1e8449", "#27ae60", "#58d68d", "#a9dfbf"],
    "60s":  ["#1a5276", "#2980b9", "#7fb3d3"],
    "90s":  ["#6c3483", "#8e44ad", "#c39bd3"],
    "std":  ["#717d7e", "#aab7b8"],
}


# ---------------------------------------------------------------------------
# Helpers
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


def heating_model(t, T0, A, alpha):
    return T0 + A * (1.0 - np.exp(-alpha * t))


def fit_heating_wls(t_pts, y_pts, s_pts, t_extrap):
    """WLS fit del modello di riscaldamento — identica a plot_temp_layer_extrap.py"""
    sigma_w = np.where(s_pts > 0, s_pts, SIGMA_FLOOR)

    T0_0    = float(y_pts[np.argmin(t_pts)])
    A_0     = max(float(np.max(y_pts) - np.min(y_pts)), 1.0)
    alpha_0 = 0.05

    try:
        popt, pcov = curve_fit(
            heating_model, t_pts, y_pts,
            p0=[T0_0, A_0, alpha_0],
            sigma=sigma_w,
            absolute_sigma=True,
            bounds=([-np.inf, 0.0, 1e-6], [np.inf, np.inf, np.inf]),
            maxfev=20000,
        )
    except Exception as e:
        warnings.warn("Fit WLS fallito: " + str(e))
        return None

    T0_fit, A_fit, alpha_fit = popt
    perr = np.sqrt(np.diag(pcov))

    y_curve = heating_model(t_extrap, T0_fit, A_fit, alpha_fit)

    # Banda di incertezza sulla curva (propagazione covarianza)
    exp_at    = np.exp(-alpha_fit * t_extrap)
    dT_dT0    = np.ones_like(t_extrap)
    dT_dA     = 1.0 - exp_at
    dT_dalpha = A_fit * t_extrap * exp_at
    var_curve = (
        dT_dT0**2    * pcov[0, 0]
        + dT_dA**2     * pcov[1, 1]
        + dT_dalpha**2 * pcov[2, 2]
        + 2 * dT_dT0    * dT_dA     * pcov[0, 1]
        + 2 * dT_dT0    * dT_dalpha * pcov[0, 2]
        + 2 * dT_dA     * dT_dalpha * pcov[1, 2]
    )
    sigma_curve = np.sqrt(np.maximum(var_curve, 0.0))

    return {
        "T0"          : float(T0_fit),
        "A"           : float(A_fit),
        "alpha"       : float(alpha_fit),
        "T_inf"       : float(T0_fit + A_fit),
        "sigma_T0"    : float(perr[0]),
        "sigma_A"     : float(perr[1]),
        "sigma_alpha" : float(perr[2]),
        "y_curve"     : y_curve,
        "y_upper"     : y_curve + sigma_curve,
        "y_lower"     : y_curve - sigma_curve,
    }


def parse_roi_pixels_raw(raw_str):
    """
    Converte la stringa roi_pixels_raw in un array numpy 1D di float.
    Gestisce sia formato CSV flat ("12.3,14.1,...") sia JSON list ("[12.3,14.1,...]").
    Ritorna None se non è possibile parsare.
    """
    if pd.isna(raw_str) or str(raw_str).strip() == "":
        return None
    s = str(raw_str).strip()
    try:
        if s.startswith("["):
            arr = np.array(json.loads(s), dtype=float)
        else:
            arr = np.array([float(x) for x in s.split(",")], dtype=float)
        return arr
    except Exception:
        return None


def plot_specimen_extrapolation(
    specimen, t_pts, y_pts, s_pts, t_extrap, fit_res, pause_group, output_dir
):
    """
    Genera e salva un grafico di estrapolazione per un singolo provino:
    - curva fit con banda di incertezza
    - punti layer con errore
    - stella a T0 stimato
    - annotazione parametri fit
    """
    palette = PAUSE_PALETTES.get(pause_group, PAUSE_PALETTES["std"])
    color   = palette[0]

    fig, ax = plt.subplots(figsize=(8, 5))

    # Banda incertezza
    ax.fill_between(
        t_extrap, fit_res["y_lower"], fit_res["y_upper"],
        color=color, alpha=0.18, linewidth=0, label="±1σ curva"
    )
    # Curva fit
    ax.plot(t_extrap, fit_res["y_curve"], color=color, linewidth=2.0, label="Fit WLS")
    # Punti layer
    ax.scatter(t_pts, y_pts, color=color, s=65, zorder=5,
               edgecolors="white", linewidths=0.7, label="Media ROI per layer")
    ax.errorbar(t_pts, y_pts, yerr=s_pts, fmt="none",
                ecolor=color, elinewidth=1.0, capsize=3, alpha=0.6)
    # Stella T0
    ax.scatter(0, fit_res["T0"], color=color, s=120, marker="*",
               zorder=6, edgecolors="white", linewidths=0.7, label=f"T₀={fit_res['T0']:.2f}°C")
    # Linea verticale restart
    ax.axvline(0, color="#555", linewidth=1.0, linestyle=":", alpha=0.7)

    # Annotazione parametri
    txt = (
        f"T₀ = {fit_res['T0']:.2f} ± {fit_res['sigma_T0']:.3f} °C\n"
        f"A  = {fit_res['A']:.2f} ± {fit_res['sigma_A']:.3f} °C\n"
        f"α  = {fit_res['alpha']:.5f} ± {fit_res['sigma_alpha']:.6f} 1/s\n"
        f"T∞ = {fit_res['T_inf']:.2f} °C\n"
        f"dT/dt|₀ = {fit_res['A'] * fit_res['alpha']:.4f} °C/s"
    )
    ax.text(0.97, 0.05, txt, transform=ax.transAxes, fontsize=8.5,
            va="bottom", ha="right",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.82, ec="#cccccc"))

    ax.set_title(
        f"{specimen}  [pausa {pause_group}]\n"
        f"Dataset: {DATASET_NAME}  |  Modello: T(t)=T₀+A·(1−e^(−αt))",
        fontsize=10, pad=10
    )
    ax.set_xlabel("Tempo dal restart [s]", fontsize=10)
    ax.set_ylabel("Temperatura media ROI [°C]", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.4, linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=8, framealpha=0.85)

    plt.tight_layout()
    safe_name = re.sub(r'[^\w\-]', '_', specimen)
    out_path  = output_dir / f"extrap_{safe_name}.png"
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Carica dataset
# ---------------------------------------------------------------------------
print(f"[INFO] Carico dataset: {dataset_path}")
df_full = pd.read_csv(dataset_path)

df = df_full[
    (df_full["frame_type"] == "core") &
    (df_full["roi_valid_frac"] >= ROI_MIN_VALID_FRAC)
].copy()
print(f"[INFO] Righe dopo filtro core + roi_valid_frac >= {ROI_MIN_VALID_FRAC}: {len(df)}")

# Verifica presenza colonna pixel
HAS_PIXEL_COL = "roi_pixels_raw" in df.columns
if not HAS_PIXEL_COL:
    print("[WARN] Colonna 'roi_pixels_raw' non trovata — feature table pixel NON verrà prodotta.")

# ---------------------------------------------------------------------------
# Stima dt_layer medio
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
            mask  = (df_spec["frame_idx"] >= fs) & (df_spec["frame_idx"] <= fe)
            dl    = df_spec[mask]
            if not dl.empty:
                t_means.append((dl["frame_idx"].mean() - restart_frame) / 3.0)
        if len(t_means) >= 2:
            diffs = np.diff(sorted(t_means))
            dt_estimates.extend(diffs.tolist())
    dt_layer = float(np.median(dt_estimates)) if dt_estimates else (1.0 / 3.0)
    print(f"[INFO] dt_layer stimato: {dt_layer:.3f} s  (mediana di {len(dt_estimates)} differenze)")

rate_std_s = RATE_STD_PER_LAYER / dt_layer
print(f"[INFO] rate_std = {RATE_STD_PER_LAYER} °C/layer  =>  {rate_std_s:.4f} °C/s")


# ===========================================================================
# LOOP PRINCIPALE — un provino alla volta
# ===========================================================================

rows_roi = []   # una riga per provino
rows_px  = []   # una riga per (provino, pixel)
skipped  = []

all_specimens = sorted(df["specimen"].unique())

for specimen in all_specimens:

    # ------------------------------------------------------------------
    # Filtri
    # ------------------------------------------------------------------
    if specimen in ALWAYS_EXCLUDE:
        skipped.append((specimen, "always_exclude"))
        continue
    if specimen in STD_SPECIMENS and not INCLUDE_STD_SPECIMENS:
        skipped.append((specimen, "STD escluso"))
        continue

    spec_info = SPECIMENS.get(specimen)
    if spec_info is None:
        skipped.append((specimen, "non in SPECIMENS"))
        continue

    restart_frame = spec_info.get("restart_frame")
    layers_front  = spec_info.get("layers_front")
    if restart_frame is None or layers_front is None:
        skipped.append((specimen, "restart_frame o layers_front None"))
        continue

    df_spec = df[df["specimen"] == specimen].copy()
    if df_spec.empty:
        skipped.append((specimen, "nessuna riga dopo filtri"))
        continue

    pause_group = get_pause_group(specimen)

    # ------------------------------------------------------------------
    # Aggregazione per layer: t, media ROI, std ROI
    # ------------------------------------------------------------------
    t_pts   = []
    y_pts   = []    # media di roi_mean_C per il layer
    s_pts   = []    # std  di roi_mean_C per il layer
    std_pts = []    # media di roi_std_C  per il layer (std spaziale intra-frame)

    # Per la tabella pixel: dizionario layer_idx → array (n_pixels,)
    px_layer_means = {}  # layer_idx (1-based) → array (n_px,) temperature medie pixel

    n_layers_found = len(layers_front)

    for k_idx, (frame_start, frame_end) in enumerate(layers_front):
        layer_num = k_idx + 1  # 1-based (T1, T2, ...)
        mask      = (
            (df_spec["frame_idx"] >= frame_start) &
            (df_spec["frame_idx"] <= frame_end)
        )
        df_layer = df_spec[mask]
        if df_layer.empty:
            continue

        t_mean = (df_layer["frame_idx"].mean() - restart_frame) / 3.0
        y_mean = float(df_layer["roi_mean_C"].mean())
        y_std  = float(df_layer["roi_mean_C"].std(ddof=1)) if len(df_layer) > 1 else 0.0
        std_sp = float(df_layer["roi_std_C"].mean()) if "roi_std_C" in df_layer.columns else np.nan

        t_pts.append(t_mean)
        y_pts.append(y_mean)
        s_pts.append(y_std)
        std_pts.append(std_sp)

        # ---------- pixel level ----------
        if HAS_PIXEL_COL:
            px_arrays = []
            for raw in df_layer["roi_pixels_raw"].dropna():
                arr = parse_roi_pixels_raw(raw)
                if arr is not None:
                    px_arrays.append(arr)
            if px_arrays:
                # media temporale del pixel su tutti i frame del layer
                try:
                    stack = np.vstack(px_arrays)   # (n_frames, n_pixels)
                    px_layer_means[layer_num] = stack.mean(axis=0)
                except ValueError:
                    pass  # frame con dimensioni diverse — skip pixel per questo layer

    if len(t_pts) < 3:
        skipped.append((specimen, f"solo {len(t_pts)} layer (min 3), skip"))
        continue

    t_pts   = np.array(t_pts)
    y_pts   = np.array(y_pts)
    s_pts   = np.array(s_pts)
    std_pts = np.array(std_pts)
    t_extrap = np.linspace(0.0, t_pts.max() * 1.05, 400)

    # ------------------------------------------------------------------
    # Fit WLS sulla media ROI
    # ------------------------------------------------------------------
    fit_res = fit_heating_wls(t_pts, y_pts, s_pts, t_extrap)
    fit_ok  = fit_res is not None

    # ------------------------------------------------------------------
    # Costruzione riga ROI
    # ------------------------------------------------------------------
    row_roi = {
        "specimen"       : specimen,
        "pause_group"    : pause_group,
        "n_layers"       : len(t_pts),
        "fit_success"    : fit_ok,
    }

    # T1..T_n (fino a 6, poi NaN)
    for k in range(1, 7):
        if k <= len(y_pts):
            row_roi[f"T{k}"]    = round(y_pts[k - 1], 4)
            row_roi[f"t{k}_s"]  = round(t_pts[k - 1], 4)   # tempo in secondi del layer k
        else:
            row_roi[f"T{k}"]   = np.nan
            row_roi[f"t{k}_s"] = np.nan

    if fit_ok:
        row_roi["T0_roi"]          = round(fit_res["T0"],    4)
        row_roi["A_roi"]           = round(fit_res["A"],     4)
        row_roi["alpha_roi"]       = round(fit_res["alpha"], 6)
        row_roi["T_inf_roi"]       = round(fit_res["T_inf"], 4)
        row_roi["heating_rate_t0"] = round(fit_res["A"] * fit_res["alpha"], 6)
        row_roi["sigma_T0"]        = round(fit_res["sigma_T0"],    4)
        row_roi["sigma_A"]         = round(fit_res["sigma_A"],     4)
        row_roi["sigma_alpha"]     = round(fit_res["sigma_alpha"], 6)
    else:
        for col in ["T0_roi", "A_roi", "alpha_roi", "T_inf_roi",
                    "heating_rate_t0", "sigma_T0", "sigma_A", "sigma_alpha"]:
            row_roi[col] = np.nan

    row_roi["roi_std_mean"] = round(float(np.nanmean(std_pts)), 4)
    rows_roi.append(row_roi)

    # ------------------------------------------------------------------
    # Costruzione righe pixel
    # ------------------------------------------------------------------
    if HAS_PIXEL_COL and px_layer_means:
        # Determina la dimensione consistente dei pixel disponibili
        sizes = {arr.size for arr in px_layer_means.values()}
        if len(sizes) == 1:
            n_px = sizes.pop()

            # Prova a determinare la forma della ROI (wide x depth)
            # dal nome del dataset: ROI_wide_<w>_<d>_depth_<d1>_<d2>
            match_shape = re.search(r'wide_(\d+)_(\d+)_depth_(\d+)_(\d+)', DATASET_NAME)
            if match_shape:
                n_rows = int(match_shape.group(2))  # wide (altezza)
                n_cols = int(match_shape.group(4))  # depth (larghezza)
                if n_rows * n_cols != n_px:
                    n_rows, n_cols = 1, n_px        # fallback lineare
            else:
                n_rows, n_cols = 1, n_px

            for px_idx in range(n_px):
                px_row_idx = px_idx // n_cols
                px_col_idx = px_idx %  n_cols

                row_px = {
                    "specimen"    : specimen,
                    "pause_group" : pause_group,
                    "pixel_row"   : px_row_idx,
                    "pixel_col"   : px_col_idx,
                    "pixel_flat"  : px_idx,
                }

                layer_temps = []
                for k in range(1, 7):
                    if k in px_layer_means:
                        val = round(float(px_layer_means[k][px_idx]), 4)
                    else:
                        val = np.nan
                    row_px[f"T{k}"] = val
                    layer_temps.append(val)

                # delta rispetto al layer 1
                T1_px = row_px["T1"]
                for k in range(1, 7):
                    v = row_px[f"T{k}"]
                    row_px[f"delta_T{k}"] = round(v - T1_px, 4) if not np.isnan(v) and not np.isnan(T1_px) else np.nan

                valid_temps = [v for v in layer_temps if not np.isnan(v)]
                row_px["T_mean_across_layers"] = round(float(np.mean(valid_temps)),   4) if valid_temps else np.nan
                row_px["T_std_across_layers"]  = round(float(np.std(valid_temps, ddof=1)), 4) if len(valid_temps) > 1 else np.nan

                rows_px.append(row_px)
        else:
            print(f"[WARN] {specimen}: dimensioni pixel inconsistenti tra layer {sizes} — righe pixel saltate.")

    # ------------------------------------------------------------------
    # Grafico di estrapolazione
    # ------------------------------------------------------------------
    if SAVE_PLOTS and fit_ok:
        out_plot = plot_specimen_extrapolation(
            specimen, t_pts, y_pts, s_pts, t_extrap, fit_res, pause_group, PLOTS_DIR
        )
        print(f"  [PLOT] {out_plot.name}")

    print(
        f"[OK] {specimen:<28}  layers={len(t_pts)}"
        + (f"  T0={fit_res['T0']:.2f}°C  α={fit_res['alpha']:.5f}  A={fit_res['A']:.2f}" if fit_ok else "  fit FALLITO")
    )


# ===========================================================================
# SALVATAGGIO
# ===========================================================================

# --- Feature table ROI ---
df_roi = pd.DataFrame(rows_roi)

# Riordina colonne in modo leggibile
col_order_roi = (
    ["specimen", "pause_group", "n_layers", "fit_success"]
    + [f"T{k}"    for k in range(1, 7)]
    + [f"t{k}_s"  for k in range(1, 7)]
    + ["T0_roi", "A_roi", "alpha_roi", "T_inf_roi",
       "heating_rate_t0", "sigma_T0", "sigma_A", "sigma_alpha",
       "roi_std_mean"]
)
col_order_roi = [c for c in col_order_roi if c in df_roi.columns]
df_roi = df_roi[col_order_roi]

out_roi = OUTPUT_DIR / f"feature_table_roi_{DATASET_NAME}.csv"
df_roi.to_csv(out_roi, index=False)
print(f"\n[SAVED] Feature table ROI  → {out_roi}  ({len(df_roi)} righe)")

# --- Feature table pixel ---
if rows_px:
    df_px = pd.DataFrame(rows_px)

    col_order_px = (
        ["specimen", "pause_group", "pixel_row", "pixel_col", "pixel_flat"]
        + [f"T{k}"       for k in range(1, 7)]
        + [f"delta_T{k}" for k in range(1, 7)]
        + ["T_mean_across_layers", "T_std_across_layers"]
    )
    col_order_px = [c for c in col_order_px if c in df_px.columns]
    df_px = df_px[col_order_px]

    out_px = OUTPUT_DIR / f"feature_table_px_{DATASET_NAME}.csv"
    df_px.to_csv(out_px, index=False)
    print(f"[SAVED] Feature table pixel → {out_px}  ({len(df_px)} righe)")
else:
    print("[INFO] Nessuna riga pixel prodotta (colonna roi_pixels_raw assente o vuota).")

# --- Report provini saltati ---
if skipped:
    print("\n[INFO] Provini saltati:")
    for name, reason in skipped:
        print(f"  - {name}: {reason}")

print(f"\n[DONE] Output in: {OUTPUT_DIR}")
