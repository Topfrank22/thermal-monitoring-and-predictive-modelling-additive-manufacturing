"""
plot_temp_layer_extrap.py
--------------------------
Per ogni provino calcola UN punto per layer:
  - X  = tempo medio del layer dal restart [s]  (t_i)
  - Y  = media di roi_mean_C sui frame del layer (T_i_bar)
  - sy = std  di roi_mean_C sui frame del layer  (sigma_i)

MODELLO DI RISCALDAMENTO (accumulo termico):
  T(t) = T0 + A * (1 - exp(-alpha * t))

  T0    = temperatura al restart (t=0)          [stimata dal fit]
  A     = escursione termica verso il regime     [stimata dal fit]
  alpha = rate di riscaldamento [1/s]            [stimato dal fit]
  T_inf = T0 + A  (temperatura asintotica stimata)

LINEE TRATTEGGIATE STD (solo nei grafici per famiglia):
  Rappresentano il rate di riscaldamento "naturale" senza pausa.
  rate_std = 0.240 degC/layer (media dei due provini standard:
             Rec-027_std_2: +0.239, Rec-G3_std_1: +0.241).
  Convertito in degC/s: rate_std_s = 0.240 / dt_layer
  dove dt_layer = durata media di un layer in secondi.

  Per ogni provino viene tracciata una linea tratteggiata del suo stesso
  colore (con alpha ridotto) con pendenza rate_std_s:

  - Caso normale (t* esiste):
      dT/dt|_{t*} = A*alpha*exp(-alpha*t*) = rate_std_s  =>  t* = -ln(rate_std_s/(A*alpha))/alpha
      shift: c = T(t*) - rate_std_s * t*
      La linea parte da t* fino alla fine dell'asse X.
      Un marcatore 'x' indica il punto di tangenza.

  - Caso anomalo (pendenza massima della curva < rate_std_s):
      Il provino non raggiunge mai il regime STD.
      La linea tratteggiata parte dall'ultimo punto della curva
      con pendenza rate_std_s ed e' estesa di EXTRA_FRAC a destra,
      cosi' si vede visivamente quanto manca.
      Un marcatore 'triangle_right' segnala il punto di partenza.

Output:
    Analisi_dataset/<DATASET_NAME>/LayerExtrap_<METRIC>_<MODEL>.png       (grafico globale)
    Analisi_dataset/<DATASET_NAME>/LayerExtrap_<METRIC>_<MODEL>_<PAUSA>.png  (per famiglia)
"""

import sys
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
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

DATASET_NAME = "ROI_wide_3_10_depth_1_3"

# Metrica: "mean" | "max" | "min"
PLOT_METRIC = "mean"

# Modello: "heating" | "linear" | "quadratic"
FIT_MODEL = "heating"

SIGMA_FLOOR = 0.1
SHOW_SHADOW = True
ROI_MIN_VALID_FRAC = 1.0
INCLUDE_STD_SPECIMENS = False

# Rate di riscaldamento standard [degC/layer]
# Media dei due provini standard (Rec-027_std_2: 0.239, Rec-G3_std_1: 0.241)
RATE_STD_PER_LAYER = 0.240

# Durata media di un layer [s]: i frame sono a 3 fps, layer = ~1 frame medio
# Usato per convertire rate_std da degC/layer a degC/s
# Viene stimato automaticamente dai dati (vedi sotto), ma si puo' fissare qui
DT_LAYER_OVERRIDE = None   # None = stima automatica dai dati

# Estensione extra a destra per i provini che non raggiungono il rate STD
# (frazione dell'asse X totale del grafico)
EXTRA_FRAC = 0.20

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
    "mean": {"col": "roi_mean_C", "label": "Temperatura media ROI [\u00b0C]",    "file_prefix": "LayerExtrap_Mean"},
    "max":  {"col": "roi_max_C",  "label": "Temperatura massima ROI [\u00b0C]", "file_prefix": "LayerExtrap_Max"},
    "min":  {"col": "roi_min_C",  "label": "Temperatura minima ROI [\u00b0C]",  "file_prefix": "LayerExtrap_Min"},
}

if PLOT_METRIC not in METRIC_CONFIG:
    raise ValueError(f"PLOT_METRIC deve essere 'mean', 'max' o 'min'.")

metric_col   = METRIC_CONFIG[PLOT_METRIC]["col"]
metric_label = METRIC_CONFIG[PLOT_METRIC]["label"]
file_prefix  = METRIC_CONFIG[PLOT_METRIC]["file_prefix"]
output_fname = f"{file_prefix}_{FIT_MODEL}.png"


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
STD_SPECIMENS  = {"Rec-027_std_2", "Rec-G3_std_1"}


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


def fit_heating_wls(
    t_pts: np.ndarray,
    y_pts: np.ndarray,
    s_pts: np.ndarray,
    t_extrap: np.ndarray,
) -> dict:
    sigma_w = np.where(s_pts > 0, s_pts, SIGMA_FLOOR)

    def model(t, T0, A, alpha):
        return T0 + A * (1.0 - np.exp(-alpha * t))

    T0_0    = float(y_pts[np.argmin(t_pts)])
    A_0     = float(np.max(y_pts) - np.min(y_pts))
    if A_0 < 1.0:
        A_0 = 1.0
    alpha_0 = 0.05

    try:
        popt, pcov = curve_fit(
            model, t_pts, y_pts,
            p0=[T0_0, A_0, alpha_0],
            sigma=sigma_w,
            absolute_sigma=True,
            bounds=(
                [-np.inf, 0.0,    1e-6],
                [ np.inf, np.inf, np.inf],
            ),
            maxfev=20000,
        )
    except Exception as e:
        warnings.warn("Fit heating WLS fallito: " + str(e))
        return {"success": False}

    T0_fit, A_fit, alpha_fit = popt
    perr = np.sqrt(np.diag(pcov))
    sigma_T0, sigma_A, sigma_alpha = perr

    y_curve = model(t_extrap, T0_fit, A_fit, alpha_fit)

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
        "success"     : True,
        "y_curve"     : y_curve,
        "T0"          : T0_fit,
        "A"           : A_fit,
        "alpha"       : alpha_fit,
        "T_inf"       : T0_fit + A_fit,
        "sigma_T0"    : sigma_T0,
        "sigma_A"     : sigma_A,
        "sigma_alpha" : sigma_alpha,
        "y_upper"     : y_curve + sigma_curve,
        "y_lower"     : y_curve - sigma_curve,
    }


def fit_poly(
    t_pts: np.ndarray,
    y_pts: np.ndarray,
    s_pts: np.ndarray,
    degree: int,
    t_extrap: np.ndarray,
) -> dict:
    try:
        coeffs  = np.polyfit(t_pts, y_pts, degree)
        poly    = np.poly1d(coeffs)
        y_curve = poly(t_extrap)
        T0      = float(poly(0.0))

        y_upper = y_lower = None
        if SHOW_SHADOW and np.any(s_pts > 0):
            try:
                cu = np.poly1d(np.polyfit(t_pts, y_pts + s_pts, degree))
                cl = np.poly1d(np.polyfit(t_pts, y_pts - s_pts, degree))
                y_upper = cu(t_extrap)
                y_lower = cl(t_extrap)
            except Exception:
                pass

        return {"success": True, "y_curve": y_curve, "T0": T0,
                "y_upper": y_upper, "y_lower": y_lower}
    except Exception as e:
        warnings.warn("Fit polinomiale grado " + str(degree) + " fallito: " + str(e))
        return {"success": False}


def compute_tangency(A: float, alpha: float, rate_std_s: float, t_max: float):
    """
    Calcola il punto di tangenza t* dove la derivata del modello esponenziale
    eguaglia rate_std_s [degC/s]:
        A * alpha * exp(-alpha * t*) = rate_std_s
        => t* = -ln(rate_std_s / (A * alpha)) / alpha

    Ritorna t* se 0 <= t* <= t_max, altrimenti None.
    """
    denom = A * alpha
    if denom <= 0 or rate_std_s <= 0:
        return None
    ratio = rate_std_s / denom
    if ratio <= 0 or ratio > 1.0:  # ratio > 1 => t* < 0 o pendenza max < rate_std
        return None
    t_star = -np.log(ratio) / alpha
    if t_star < 0 or t_star > t_max:
        return None
    return float(t_star)


# ---------------------------------------------------------------------------
# Carica dataset
# ---------------------------------------------------------------------------
print("[INFO] Carico dataset: " + str(dataset_path))
df_full = pd.read_csv(dataset_path)

df = df_full[
    (df_full["frame_type"] == "core") &
    (df_full["roi_valid_frac"] >= ROI_MIN_VALID_FRAC)
].copy()
print("[INFO] Righe dopo filtro core + roi_valid_frac >= " + str(ROI_MIN_VALID_FRAC) + ": " + str(len(df)))


# ---------------------------------------------------------------------------
# Stima dt_layer medio (durata media di un layer in secondi)
# Usata per convertire rate_std da degC/layer a degC/s
# ---------------------------------------------------------------------------
if DT_LAYER_OVERRIDE is not None:
    dt_layer = float(DT_LAYER_OVERRIDE)
    print("[INFO] dt_layer (override): " + str(dt_layer) + " s")
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
            diffs = np.diff(sorted(t_means))
            dt_estimates.extend(diffs.tolist())
    dt_layer = float(np.median(dt_estimates)) if dt_estimates else (1.0 / 3.0)
    print("[INFO] dt_layer stimato: " + f"{dt_layer:.3f}" + " s  (mediana di " + str(len(dt_estimates)) + " differenze)")

# Rate STD convertito in degC/s
rate_std_s = RATE_STD_PER_LAYER / dt_layer
print("[INFO] rate_std = " + str(RATE_STD_PER_LAYER) + " degC/layer  =>  " + f"{rate_std_s:.4f}" + " degC/s")


# ===========================================================================
# FASE 1: raccolta dati per tutti i provini
# ===========================================================================

all_specimens  = df["specimen"].unique()
skipped        = []

specimen_data = {}
pause_counters_global = {k: 0 for k in PAUSE_PALETTES}

if FIT_MODEL == "heating":
    COL_PROVINO  = "Provino"
    COL_T0       = "T0 [\u00b0C]"
    COL_ST0      = "\u00b1\u03c3_T0"
    COL_A        = "A [\u00b0C]"
    COL_SA       = "\u00b1\u03c3_A"
    COL_ALPHA    = "alpha [1/s]"
    COL_SALPHA   = "\u00b1\u03c3_\u03b1"
    COL_TINF     = "T_inf"
    header = (
        f"{COL_PROVINO:<25}  {COL_T0:>8}  {COL_ST0:>6}  "
        f"{COL_A:>7}  {COL_SA:>6}  "
        f"{COL_ALPHA:>11}  {COL_SALPHA:>9}  "
        f"{COL_TINF:>7}  N"
    )
    print("")
    print(header)
    print("-" * 95)

for specimen in sorted(all_specimens):

    if specimen in ALWAYS_EXCLUDE:
        skipped.append((specimen, "invalido"))
        continue
    if specimen in STD_SPECIMENS and not INCLUDE_STD_SPECIMENS:
        skipped.append((specimen, "standard - escluso"))
        continue

    spec_info = SPECIMENS.get(specimen)
    if spec_info is None:
        skipped.append((specimen, "non trovato in SPECIMENS"))
        continue

    restart_frame = spec_info.get("restart_frame")
    if restart_frame is None:
        skipped.append((specimen, "restart_frame None"))
        continue

    layers_front = spec_info.get("layers_front")
    if layers_front is None:
        skipped.append((specimen, "layers_front None"))
        continue

    df_spec = df[df["specimen"] == specimen].copy()
    if df_spec.empty:
        skipped.append((specimen, "nessuna riga dopo filtri"))
        continue

    t_pts = []
    y_pts = []
    s_pts = []

    for frame_start, frame_end in layers_front:
        mask = (
            (df_spec["frame_idx"] >= frame_start) &
            (df_spec["frame_idx"] <= frame_end)
        )
        df_layer = df_spec[mask]
        if df_layer.empty:
            continue

        t_mean = (df_layer["frame_idx"].mean() - restart_frame) / 3.0
        y_mean = df_layer[metric_col].mean()
        y_std  = df_layer[metric_col].std(ddof=1) if len(df_layer) > 1 else 0.0

        t_pts.append(t_mean)
        y_pts.append(y_mean)
        s_pts.append(y_std)

    if len(t_pts) < 3:
        skipped.append((specimen, "solo " + str(len(t_pts)) + " layer (min 3), skip"))
        continue

    t_pts = np.array(t_pts)
    y_pts = np.array(y_pts)
    s_pts = np.array(s_pts)
    t_extrap = np.linspace(0.0, t_pts.max() * 1.05, 400)

    if FIT_MODEL == "heating":
        res = fit_heating_wls(t_pts, y_pts, s_pts, t_extrap)
    elif FIT_MODEL == "linear":
        res = fit_poly(t_pts, y_pts, s_pts, 1, t_extrap)
    elif FIT_MODEL == "quadratic":
        res = fit_poly(t_pts, y_pts, s_pts, 2, t_extrap)
    else:
        raise ValueError("FIT_MODEL non riconosciuto: '" + FIT_MODEL + "'")

    if not res["success"]:
        skipped.append((specimen, "fit fallito"))
        continue

    if FIT_MODEL == "heating":
        print(
            f"{specimen:<25}  {res['T0']:>8.2f}  {res['sigma_T0']:>6.3f}  "
            f"{res['A']:>7.2f}  {res['sigma_A']:>6.3f}  "
            f"{res['alpha']:>11.5f}  {res['sigma_alpha']:>9.6f}  "
            f"{res['T_inf']:>7.2f}  {len(t_pts)}"
        )

    pause_group  = get_pause_group(specimen)
    color_idx    = pause_counters_global[pause_group]
    pause_counters_global[pause_group] += 1

    specimen_data[specimen] = {
        "pause"    : pause_group,
        "t_pts"   : t_pts,
        "y_pts"   : y_pts,
        "s_pts"   : s_pts,
        "t_extrap": t_extrap,
        "res"     : res,
        "color_idx": color_idx,
    }

if skipped:
    print("\n[INFO] Provini saltati:")
    for name, reason in skipped:
        print("  - " + name + ": " + reason)


# ===========================================================================
# FASE 2: funzione di plot
# ===========================================================================

def plot_specimens(
    ax,
    specimens_to_plot: list,
    show_std_tangent: bool = False,
    x_max_override: float = None,
):
    """
    Disegna le curve di riscaldamento per i provini in specimens_to_plot.

    Se show_std_tangent=True e FIT_MODEL=="heating", per ogni provino:

    - Caso normale (t* esiste):
        Linea tratteggiata del colore del provino (alpha ridotto) con pendenza
        rate_std_s, che parte dal punto di tangenza t* fino alla fine dell'asse X.
        Marcatore 'x' sul punto di tangenza.

    - Caso anomalo (pendenza massima < rate_std_s):
        La curva non raggiunge mai il regime STD.
        Linea tratteggiata del colore del provino partente dall'ultimo punto
        della curva, estesa di EXTRA_FRAC * x_range a destra.
        Marcatore 'triangle_right' sul punto di partenza.
        La linea ha pendenza rate_std_s (non quella della curva).

    x_max_override: se fornito, forza l'asse X fino a quel valore (usato per
        estendere il grafico quando ci sono provini anomali).
    """
    legend_handles = []
    t_max_all = 0.0
    # raccoglie info per gestire l'asse X dopo aver plottato tutto
    anomaly_t_ends = []   # t_end dei provini anomali (per calcolare x_max esteso)

    # Prima passata: raccoglie t_max e identifica anomali per calcolare x_max globale
    for specimen in specimens_to_plot:
        sd = specimen_data[specimen]
        t_max_all = max(t_max_all, sd["t_extrap"].max())
        if show_std_tangent and FIT_MODEL == "heating":
            res = sd["res"]
            t_star = compute_tangency(res["A"], res["alpha"], rate_std_s, sd["t_extrap"].max())
            if t_star is None:
                anomaly_t_ends.append(sd["t_extrap"].max())

    # x_max per il grafico: se ci sono anomali, estendi
    if anomaly_t_ends:
        x_range_base = t_max_all
        x_max = t_max_all + EXTRA_FRAC * x_range_base
    else:
        x_max = t_max_all
    if x_max_override is not None:
        x_max = x_max_override

    normal_tangent_added  = False
    anomaly_tangent_added = False

    for specimen in specimens_to_plot:
        sd = specimen_data[specimen]
        res         = sd["res"]
        t_pts       = sd["t_pts"]
        y_pts       = sd["y_pts"]
        s_pts       = sd["s_pts"]
        t_extrap    = sd["t_extrap"]
        pause_group = sd["pause"]
        color_idx   = sd["color_idx"]

        palette = PAUSE_PALETTES.get(pause_group, PAUSE_PALETTES["std"])
        color   = palette[color_idx % len(palette)]

        # Curva principale
        ax.plot(t_extrap, res["y_curve"], color=color, linewidth=1.7,
                alpha=0.88, zorder=3)

        if SHOW_SHADOW and res.get("y_upper") is not None:
            ax.fill_between(
                t_extrap, res["y_lower"], res["y_upper"],
                color=color, alpha=0.13, linewidth=0, zorder=2
            )

        # Stella a t=0
        ax.scatter(0, res["T0"], color=color, s=90, marker="*",
                   zorder=6, edgecolors="white", linewidths=0.5)

        # Punti layer + errori
        ax.scatter(t_pts, y_pts, color=color, s=55, zorder=5,
                   edgecolors="white", linewidths=0.6)
        ax.errorbar(t_pts, y_pts, yerr=s_pts, fmt="none",
                    ecolor=color, elinewidth=1.0, capsize=3, alpha=0.55, zorder=4)

        # Legenda provino
        T0_str    = "T\u2080=" + f"{res['T0']:.1f}" + "\u00b0C"
        alpha_str = "\u03b1=" + f"{res['alpha']:.4f}"
        if FIT_MODEL == "heating":
            lbl = specimen + "  " + T0_str + "  " + alpha_str
        else:
            lbl = specimen + "  " + T0_str

        legend_handles.append(
            mlines.Line2D([], [], color=color, marker="o", markersize=5,
                          linewidth=1.5, label=lbl)
        )

        # ---------------------------------------------------------------
        # Linee tratteggiate STD per-provino (solo modello heating)
        # ---------------------------------------------------------------
        if show_std_tangent and FIT_MODEL == "heating":
            t_star = compute_tangency(res["A"], res["alpha"], rate_std_s, t_extrap.max())

            if t_star is not None:
                # --- CASO NORMALE: t* esiste ---
                T_star  = res["T0"] + res["A"] * (1.0 - np.exp(-res["alpha"] * t_star))
                c_shift = T_star - rate_std_s * t_star

                t_line = np.linspace(t_star, x_max, 300)
                T_line = rate_std_s * t_line + c_shift

                ax.plot(t_line, T_line,
                        color=color, linewidth=1.4, linestyle="--",
                        alpha=0.55, zorder=8)
                ax.scatter(t_star, T_star,
                           color=color, s=90, marker="x",
                           linewidths=2.0, zorder=9)

                if not normal_tangent_added:
                    legend_handles.append(
                        mlines.Line2D([], [], color="gray", linewidth=1.4,
                                      linestyle="--", alpha=0.7,
                                      label="proiezione rate STD (" + str(RATE_STD_PER_LAYER) + " \u00b0C/layer)")
                    )
                    legend_handles.append(
                        mlines.Line2D([], [], color="gray", marker="x",
                                      markersize=7, linewidth=0,
                                      markeredgewidth=2.0,
                                      label="punto di tangenza (dT/dt = rate STD)")
                    )
                    normal_tangent_added = True

            else:
                # --- CASO ANOMALO: pendenza max < rate_std_s ---
                # La curva non raggiunge mai il regime STD
                t_end   = t_extrap.max()
                T_end   = res["T0"] + res["A"] * (1.0 - np.exp(-res["alpha"] * t_end))
                c_shift = T_end - rate_std_s * t_end

                t_extra_end = t_end + EXTRA_FRAC * t_max_all
                t_line = np.linspace(t_end, t_extra_end, 200)
                T_line = rate_std_s * t_line + c_shift

                ax.plot(t_line, T_line,
                        color=color, linewidth=1.4, linestyle="--",
                        alpha=0.55, zorder=8)
                ax.scatter(t_end, T_end,
                           color=color, s=100, marker=">",
                           linewidths=1.5, zorder=9, edgecolors=color)

                if not anomaly_tangent_added:
                    legend_handles.append(
                        mlines.Line2D([], [], color="gray", linewidth=1.4,
                                      linestyle="--", alpha=0.7,
                                      marker=">", markersize=6,
                                      label="proiezione STD: pendenza max < rate STD")
                    )
                    anomaly_tangent_added = True

                print("[WARN] " + specimen + ": pendenza max (" +
                      f"{res['A'] * res['alpha']:.4f}" + " \u00b0C/s) < rate_std_s (" +
                      f"{rate_std_s:.4f}" + " \u00b0C/s) — caso anomalo")

    # Forza asse X al valore calcolato
    ax.set_xlim(left=ax.get_xlim()[0], right=x_max * 1.02)

    return legend_handles


# ===========================================================================
# FASE 3: grafico GLOBALE (tutti i provini, senza linee STD)
# ===========================================================================

fig_g, ax_g = plt.subplots(figsize=(13, 7))
ax_g.axvline(0, color="#555555", linewidth=1.0, linestyle=":", alpha=0.7)
ax_g.text(0.5, 0.01, "restart", fontsize=8, color="#555555",
          transform=ax_g.get_xaxis_transform(), va="bottom")

all_ok = sorted(specimen_data.keys())
legend_handles_g = plot_specimens(ax_g, all_ok, show_std_tangent=False)

ordered_g = []
for pause in ["10s", "30s", "60s", "90s", "std"]:
    if pause == "std" and not INCLUDE_STD_SPECIMENS:
        continue
    grp = [h for h in legend_handles_g
           if get_pause_group(h.get_label().split("  ")[0]) == pause]
    if grp:
        ordered_g.append(
            mlines.Line2D([], [], color="none",
                          label="\u25a0 Pausa " + pause, linewidth=0)
        )
        ordered_g.extend(grp)
ordered_g.append(
    mlines.Line2D([], [], color="gray", marker="*", markersize=8,
                  linewidth=0, label="T\u2080 stimata a t=0")
)

title_metric = {"mean": "media", "max": "massima", "min": "minima"}[PLOT_METRIC]
fit_labels = {
    "heating"  : "riscaldamento WLS  T(t)=T\u2080+A\u00b7(1\u2212e^(\u2212\u03b1t))",
    "linear"   : "lineare OLS",
    "quadratic": "quadratico OLS",
}
fit_info = fit_labels[FIT_MODEL]

ax_g.set_title(
    "Temperatura " + title_metric + " ROI per layer + estrapolazione a t=0  [tutti i provini]\n"
    "Dataset: " + DATASET_NAME + "  |  Modello: " + fit_info,
    fontsize=12, pad=12
)
ax_g.set_xlabel("Tempo dal restart [s]", fontsize=11)
ax_g.set_ylabel(metric_label, fontsize=11)
ax_g.grid(True, linestyle="--", alpha=0.4, linewidth=0.7)
ax_g.spines[["top", "right"]].set_visible(False)
ax_g.legend(handles=ordered_g, loc="lower right", fontsize=8,
            framealpha=0.85, edgecolor="#cccccc", ncol=1)

plt.tight_layout()
out_path_g = output_dir / output_fname
fig_g.savefig(out_path_g, dpi=150, bbox_inches="tight")
print("\n[OK] Grafico globale: " + str(out_path_g))
plt.show()
plt.close(fig_g)


# ===========================================================================
# FASE 4: grafici PER FAMIGLIA con linee STD tratteggiate per-provino
# ===========================================================================

for pause_family in ["10s", "30s", "60s", "90s"]:
    family_specimens = [
        s for s in sorted(specimen_data.keys())
        if specimen_data[s]["pause"] == pause_family
    ]
    if not family_specimens:
        continue

    fig_f, ax_f = plt.subplots(figsize=(12, 7))
    ax_f.axvline(0, color="#555555", linewidth=1.0, linestyle=":", alpha=0.7)
    ax_f.text(0.5, 0.01, "restart", fontsize=8, color="#555555",
              transform=ax_f.get_xaxis_transform(), va="bottom")

    legend_handles_f = plot_specimens(
        ax_f, family_specimens, show_std_tangent=(FIT_MODEL == "heating")
    )

    ax_f.set_title(
        "Famiglia pausa " + pause_family + "  \u2014  Temperatura " + title_metric + " ROI per layer\n"
        "Dataset: " + DATASET_NAME + "  |  Modello: " + fit_info
        + "  |  linea tratteggiata = proiezione rate STD (" + str(RATE_STD_PER_LAYER) + " \u00b0C/layer)",
        fontsize=11, pad=12
    )
    ax_f.set_xlabel("Tempo dal restart [s]", fontsize=11)
    ax_f.set_ylabel(metric_label, fontsize=11)
    ax_f.grid(True, linestyle="--", alpha=0.4, linewidth=0.7)
    ax_f.spines[["top", "right"]].set_visible(False)
    ax_f.legend(handles=legend_handles_f, loc="lower right", fontsize=8,
                framealpha=0.85, edgecolor="#cccccc", ncol=1)

    plt.tight_layout()
    fname_family = f"{file_prefix}_{FIT_MODEL}_{pause_family}.png"
    out_path_f   = output_dir / fname_family
    fig_f.savefig(out_path_f, dpi=150, bbox_inches="tight")
    print("[OK] Grafico famiglia " + pause_family + ": " + str(out_path_f))
    plt.show()
    plt.close(fig_f)

print("\n[DONE] Tutti i grafici salvati in: " + str(output_dir))
