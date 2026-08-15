import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
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
    raise FileNotFoundError("Folder 'creazione_dataset' not found.")


CONFIG_DIR = None
for parent in Path(__file__).resolve().parents:
    if (parent / "config.py").exists():
        CONFIG_DIR = parent
        break
if CONFIG_DIR is None:
    raise FileNotFoundError("config.py not found.")


ANALISI_DIR = SCRIPT_DIR.parent / "Analisi_dataset"


sys.path.insert(0, str(CONFIG_DIR))
sys.path.insert(0, str(CREAZIONE_DS))


from frame_selector import SPECIMENS  # noqa: E402


# ===========================================================================
# CONFIG
# ===========================================================================


DATASET_NAME = "ROI_wide_3_10_depth_1_4"  # <--- change here to switch ROI
LEGENDA = False  # True = show legend, False = hide legend


RATE_STD_PER_LAYER = 0.240
DT_LAYER_OVERRIDE = None
SIGMA_FLOOR = 0.1
SHOW_CONFIDENCE = True    # show +/-1sigma band on the fit


# Thresholds to detect degenerate fits
DEGEN_T0_FACTOR = 3.0     # |T0_fit| > DEGEN_T0_FACTOR * max(T_obs) => degenerate
DEGEN_SIGMA_T0 = 50.0     # sigma_T0 > threshold [degC] => degenerate
Y_MARGIN_DEG = 5.0        # extra margin on Y axis above/below observed data


ALWAYS_EXCLUDE = {"Rec-023"}
STD_SPECIMENS = {"Rec-027_std_2", "Rec-G3_std_1"}
INCLUDE_STD = False


# ===========================================================================
# END CONFIG
# ===========================================================================


DATASETS_DIR = CREAZIONE_DS / "datasets"
dataset_path = DATASETS_DIR / f"{DATASET_NAME}.csv"


OUTPUT_DIR = ANALISI_DIR / DATASET_NAME / "grafici_extrapolation_pixel"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def fit_heating_wls(t_pts, y_pts, s_pts, t_extrap):
    """Fit WLS exponential model. Returns dict with curve and parameters."""
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
        return {"success": False, "error": str(e)}

    T0_fit, A_fit, alpha_fit = popt
    perr = np.sqrt(np.diag(pcov))

    # --- Degenerate fit check ---
    T_obs_max = float(np.nanmax(np.abs(y_pts)))
    if abs(T0_fit) > DEGEN_T0_FACTOR * T_obs_max or perr[0] > DEGEN_SIGMA_T0:
        return {
            "success": False,
            "error": f"degenerate fit: T0={T0_fit:.1f} sigma_T0={perr[0]:.1f}",
        }

    y_curve = model(t_extrap, T0_fit, A_fit, alpha_fit)

    # Confidence band (first-order error propagation)
    exp_at = np.exp(-alpha_fit * t_extrap)
    dT_dT0 = np.ones_like(t_extrap)
    dT_dA = 1.0 - exp_at
    dT_dalpha = A_fit * t_extrap * exp_at
    var_curve = (
        dT_dT0**2 * pcov[0, 0]
        + dT_dA**2 * pcov[1, 1]
        + dT_dalpha**2 * pcov[2, 2]
        + 2 * dT_dT0 * dT_dA * pcov[0, 1]
        + 2 * dT_dT0 * dT_dalpha * pcov[0, 2]
        + 2 * dT_dA * dT_dalpha * pcov[1, 2]
    )
    sigma_curve = np.sqrt(np.maximum(var_curve, 0.0))

    return {
        "success": True,
        "T0": float(T0_fit),
        "A": float(A_fit),
        "alpha": float(alpha_fit),
        "T_inf": float(T0_fit + A_fit),
        "sigma_T0": float(perr[0]),
        "sigma_A": float(perr[1]),
        "sigma_alpha": float(perr[2]),
        "y_curve": y_curve,
        "y_upper": y_curve + sigma_curve,
        "y_lower": y_curve - sigma_curve,
    }


def parse_roi_pixels(raw_str, n_pixels_expected=None):
    try:
        vals = np.array([float(v) for v in str(raw_str).split(",") if v.strip()])
        if len(vals) == 0:
            return None
        if n_pixels_expected is not None and len(vals) != n_pixels_expected:
            if abs(len(vals) - n_pixels_expected) > max(2, n_pixels_expected * 0.2):
                return None
            if len(vals) < n_pixels_expected:
                padded = np.full(n_pixels_expected, np.nan)
                padded[:len(vals)] = vals
                return padded
            else:
                return vals[:n_pixels_expected]
        return vals
    except Exception:
        return None


def detect_roi_shape(df_sample):
    roi_cols = {"roi_c0", "roi_c1", "roi_r0", "roi_r1"}
    if roi_cols.issubset(df_sample.columns):
        row = df_sample.iloc[0]
        n_cols = int(row["roi_c1"] - row["roi_c0"] + 1)
        n_rows = int(row["roi_r1"] - row["roi_r0"] + 1)
        if n_cols > 0 and n_rows > 0:
            return n_rows, n_cols
    for raw in df_sample["roi_pixels_raw"].dropna():
        vals = parse_roi_pixels(raw)
        if vals is not None and len(vals) > 0:
            return 1, len(vals)
    return None, None


# ---------------------------------------------------------------------------
# Load dataset
# ---------------------------------------------------------------------------
print(f"[INFO] Loading dataset: {dataset_path}")
df_full = pd.read_csv(dataset_path)

df = df_full[
    (df_full["frame_type"] == "core") &
    (df_full["roi_valid_frac"] >= 1.0)
].copy()
print(f"[INFO] Rows after filter: {len(df)}")


n_rows_roi, n_cols_roi = detect_roi_shape(df)
if n_rows_roi is None:
    raise ValueError("Unable to infer ROI dimensions from the dataset.")
n_pixels_total = n_rows_roi * n_cols_roi
print(f"[INFO] ROI shape detected: {n_rows_roi} rows x {n_cols_roi} columns = {n_pixels_total} pixels")


# ---------------------------------------------------------------------------
# Estimate dt_layer
# ---------------------------------------------------------------------------
if DT_LAYER_OVERRIDE is not None:
    dt_layer = float(DT_LAYER_OVERRIDE)
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
    print(f"[INFO] Estimated dt_layer: {dt_layer:.3f} s")


rate_std_s = RATE_STD_PER_LAYER / dt_layer


# ---------------------------------------------------------------------------
# Shared colormap (same used in ROI map)
# ---------------------------------------------------------------------------
cmap = matplotlib.colormaps.get_cmap("tab20") if hasattr(cm, '__module__') else cm.get_cmap("tab20")
try:
    import matplotlib
    cmap = matplotlib.colormaps.get_cmap("tab20")
except Exception:
    cmap = cm.get_cmap("tab20", n_pixels_total)


pixel_colors = {
    (r, c): cmap(r * n_cols_roi + c)
    for r in range(n_rows_roi)
    for c in range(n_cols_roi)
}


# ---------------------------------------------------------------------------
# Loop specimens => 1 plot per specimen
# ---------------------------------------------------------------------------
specimens_list = sorted(df["specimen"].unique())


for specimen in specimens_list:
    if specimen in ALWAYS_EXCLUDE:
        continue
    if specimen in STD_SPECIMENS and not INCLUDE_STD:
        continue

    spec_info = SPECIMENS.get(specimen)
    if spec_info is None:
        continue

    restart_frame = spec_info.get("restart_frame")
    layers_front = spec_info.get("layers_front")
    if restart_frame is None or layers_front is None:
        continue

    df_spec = df[df["specimen"] == specimen]
    if df_spec.empty:
        continue

    n_layers_spec = len(layers_front)
    T_px_layer = np.full((n_pixels_total, n_layers_spec), np.nan)
    S_px_layer = np.full((n_pixels_total, n_layers_spec), np.nan)
    t_layer = np.full(n_layers_spec, np.nan)

    for k_idx, (fs, fe) in enumerate(layers_front):
        mask = (df_spec["frame_idx"] >= fs) & (df_spec["frame_idx"] <= fe)
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
        S_px_layer[:, k_idx] = np.nanstd(px_matrix, axis=0, ddof=1) if px_matrix.shape[0] > 1 else 0.0

    valid_layers = ~np.isnan(t_layer)
    if valid_layers.sum() < 3:
        print(f"  [SKIP] {specimen}: only {valid_layers.sum()} valid layers")
        continue

    t_pts_spec = t_layer[valid_layers]
    T_px_valid = T_px_layer[:, valid_layers]
    S_px_valid = S_px_layer[:, valid_layers]
    t_extrap = np.linspace(0.0, t_pts_spec.max() * 1.05, 400)

    # Y limits from observed data (all valid pixels)
    all_obs = T_px_valid[~np.all(np.isnan(T_px_valid), axis=1)]
    y_obs_min = float(np.nanmin(all_obs)) - Y_MARGIN_DEG
    y_obs_max = float(np.nanmax(all_obs)) + Y_MARGIN_DEG

    fig, ax = plt.subplots(figsize=(11, 7))
    ax.axvline(0, color="#555555", linewidth=1.0, linestyle=":", alpha=0.7)
    ax.text(0.5, 0.01, "restart", fontsize=8, color="#555555",
            transform=ax.get_xaxis_transform(), va="bottom")

    legend_handles = []
    n_fit_ok = 0
    n_fit_fail = 0

    for px_id in range(n_pixels_total):
        row_px = px_id // n_cols_roi
        col_px = px_id % n_cols_roi
        color = pixel_colors[(row_px, col_px)]

        y_px = T_px_valid[px_id, :]
        s_px = S_px_valid[px_id, :]

        if np.all(np.isnan(y_px)):
            n_fit_fail += 1
            continue

        res = fit_heating_wls(t_pts_spec, y_px, s_px, t_extrap)

        if not res["success"]:
            n_fit_fail += 1
            ax.scatter(t_pts_spec, y_px, color=color, s=30, alpha=0.6,
                       marker="x", zorder=3, linewidths=1.5)
            lbl = f"px({row_px},{col_px}) [fit failed]"
            if LEGENDA:
                legend_handles.append(
                    mlines.Line2D([], [], color=color, marker="x", markersize=6,
                                  linewidth=0, label=lbl)
                )
            continue

        n_fit_ok += 1

        ax.plot(t_extrap, res["y_curve"], color=color, linewidth=1.5, alpha=0.85, zorder=4)

        if SHOW_CONFIDENCE:
            ax.fill_between(t_extrap, res["y_lower"], res["y_upper"],
                            color=color, alpha=0.08, linewidth=0, zorder=2)

        ax.scatter(t_pts_spec, y_px, color=color, s=45, zorder=5,
                   edgecolors="white", linewidths=0.5)
        ax.errorbar(t_pts_spec, y_px, yerr=s_px, fmt="none",
                    ecolor=color, elinewidth=0.9, capsize=2, alpha=0.45, zorder=3)
        ax.scatter(0, res["T0"], color=color, s=80, marker="*",
                   zorder=6, edgecolors="white", linewidths=0.5)

        lbl = (
            f"px({row_px},{col_px})  "
            f"T0={res['T0']:.1f}C  "
            f"alpha={res['alpha']:.4f}  "
            f"Tinf={res['T_inf']:.1f}C"
        )
        if LEGENDA:
            legend_handles.append(
                mlines.Line2D([], [], color=color, marker="o", markersize=5,
                              linewidth=1.5, label=lbl)
            )

    # Clip Y axis to observed data range — ignore out-of-range fit values
    ax.set_ylim(y_obs_min, y_obs_max)

    ax.set_title(
        f"Pixel-wise extrapolation — {specimen}\n"
        f"Dataset: {DATASET_NAME}  |  ROI: {n_rows_roi}x{n_cols_roi} px  |  "
        f"Fit OK: {n_fit_ok}/{n_pixels_total}",
        fontsize=11, pad=10
    )
    ax.set_xlabel("Time from restart [s]", fontsize=11)
    ax.set_ylabel("Temperature [C]", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.35, linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)

    if LEGENDA and legend_handles:
        ax.legend(handles=legend_handles, loc="lower right", fontsize=8,
                  framealpha=0.85, edgecolor="#cccccc", ncol=1)

    plt.tight_layout()
    out_path = OUTPUT_DIR / f"LayerExtrap_Pixel_{specimen}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  [OK]   {specimen}: {out_path.name}  (fit OK: {n_fit_ok}, fail: {n_fit_fail})")
    plt.close(fig)

print(f"\n[DONE] Plots saved in: {OUTPUT_DIR}")
print(f"  Total specimens: {len([s for s in specimens_list if s not in ALWAYS_EXCLUDE and (s not in STD_SPECIMENS or INCLUDE_STD)])}")


# ---------------------------------------------------------------------------
# Generate ROI map automatically in the same folder
# ---------------------------------------------------------------------------
print("\n[INFO] Generating ROI map...")


try:
    row0 = df.iloc[0]
    roi_c0 = int(row0["roi_c0"])
    roi_c1 = int(row0["roi_c1"])
    roi_r0 = int(row0["roi_r0"])
    roi_r1 = int(row0["roi_r1"])
    n_cols_m = roi_c1 - roi_c0 + 1
    n_rows_m = roi_r1 - roi_r0 + 1
    n_pixels_m = n_rows_m * n_cols_m

    nozzle_x_abs = df["tip_x"].median()
    nozzle_y_abs = df["tip_y"].median()
    nozzle_col_rel = nozzle_x_abs - roi_c0
    nozzle_row_rel = nozzle_y_abs - roi_r0

    CELL_SIZE = 1.6
    fig_w = max(n_cols_m * CELL_SIZE + 3.0, 7.0)
    fig_h = max(n_rows_m * CELL_SIZE + 3.0, 5.0)
    fig_m, ax_m = plt.subplots(figsize=(fig_w, fig_h))
    ax_m.set_aspect("equal")
    ax_m.set_facecolor("#f5f5f5")

    for r in range(n_rows_m):
        for c in range(n_cols_m):
            px_id = r * n_cols_m + c
            color = cmap(px_id)
            x0 = c
            y0 = n_rows_m - 1 - r
            rect = mpatches.FancyBboxPatch(
                (x0 + 0.04, y0 + 0.04), 0.92, 0.92,
                boxstyle="round,pad=0.04",
                linewidth=1.5, edgecolor="white",
                facecolor=color, alpha=0.85, zorder=2,
            )
            ax_m.add_patch(rect)
            ax_m.text(x0 + 0.5, y0 + 0.62, f"px({r},{c})",
                      ha="center", va="center", fontsize=11, fontweight="bold",
                      color="white", zorder=4)
            ax_m.text(x0 + 0.5, y0 + 0.30, f"id={px_id}",
                      ha="center", va="center", fontsize=8.5,
                      color="white", alpha=0.90, zorder=4)

    outer = mpatches.FancyBboxPatch(
        (0, 0), n_cols_m, n_rows_m,
        boxstyle="square,pad=0", linewidth=2.5,
        edgecolor="#333333", facecolor="none", zorder=5,
    )
    ax_m.add_patch(outer)

    nozzle_ax_x = nozzle_col_rel
    nozzle_ax_y = n_rows_m - nozzle_row_rel

    ax_m.plot(
        nozzle_ax_x, nozzle_ax_y,
        marker="v", markersize=18, color="#c0392b",
        markeredgecolor="white", markeredgewidth=1.5,
        zorder=8, clip_on=False,
        label=f"Nozzle tip\n(col={nozzle_col_rel:.1f}, row={nozzle_row_rel:.1f} px from ROI)",
    )
    if nozzle_ax_y > n_rows_m:
        ax_m.plot([nozzle_ax_x, nozzle_ax_x], [n_rows_m, nozzle_ax_y],
                  color="#c0392b", linewidth=1.2, linestyle="--",
                  alpha=0.6, zorder=7, clip_on=False)
    elif nozzle_ax_y < 0:
        ax_m.plot([nozzle_ax_x, nozzle_ax_x], [nozzle_ax_y, 0],
                  color="#c0392b", linewidth=1.2, linestyle="--",
                  alpha=0.6, zorder=7, clip_on=False)

    ax_m.annotate(
        f"Nozzle\n({nozzle_col_rel:+.1f}, {nozzle_row_rel:+.1f}) px",
        xy=(nozzle_ax_x, nozzle_ax_y),
        xytext=(nozzle_ax_x + 0.5, nozzle_ax_y + (1.0 if nozzle_ax_y >= n_rows_m else -1.0)),
        fontsize=9, color="#c0392b",
        arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.2),
        zorder=9, clip_on=False,
    )

    ax_m.set_xlim(-0.5, n_cols_m + 0.5)
    ax_m.set_ylim(-0.5, n_rows_m + 1.2)
    ax_m.set_xticks(np.arange(n_cols_m) + 0.5)
    ax_m.set_xticklabels(
        [f"col {roi_c0+c}\n(roi col {c})" for c in range(n_cols_m)], fontsize=8
    )
    ax_m.set_yticks(np.arange(n_rows_m) + 0.5)
    ax_m.set_yticklabels(
        [f"row {roi_r0+(n_rows_m-1-r)}\n(roi row {n_rows_m-1-r})" for r in range(n_rows_m)],
        fontsize=8
    )
    ax_m.tick_params(axis="both", length=0)
    for spine in ax_m.spines.values():
        spine.set_visible(False)
    for c in range(n_cols_m + 1):
        ax_m.axvline(c, color="#cccccc", linewidth=0.5, zorder=1)
    for r in range(n_rows_m + 1):
        ax_m.axhline(r, color="#cccccc", linewidth=0.5, zorder=1)

    ax_m.set_title(
        f"ROI map — {DATASET_NAME}\n"
        f"Shape: {n_rows_m} rows x {n_cols_m} columns = {n_pixels_m} pixels  |  "
        f"Absolute ROI: col [{roi_c0},{roi_c1}], row [{roi_r0},{roi_r1}]",
        fontsize=12, pad=14,
    )
    ax_m.legend(loc="upper left", fontsize=9, framealpha=0.85, edgecolor="#cccccc")
    fig_m.text(
        0.5, 0.01,
        "Convention: pixel_id = row * n_cols + col  (row-major, row 0 = top of ROI)",
        ha="center", fontsize=8, color="#888888",
    )
    plt.tight_layout(rect=[0, 0.03, 1, 1])

    map_path = OUTPUT_DIR / f"ROI_map_{DATASET_NAME}.png"
    fig_m.savefig(map_path, dpi=150, bbox_inches="tight")
    plt.close(fig_m)
    print(f"[OK] ROI map saved: {map_path}")


except Exception as e:
    print(f"[WARN] ROI map not generated: {e}")