"""
plot_thermal_drift.py
=====================
Visualizes the layer-mean ROI temperature in chronological order
for each STD specimen, to inspect the thermal drift hypothesis.

For each dataset and each specimen:
  - One subplot per specimen showing mean ROI temperature vs layer index
  - Points coloured by layer index (cool→warm colormap = early→late layer)
  - Linear regression trend line overlaid
  - Pearson r and p-value annotated

Filters applied (same as test_indipendenza.py):
  1. frame_type == 'core'
  2. roi_complete == True

Outputs saved in: test_ipotesi/risultati/
  - thermal_drift_<dataset>.png   : one figure per dataset

Usage:
    python test_ipotesi/plot_thermal_drift.py
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from scipy import stats

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(SCRIPT_DIR, "..", "..", "creazione_dataset", "datasets")
RISULTATI_DIR = os.path.join(SCRIPT_DIR, "risultati")
os.makedirs(RISULTATI_DIR, exist_ok=True)

STD_FILES = {
    "ROI_wide_2_6_depth_1_2":   "ROI_wide_2_6_depth_1_2_STD.csv",
    "ROI_wide_3_7_depth_1_2":   "ROI_wide_3_7_depth_1_2_STD.csv",
    "ROI_wide_3_10_depth_1_3":  "ROI_wide_3_10_depth_1_3_STD.csv",
}

TEMP_COL  = "roi_mean_C"
COLORMAP  = "plasma"   # cool (early layers) → warm (late layers)


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def load_and_filter(filepath):
    df = pd.read_csv(filepath)
    if "frame_type" in df.columns:
        df = df[df["frame_type"] == "core"].copy()
    if "roi_complete" in df.columns:
        df = df[df["roi_complete"] == True].copy()
    return df


def layer_series(df, specimen):
    """Return (layer_indices, mean_temps) arrays for one specimen."""
    sub = df[df["specimen"] == specimen]
    grp = sub.groupby("layer_index")[TEMP_COL].mean().sort_index()
    return grp.index.to_numpy(), grp.values


# ─────────────────────────────────────────────
# MAIN PLOT
# ─────────────────────────────────────────────
def plot_drift(dataset_name, filename):
    filepath = os.path.join(DATASET_DIR, filename)
    if not os.path.exists(filepath):
        print(f"[SKIP] File not found: {filepath}")
        return

    df = load_and_filter(filepath)
    specimens = sorted(df["specimen"].unique())
    n_spec = len(specimens)

    fig, axes = plt.subplots(
        1, n_spec,
        figsize=(7 * n_spec, 5),
        squeeze=False
    )

    fig.suptitle(
        f"Thermal Drift — Layer-mean ROI Temperature\n"
        f"Dataset: {dataset_name}   |   filters: core frames + roi_complete",
        fontsize=12, y=1.03
    )

    # Shared colormap normalised over all layers present in this dataset
    all_layers = sorted(df["layer_index"].unique())
    norm = mcolors.Normalize(vmin=min(all_layers), vmax=max(all_layers))
    cmap = cm.get_cmap(COLORMAP)

    for col, spec in enumerate(specimens):
        ax = axes[0][col]
        layers, temps = layer_series(df, spec)

        if len(layers) < 2:
            ax.set_title(f"{spec}\n(insufficient data)")
            ax.axis("off")
            continue

        # ── Scatter: coloured by layer index ──
        scatter_colors = [cmap(norm(l)) for l in layers]
        sc = ax.scatter(
            layers, temps,
            c=layers, cmap=COLORMAP, norm=norm,
            s=60, zorder=3, edgecolors="white", linewidths=0.4
        )

        # ── Linear trend line ──
        slope, intercept, r, p, _ = stats.linregress(layers, temps)
        x_fit = np.array([layers.min(), layers.max()])
        y_fit = slope * x_fit + intercept
        ax.plot(
            x_fit, y_fit,
            color="crimson", linewidth=1.8, linestyle="--",
            label=f"trend: {slope:+.3f} °C/layer"
        )

        # ── Annotation ──
        p_str = f"{p:.4f}" if p >= 0.0001 else "<0.0001"
        ax.annotate(
            f"Pearson r = {r:.3f}\np = {p_str}\nslope = {slope:+.3f} °C/layer",
            xy=(0.05, 0.95), xycoords="axes fraction",
            va="top", ha="left", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7)
        )

        # ── Formatting ──
        ax.set_title(f"{spec}  (N layers = {len(layers)})", fontsize=10)
        ax.set_xlabel("Layer index (chronological)", fontsize=9)
        ax.set_ylabel("Mean ROI temperature [°C]", fontsize=9)
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(True, linestyle=":", alpha=0.4)
        ax.tick_params(labelsize=8)

    # ── Shared colorbar ──
    sm = cm.ScalarMappable(cmap=COLORMAP, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes[0], orientation="vertical",
                        fraction=0.03, pad=0.04)
    cbar.set_label("Layer index", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    plt.tight_layout()
    out = os.path.join(RISULTATI_DIR, f"thermal_drift_{dataset_name}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def main():
    print("Generating thermal drift plots...")
    for dataset_name, filename in STD_FILES.items():
        print(f"\nDataset: {dataset_name}")
        plot_drift(dataset_name, filename)
    print("\nDone.")


if __name__ == "__main__":
    main()
