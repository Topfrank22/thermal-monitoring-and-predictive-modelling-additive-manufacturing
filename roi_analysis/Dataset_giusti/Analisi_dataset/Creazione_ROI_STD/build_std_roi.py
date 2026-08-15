"""
build_std_roi.py
================
Builds the Standard ROI reference from STD specimens using linear regression
to remove the thermal drift across layers.

Outputs (all saved in the same folder as this script):
  std_roi_summary.txt          - human-readable summary
  std_roi_data.npz             - numpy archive (for other scripts)
  plot_linear_model.png        - linear fit of mean STD temperature vs layer index
  plot_std_roi_mean.png        - mean STD ROI (drift-removed) as a thermal heatmap
  plot_std_roi_std.png         - pixel-wise std dev
  plot_std_roi_max.png         - pixel-wise max (drift-removed)
  plot_std_roi_min.png         - pixel-wise min (drift-removed)

Usage:
  python build_std_roi.py

Change only DATASET_NAME in the CONFIG section to switch ROI geometry.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

# ============================================================
#  CONFIG
# ============================================================
DATASET_NAME = "ROI_wide_2_6_depth_1_2"

DATASETS_DIR = Path(r"C:\Users\tomga\Desktop\Lab Data science\Dataset_giusti\creazione_dataset\datasets")
OUTPUT_DIR   = Path(__file__).parent
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
# ============================================================


# ------------------------------------------------------------------
# 1. Load data
# ------------------------------------------------------------------
std_csv = DATASETS_DIR / f"{DATASET_NAME}_STD.csv"

if not std_csv.exists():
    raise FileNotFoundError(
        f"STD dataset not found: {std_csv}\n"
        f"Expected: {DATASET_NAME}_STD.csv in {DATASETS_DIR}"
    )

df = pd.read_csv(std_csv)
df_core = df[
    (df["frame_type"] == "core") &
    (df["roi_complete"] == True)
].copy()

if len(df_core) == 0:
    raise ValueError(
        f"No core+complete rows found in {std_csv.name}.\n"
        f"Specimens in file: {sorted(df['specimen'].unique().tolist())}"
    )

print(f"Loaded {len(df_core)} core+complete rows from {len(df_core['specimen'].unique())} STD specimens")
print(f"  CSV      : {std_csv.name}")
print(f"  Specimens: {sorted(df_core['specimen'].unique().tolist())}")
print(f"  Layers   : {sorted(df_core['layer_index'].unique().tolist())}")


# ------------------------------------------------------------------
# 2. Parse roi_pixels_raw
# ------------------------------------------------------------------
def parse_pixels(raw_str):
    return np.array([float(v) for v in str(raw_str).split(",") if v.strip() != ""])

df_core["pixels"] = df_core["roi_pixels_raw"].apply(parse_pixels)

# ------------------------------------------------------------------
# Infer ROI shape directly from coordinate columns
# ------------------------------------------------------------------
sample = df_core.iloc[0]
n_pixels = len(sample["pixels"])

ROI_H = int(sample["roi_r1"]) - int(sample["roi_r0"])
ROI_W = int(sample["roi_c1"]) - int(sample["roi_c0"])

# If shape doesn't match pixel count, fall back to 1 x n_pixels
if ROI_H * ROI_W != n_pixels:
    ROI_H = 1
    ROI_W = n_pixels

print(f"ROI shape: {ROI_H} rows x {ROI_W} cols = {ROI_H * ROI_W} pixels")

assert ROI_H * ROI_W == n_pixels, (
    f"Shape mismatch: {ROI_H}x{ROI_W}={ROI_H*ROI_W} != {n_pixels} pixels in data."
)


# ------------------------------------------------------------------
# 3. Linear regression: roi_mean_C ~ layer_index
# ------------------------------------------------------------------
layer_vals = df_core["layer_index"].values.astype(float)
mean_vals  = df_core["roi_mean_C"].values

slope, intercept, r_value, p_value, std_err = stats.linregress(layer_vals, mean_vals)
print(f"\nLinear model: T = {intercept:.4f} + {slope:.4f} * layer")
print(f"  r = {r_value:.4f},  p = {p_value:.4e},  std_err = {std_err:.6f}")


# ------------------------------------------------------------------
# 4. Drift correction (relative to layer 1)
# ------------------------------------------------------------------
T_hat_layer1 = intercept + slope * 1.0
df_core["drift"]            = slope * (df_core["layer_index"] - 1.0)
df_core["mean_corrected"]   = df_core["roi_mean_C"] - df_core["drift"]
df_core["pixels_corrected"] = df_core.apply(
    lambda row: row["pixels"] - row["drift"], axis=1
)


# ------------------------------------------------------------------
# 5. Pixel-wise aggregate statistics
# ------------------------------------------------------------------
all_pixels = np.vstack(df_core["pixels_corrected"].values)

std_mean_flat = all_pixels.mean(axis=0)
std_std_flat  = all_pixels.std(axis=0, ddof=1)
std_max_flat  = all_pixels.max(axis=0)
std_min_flat  = all_pixels.min(axis=0)

std_mean_map = std_mean_flat.reshape(ROI_H, ROI_W)
std_std_map  = std_std_flat.reshape(ROI_H, ROI_W)
std_max_map  = std_max_flat.reshape(ROI_H, ROI_W)
std_min_map  = std_min_flat.reshape(ROI_H, ROI_W)

global_mean = float(std_mean_flat.mean())
global_std  = float(std_std_flat.mean())
global_max  = float(std_max_flat.max())
global_min  = float(std_min_flat.min())

print(f"\nDrift-corrected STD ROI summary:")
print(f"  Global mean temperature : {global_mean:.3f} °C")
print(f"  Mean pixel-wise std dev : {global_std:.3f} °C")
print(f"  Global max temperature  : {global_max:.3f} °C")
print(f"  Global min temperature  : {global_min:.3f} °C")


# ------------------------------------------------------------------
# 6. Save machine-readable archive
# ------------------------------------------------------------------
np.savez(
    OUTPUT_DIR / "std_roi_data.npz",
    std_mean_map      = std_mean_map,
    std_std_map       = std_std_map,
    std_max_map       = std_max_map,
    std_min_map       = std_min_map,
    regression_params = np.array([intercept, slope, r_value, p_value, std_err]),
    roi_shape         = np.array([ROI_H, ROI_W]),
    T_hat_layer1      = np.array([T_hat_layer1]),
)
print(f"\nSaved: std_roi_data.npz")


# ------------------------------------------------------------------
# 7. Save human-readable summary
# ------------------------------------------------------------------
n_frames = len(df_core)
n_spec   = len(df_core["specimen"].unique())
with open(OUTPUT_DIR / "std_roi_summary.txt", "w") as f:
    f.write("=" * 60 + "\n")
    f.write("STD ROI REFERENCE SUMMARY\n")
    f.write(f"Dataset       : {DATASET_NAME}_STD\n")
    f.write(f"ROI shape     : {ROI_H} rows x {ROI_W} cols\n")
    f.write(f"STD specimens : {n_spec}\n")
    f.write(f"Frames used   : {n_frames} (core + roi_complete)\n")
    f.write("=" * 60 + "\n\n")
    f.write("LINEAR DRIFT MODEL\n")
    f.write(f"  T_hat(layer) = {intercept:.4f} + {slope:.4f} * layer_index\n")
    f.write(f"  Pearson r    = {r_value:.4f}\n")
    f.write(f"  p-value      = {p_value:.4e}\n")
    f.write(f"  Std error    = {std_err:.6f}\n")
    f.write(f"  T_hat(1)     = {T_hat_layer1:.4f} °C  (reference level)\n\n")
    f.write("DRIFT-CORRECTED STD ROI STATISTICS\n")
    f.write(f"  Global mean temperature  : {global_mean:.3f} °C\n")
    f.write(f"  Mean pixel-wise std dev  : {global_std:.3f} °C\n")
    f.write(f"  Global max temperature   : {global_max:.3f} °C\n")
    f.write(f"  Global min temperature   : {global_min:.3f} °C\n\n")
    f.write("PIXEL-WISE MEAN MAP (rows x cols) [°C]\n")
    for row in std_mean_map:
        f.write("  " + "  ".join(f"{v:7.3f}" for v in row) + "\n")
    f.write("\nPIXEL-WISE STD MAP (rows x cols) [°C]\n")
    for row in std_std_map:
        f.write("  " + "  ".join(f"{v:7.3f}" for v in row) + "\n")
    f.write("\nPIXEL-WISE MAX MAP (rows x cols) [°C]\n")
    for row in std_max_map:
        f.write("  " + "  ".join(f"{v:7.3f}" for v in row) + "\n")
    f.write("\nPIXEL-WISE MIN MAP (rows x cols) [°C]\n")
    for row in std_min_map:
        f.write("  " + "  ".join(f"{v:7.3f}" for v in row) + "\n")
print("Saved: std_roi_summary.txt")


# ------------------------------------------------------------------
# 8. PLOT 1 — Linear drift model
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 5))

for spec, grp in df_core.groupby("specimen"):
    layer_means = grp.groupby("layer_index")["roi_mean_C"].mean()
    ax.scatter(layer_means.index, layer_means.values, s=60, alpha=0.9, label=spec, zorder=3)

x_line = np.linspace(df_core["layer_index"].min(), df_core["layer_index"].max(), 200)
y_line = intercept + slope * x_line
ax.plot(x_line, y_line, color="black", linewidth=2, linestyle="--",
        label=f"Linear fit: T = {intercept:.2f} + {slope:.4f}·layer\n"
              f"r = {r_value:.3f},  p = {p_value:.2e}")

ax.set_xlabel("Layer index", fontsize=12)
ax.set_ylabel("Mean ROI temperature [°C]", fontsize=12)
ax.set_title("Thermal Drift — Linear Model on STD Specimens", fontsize=13, fontweight="bold")
ax.legend(fontsize=9, loc="upper left")
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "plot_linear_model.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved: plot_linear_model.png")


# ------------------------------------------------------------------
# Helper: thermal heatmap with colorbar
# ------------------------------------------------------------------
def save_heatmap(data_map, title, filename, cmap="inferno", unit="°C"):
    fig, ax = plt.subplots(figsize=(max(5, ROI_W * 1.4), max(3, ROI_H * 1.4) + 1.5))
    vmin, vmax = data_map.min(), data_map.max()
    im = ax.imshow(data_map, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax,
                   interpolation="nearest")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(unit, fontsize=11)
    for r in range(ROI_H):
        for c in range(ROI_W):
            val = data_map[r, c]
            norm_val = (val - vmin) / (vmax - vmin + 1e-9)
            txt_color = "white" if norm_val < 0.6 else "black"
            ax.text(c, r, f"{val:.2f}", ha="center", va="center",
                    fontsize=8, color=txt_color, fontweight="bold")
    ax.set_xticks(range(ROI_W))
    ax.set_yticks(range(ROI_H))
    ax.set_xticklabels([f"col {i}" for i in range(ROI_W)], fontsize=9)
    ax.set_yticklabels([f"row {i}" for i in range(ROI_H)], fontsize=9)
    ax.set_title(title, fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {filename}")


# ------------------------------------------------------------------
# 9. PLOT 2-5 — Thermal heatmaps
# ------------------------------------------------------------------
save_heatmap(
    std_mean_map,
    f"STD ROI — Mean Temperature (drift-corrected, ref = layer 1)\n"
    f"Dataset: {DATASET_NAME}_STD | N frames: {n_frames}",
    "plot_std_roi_mean.png",
    cmap="inferno"
)

save_heatmap(
    std_std_map,
    f"STD ROI — Pixel-wise Std Dev (drift-corrected)\n"
    f"Dataset: {DATASET_NAME}_STD | N frames: {n_frames}",
    "plot_std_roi_std.png",
    cmap="YlOrRd",
    unit="°C (std dev)"
)

save_heatmap(
    std_max_map,
    f"STD ROI — Pixel-wise Max Temperature (drift-corrected)\n"
    f"Dataset: {DATASET_NAME}_STD | N frames: {n_frames}",
    "plot_std_roi_max.png",
    cmap="hot"
)

save_heatmap(
    std_min_map,
    f"STD ROI — Pixel-wise Min Temperature (drift-corrected)\n"
    f"Dataset: {DATASET_NAME}_STD | N frames: {n_frames}",
    "plot_std_roi_min.png",
    cmap="cool"
)

print("\nAll done. Outputs saved in:", OUTPUT_DIR)
