"""
PCA Thermal Feature Space — FDM Paused Specimens vs. STD
=========================================================
Fitta la PCA sui 10 provini con pausa, poi proietta i 2 provini STD
nello stesso piano per confrontare i due regimi termici.

Output (salvati in output/ relativo a questo script):
  output/pca_thermal_feature_space.png
  output/pca_loadings.png
  output/thermal_profiles_by_pause.png
  output/heating_rate_t0_all_specimens.png
  output/hr_vs_T1_comparison.png

Percorsi input: modificare la sezione CONFIG se necessario.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.patches as mpatches

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════
BASE = r"C:\Users\tomga\Desktop\Lab Data science\Dataset_giusti\feature_tables\output"

PATH_ROI_PAUSA = os.path.join(BASE, "feature_table_roi_ROI_wide_3_10_depth_1_4.csv")
PATH_ROI_STD   = os.path.join(BASE, "feature_table_roi_STD_ROI_wide_3_10_depth_1_4.csv")
PATH_PIX_PAUSA = os.path.join(BASE, "feature_table_pixel_ROI_wide_3_10_depth_1_4.csv")
PATH_PIX_STD   = os.path.join(BASE, "feature_table_pixel_STD_ROI_wide_3_10_depth_1_4.csv")

# Output folder: cartella output/ accanto a questo script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR    = os.path.join(SCRIPT_DIR, "output_ROI_3_10_1_4")
os.makedirs(OUT_DIR, exist_ok=True)

# Conversione STD da °C/layer a °C/s: opzione A
DT_LAYER_SECONDS = 1.0 / 3.0

# ══════════════════════════════════════════════════════════════════════════════
# PALETTE
# ══════════════════════════════════════════════════════════════════════════════
PAUSE_COLORS  = {10: "#636EFA", 30: "#00CC96", 60: "#FFA15A", 90: "#EF553B"}
PAUSE_MARKERS = {10: "o",       30: "s",       60: "D",       90: "X"}
STD_COLOR     = "#AB63FA"

# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
df_p      = pd.read_csv(PATH_ROI_PAUSA)
df_s_full = pd.read_csv(PATH_ROI_STD)
df_s      = df_s_full[df_s_full["specimen"] != "STD_OLS_global"].copy()

print(f"Paused specimens : {len(df_p)}")
print(f"STD specimens    : {len(df_s)}")
print(f"DT layer used for STD conversion: {DT_LAYER_SECONDS:.4f} s")

# ══════════════════════════════════════════════════════════════════════════════
# 2. PER-SPECIMEN OLS FOR STD
# ══════════════════════════════════════════════════════════════════════════════
def compute_ols_per_specimen(row):
    t_cols = sorted(
        [c for c in row.index if c.startswith("T") and c[1:].isdigit()],
        key=lambda c: int(c[1:])
    )
    vals = row[t_cols].dropna().values.astype(float)
    if len(vals) < 2:
        return np.nan, np.nan
    layers = np.arange(1, len(vals) + 1)
    m, b = np.polyfit(layers, vals, 1)
    return float(m), float(b)

std_slopes, std_intercepts = zip(*[compute_ols_per_specimen(r) for _, r in df_s.iterrows()])
df_s = df_s.copy()
df_s["slope_OLS"]      = std_slopes
df_s["intercept_OLS"]  = std_intercepts
df_s["heating_rate_t0"] = df_s["slope_OLS"] / DT_LAYER_SECONDS

# ══════════════════════════════════════════════════════════════════════════════
# 3. BUILD ALIGNED FEATURE MATRIX
# ══════════════════════════════════════════════════════════════════════════════
FEAT_COLS = ["temp_0","temp_1","temp_2","temp_3","temp_4","temp_5","temp_6",
             "heating_rate","roi_std","bed_temp"]

rows_p = [{
    "specimen":     r["specimen"],
    "group":        f"{int(r['pausa_s'])}s",
    "pausa_s":      int(r["pausa_s"]),
    "temp_0":       r["T0"],
    "temp_1":       r["T1"], "temp_2": r["T2"], "temp_3": r["T3"],
    "temp_4":       r["T4"], "temp_5": r["T5"], "temp_6": r["T6"],
    "heating_rate": r["heating_rate_t0"],
    "roi_std":      r["roi_std_spaziale_mean"],
    "bed_temp":     r["bed_temp_C"],
} for _, r in df_p.iterrows()]

rows_s = [{
    "specimen":     r["specimen"],
    "group":        "STD",
    "pausa_s":      0,
    "temp_0":       r["intercept_OLS"],
    "temp_1":       r["T1"], "temp_2": r["T2"], "temp_3": r["T3"],
    "temp_4":       r["T4"], "temp_5": r["T5"], "temp_6": r["T6"],
    "heating_rate": r["heating_rate_t0"],
    "roi_std":      r["roi_std_spaziale_mean"],
    "bed_temp":     r["intercept_OLS"],
} for _, r in df_s.iterrows()]

df_all = pd.DataFrame(rows_p + rows_s)

# ══════════════════════════════════════════════════════════════════════════════
# 4. FIT PCA ON PAUSED, PROJECT STD
# ══════════════════════════════════════════════════════════════════════════════
X_pausa   = df_all[df_all["group"] != "STD"][FEAT_COLS].values
X_std     = df_all[df_all["group"] == "STD"][FEAT_COLS].values
scaler    = StandardScaler()
X_pausa_s = scaler.fit_transform(X_pausa)
X_std_s   = scaler.transform(X_std)
pca       = PCA(n_components=2)
pca.fit(X_pausa_s)
coords_p  = pca.transform(X_pausa_s)
coords_s  = pca.transform(X_std_s)
var_exp   = pca.explained_variance_ratio_
loadings  = pd.DataFrame(pca.components_.T, index=FEAT_COLS, columns=["PC1","PC2"])

df_pausa_pca = df_all[df_all["group"] != "STD"].copy().reset_index(drop=True)
df_pausa_pca[["PC1","PC2"]] = coords_p
df_std_pca   = df_all[df_all["group"] == "STD"].copy().reset_index(drop=True)
df_std_pca[["PC1","PC2"]]   = coords_s

print(f"\nVariance explained: PC1={var_exp[0]:.1%}  PC2={var_exp[1]:.1%}")
print("\nLoadings:\n", loadings.round(3))
print("\nSTD heating rates converted to °C/s:")
print(df_s[["specimen", "slope_OLS", "heating_rate_t0"]].round(4))

# ══════════════════════════════════════════════════════════════════════════════
# 5. PLOT 1 — PCA SCATTER
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 7))
ax.set_facecolor("white"); fig.patch.set_facecolor("white")

for pause in [10, 30, 60, 90]:
    sub = df_pausa_pca[df_pausa_pca["pausa_s"] == pause]
    ax.scatter(sub["PC1"], sub["PC2"],
               c=PAUSE_COLORS[pause], marker=PAUSE_MARKERS[pause],
               s=140, zorder=3, edgecolors="#333", linewidths=0.8,
               label=f"Pause {pause}s (n={len(sub)})")
    for _, row in sub.iterrows():
        ax.annotate(row["specimen"].replace("Rec-","").replace("_"," "),
                    (row["PC1"], row["PC2"]),
                    textcoords="offset points", xytext=(6,5), fontsize=8, color="#333")

ax.scatter(df_std_pca["PC1"], df_std_pca["PC2"],
           c=STD_COLOR, marker="*", s=320, zorder=4,
           edgecolors="#333", linewidths=0.8, label="STD (no pause)")
for _, row in df_std_pca.iterrows():
    ax.annotate(row["specimen"].replace("Rec-","").replace("_"," "),
                (row["PC1"], row["PC2"]),
                textcoords="offset points", xytext=(6,5),
                fontsize=8, color=STD_COLOR, fontweight="bold")

ax.axhline(0, color="#bbb", linewidth=0.8, zorder=0)
ax.axvline(0, color="#bbb", linewidth=0.8, zorder=0)
ax.set_xlabel(f"PC1  ({var_exp[0]:.1%} variance explained)", fontsize=12)
ax.set_ylabel(f"PC2  ({var_exp[1]:.1%} variance explained)", fontsize=12)
ax.set_title(
    f"PCA — Thermal Feature Space\n"
    f"PC1={var_exp[0]:.0%}, PC2={var_exp[1]:.0%}  |  STD projected onto paused-only PCA",
    fontsize=13, pad=12)
ax.legend(loc="upper right", fontsize=10, framealpha=0.9)
ax.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "pca_thermal_feature_space.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved: pca_thermal_feature_space.png")

# ══════════════════════════════════════════════════════════════════════════════
# 6. PLOT 2 — PCA LOADINGS BAR CHART
# ══════════════════════════════════════════════════════════════════════════════
FEAT_LABELS = ["T0","T1","T2","T3","T4","T5","T6","HR(t0)","ROI std","Bed T"]
x = np.arange(len(FEAT_COLS))
w = 0.38

fig, ax = plt.subplots(figsize=(10, 5))
ax.set_facecolor("white"); fig.patch.set_facecolor("white")
ax.bar(x - w/2, loadings["PC1"], w, label="PC1", color="#636EFA", edgecolor="white")
ax.bar(x + w/2, loadings["PC2"], w, label="PC2", color="#FFA15A", edgecolor="white")
ax.axhline(0, color="#555", linewidth=1.2)
ax.set_xticks(x); ax.set_xticklabels(FEAT_LABELS, fontsize=12)
ax.set_xlabel("Feature", fontsize=12)
ax.set_ylabel("Loading value", fontsize=12)
ax.set_title(
    "PCA Loadings — Feature Contributions to PC1 and PC2\n"
    "Positive = aligned with PC axis; negative = opposite direction",
    fontsize=13, pad=10)
ax.legend(fontsize=11)
ax.grid(True, axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "pca_loadings.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved: pca_loadings.png")

# ══════════════════════════════════════════════════════════════════════════════
# 7. PLOT 3 — THERMAL PROFILES
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_facecolor("white"); fig.patch.set_facecolor("white")

for pause in [10, 30, 60, 90]:
    sub = df_p[df_p["pausa_s"] == pause]
    mean_profile = sub[["T0","T1","T2","T3","T4","T5","T6"]].mean().values
    ax.plot(range(7), mean_profile,
            color=PAUSE_COLORS[pause], linewidth=2.5,
            marker=PAUSE_MARKERS[pause], markersize=8,
            label=f"Pause {pause}s (mean, n={len(sub)})")

for _, row in df_s.iterrows():
    vals, xs = [], []
    for i, c in enumerate(["T1","T2","T3","T4","T5","T6","T7"]):
        if c in row.index and pd.notna(row[c]):
            vals.append(float(row[c])); xs.append(i + 1)
    ax.plot(xs, vals, color=STD_COLOR, linewidth=2.5, linestyle="--",
            marker="*", markersize=12,
            label=f"STD: {row['specimen'].replace('Rec-','').replace('_',' ')}")

ax.set_xticks(range(8))
ax.set_xticklabels(["T0","T1","T2","T3","T4","T5","T6","T7"], fontsize=12)
ax.set_xlabel("Layer index (T0 = WLS extrapolated for paused; STD starts at T1)", fontsize=11)
ax.set_ylabel("Temperature (°C)", fontsize=12)
ax.set_title(
    "Mean Thermal Profiles by Pause Duration vs. STD Reference\n"
    "STD specimens are the hottest — no cooling between layers",
    fontsize=13, pad=10)
ax.legend(loc="lower right", fontsize=10, framealpha=0.9)
ax.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "thermal_profiles_by_pause.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved: thermal_profiles_by_pause.png")

# ══════════════════════════════════════════════════════════════════════════════
# 8. PLOT 4 — HEATING RATE t0 — ALL SPECIMENS (ALL IN °C/s)
# ══════════════════════════════════════════════════════════════════════════════
df_hr = df_all[["specimen","pausa_s","group","heating_rate"]].copy()
df_hr = df_hr.sort_values(["pausa_s","specimen"]).reset_index(drop=True)

bar_colors = []
for _, r in df_hr.iterrows():
    if r["group"] == "STD":
        bar_colors.append(STD_COLOR)
    else:
        bar_colors.append(PAUSE_COLORS[int(r["pausa_s"])])

labels = (df_hr["specimen"]
          .str.replace("Rec-", "", regex=False)
          .str.replace("_", "\n", regex=False))

fig, ax = plt.subplots(figsize=(13, 6))
ax.set_facecolor("white"); fig.patch.set_facecolor("white")

bars = ax.bar(range(len(df_hr)), df_hr["heating_rate"],
              color=bar_colors, edgecolor="white", linewidth=0.5, zorder=3)

for val, bar in zip(df_hr["heating_rate"], bars):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.005,
            f"{val:.3f}", ha="center", va="bottom", fontsize=8, color="#333")

n_std = (df_hr["group"] == "STD").sum()
ax.axvline(n_std - 0.5, color="#888", linewidth=1.5, linestyle="--")
ax.text(n_std - 0.5 + 0.1, ax.get_ylim()[1] * 0.97,
        "← STD  |  Paused →", fontsize=9, color="#666", va="top")

ax.set_xticks(range(len(df_hr)))
ax.set_xticklabels(labels, fontsize=8)
ax.set_xlabel("Specimen", fontsize=12)
ax.set_ylabel("Heating rate at t0 (°C/s)", fontsize=12)
ax.set_title(
    "Heating Rate at t0 — All Specimens\n"
    "Paused: A·α from WLS fit  |  STD: OLS slope converted from °C/layer to °C/s",
    fontsize=13, pad=10)

legend_handles = [
    mpatches.Patch(color=PAUSE_COLORS[p], label=f"Pause {p}s") for p in [10, 30, 60, 90]
] + [mpatches.Patch(color=STD_COLOR, label=f"STD converted to °C/s (dt={DT_LAYER_SECONDS:.3f}s)")]
ax.legend(handles=legend_handles, fontsize=10, loc="upper left", framealpha=0.9)
ax.grid(True, axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "heating_rate_t0_all_specimens.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved: heating_rate_t0_all_specimens.png")

# ══════════════════════════════════════════════════════════════════════════════
# 9. PLOT 5 — HR(t0) vs T1 SIDE-BY-SIDE COMPARISON
#
# Obiettivo: capire se l'heating rate accentua le differenze tra gruppi
# rispetto alla semplice temperatura al primo layer (T1).
#
# Pannello sinistro : Heating Rate at t0 [°C/s]  — tutti i provini
# Pannello destro   : T1 [°C]                    — tutti i provini
# Stesso ordinamento e colorazione per rendere il confronto immediato.
# ══════════════════════════════════════════════════════════════════════════════

df_cmp = df_all[["specimen","pausa_s","group","heating_rate","temp_1"]].copy()
df_cmp = df_cmp.sort_values(["pausa_s","specimen"]).reset_index(drop=True)

bar_colors_cmp = []
for _, r in df_cmp.iterrows():
    if r["group"] == "STD":
        bar_colors_cmp.append(STD_COLOR)
    else:
        bar_colors_cmp.append(PAUSE_COLORS[int(r["pausa_s"])])

labels_cmp = (df_cmp["specimen"]
              .str.replace("Rec-", "", regex=False)
              .str.replace("_", "\n", regex=False))

n_std_cmp = (df_cmp["group"] == "STD").sum()

fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(18, 6))
fig.patch.set_facecolor("white")

# ---------- LEFT: Heating Rate ----------
ax_left.set_facecolor("white")
bars_l = ax_left.bar(range(len(df_cmp)), df_cmp["heating_rate"],
                     color=bar_colors_cmp, edgecolor="white", linewidth=0.5, zorder=3)

for val, bar in zip(df_cmp["heating_rate"], bars_l):
    ax_left.text(bar.get_x() + bar.get_width()/2, val + 0.003,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=7.5, color="#333")

ax_left.axvline(n_std_cmp - 0.5, color="#888", linewidth=1.5, linestyle="--")
ax_left.set_xticks(range(len(df_cmp)))
ax_left.set_xticklabels(labels_cmp, fontsize=7.5)
ax_left.set_xlabel("Specimen", fontsize=11)
ax_left.set_ylabel("Heating rate at t0 (°C/s)", fontsize=11)
ax_left.set_title("Heating Rate at t0\nPaused: A·α  |  STD: OLS/dt", fontsize=12, pad=8)
ax_left.grid(True, axis="y", linestyle="--", alpha=0.4)

# Coefficient of Variation annotation
cv_hr_paused = df_cmp[df_cmp["group"] != "STD"]["heating_rate"].std() / \
               df_cmp[df_cmp["group"] != "STD"]["heating_rate"].mean()
cv_hr_all    = df_cmp["heating_rate"].std() / df_cmp["heating_rate"].mean()
ax_left.text(0.98, 0.97,
             f"CV (paused) = {cv_hr_paused:.1%}\nCV (all)    = {cv_hr_all:.1%}",
             transform=ax_left.transAxes, ha="right", va="top",
             fontsize=9, color="#333",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#ccc"))

# ---------- RIGHT: T1 ----------
ax_right.set_facecolor("white")
bars_r = ax_right.bar(range(len(df_cmp)), df_cmp["temp_1"],
                      color=bar_colors_cmp, edgecolor="white", linewidth=0.5, zorder=3)

for val, bar in zip(df_cmp["temp_1"], bars_r):
    ax_right.text(bar.get_x() + bar.get_width()/2, val + 0.3,
                  f"{val:.1f}", ha="center", va="bottom", fontsize=7.5, color="#333")

ax_right.axvline(n_std_cmp - 0.5, color="#888", linewidth=1.5, linestyle="--")
ax_right.set_xticks(range(len(df_cmp)))
ax_right.set_xticklabels(labels_cmp, fontsize=7.5)
ax_right.set_xlabel("Specimen", fontsize=11)
ax_right.set_ylabel("T1 (°C)  — temperature at first measured layer", fontsize=11)
ax_right.set_title("T1 — Temperature at Layer 1\nDirect IR measurement", fontsize=12, pad=8)
ax_right.grid(True, axis="y", linestyle="--", alpha=0.4)

cv_t1_paused = df_cmp[df_cmp["group"] != "STD"]["temp_1"].std() / \
               df_cmp[df_cmp["group"] != "STD"]["temp_1"].mean()
cv_t1_all    = df_cmp["temp_1"].std() / df_cmp["temp_1"].mean()
ax_right.text(0.98, 0.97,
              f"CV (paused) = {cv_t1_paused:.1%}\nCV (all)    = {cv_t1_all:.1%}",
              transform=ax_right.transAxes, ha="right", va="top",
              fontsize=9, color="#333",
              bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#ccc"))

# ---------- Shared legend ----------
legend_handles = [
    mpatches.Patch(color=PAUSE_COLORS[p], label=f"Pause {p}s") for p in [10, 30, 60, 90]
] + [mpatches.Patch(color=STD_COLOR, label="STD")]
fig.legend(handles=legend_handles, loc="lower center", ncol=5,
           fontsize=10, framealpha=0.9, bbox_to_anchor=(0.5, -0.02))

fig.suptitle(
    "Heating Rate vs T1 — Does HR accentuate group differences?\n"
    "Compare CV values: higher CV = more spread across groups",
    fontsize=13, y=1.02
)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "hr_vs_T1_comparison.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved: hr_vs_T1_comparison.png")

print(f"\nDone. All outputs in: {OUT_DIR}")
