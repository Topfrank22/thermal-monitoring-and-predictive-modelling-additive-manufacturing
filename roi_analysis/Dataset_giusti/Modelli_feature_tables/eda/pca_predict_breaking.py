#!/usr/bin/env python3
"""
PCA-Based Breaking Load Prediction
==================================
Uses the PCA components (fitted on paused specimens) to predict breaking load.

Models compared via LOOCV (trained on ALL specimens):
  Model 1 : Linear Regression    peso ~ PC1
  Model 2 : Linear Regression    peso ~ PC1 + PC2
  Model 3 : Polynomial Degree-2  peso ~ PC1 + PC1²
  Model 4 : SVR (RBF kernel)     peso ~ PC1 + PC2
  Model 5 : Gaussian Process     peso ~ PC1 + PC2

This version is grid-friendly:
- PATH_ROI_PAUSA, PATH_ROI_STD, PATH_MECH, ROOT_OUT are configurable
- the grid search can rewrite only these variables
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from scipy import stats
from sklearn.model_selection import LeaveOneOut
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG & FOLDERS
# ══════════════════════════════════════════════════════════════════════════════
BASE = r"C:\Users\tomga\Desktop\Lab Data science\Dataset_giusti\feature_tables\output"
PATH_ROI_PAUSA = 'C:\\Users\\tomga\\Desktop\\Lab Data science\\Dataset_giusti\\feature_tables\\output\\feature_table_roi_ROI_x3-12_y2-5_offNonG31_offG30.csv'
PATH_ROI_STD = 'C:\\Users\\tomga\\Desktop\\Lab Data science\\Dataset_giusti\\feature_tables\\output\\feature_table_roi_STD_ROI_x3-12_y2-5_offNonG31_offG30.csv'
PATH_MECH = 'C:\\Users\\tomga\\Desktop\\Lab Data science\\Dataset_giusti\\mechanical_properties\\mechanical_properties_summary.csv'

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_OUT = 'C:\\Users\\tomga\\Desktop\\Lab Data science\\Dataset_giusti\\roi_grid_search\\grid_search_results\\ROI_x3-12_y2-5_offNonG31_offG30\\analysis'

DIRS = {
    "pca": os.path.join(ROOT_OUT, "01_PCA_Analysis"),
    "m1": os.path.join(ROOT_OUT, "02_Model_Linear_PC1"),
    "m2": os.path.join(ROOT_OUT, "03_Model_Linear_PC1_PC2"),
    "m3": os.path.join(ROOT_OUT, "04_Model_Poly_Deg2"),
    "m4": os.path.join(ROOT_OUT, "05_Model_SVR"),
    "m5": os.path.join(ROOT_OUT, "06_Model_GPR"),
    "comp": os.path.join(ROOT_OUT, "07_Model_Comparison"),
    "sum": os.path.join(ROOT_OUT, "08_Summary"),
}

for d in DIRS.values():
    os.makedirs(d, exist_ok=True)

DT_LAYER_SECONDS = 1.0 / 3.0
PAUSE_COLORS = {10: "#636EFA", 30: "#00CC96", 60: "#FFA15A", 90: "#EF553B"}
PAUSE_MARKERS = {10: "o", 30: "s", 60: "D", 90: "X"}
STD_COLOR = "#AB63FA"


def get_color(row):
    return STD_COLOR if row["group"] == "STD" else PAUSE_COLORS[int(row["pausa_s"])]


def get_marker(row):
    return "*" if row["group"] == "STD" else PAUSE_MARKERS[int(row["pausa_s"])]


def get_label(row):
    return "STD" if row["group"] == "STD" else f"Pause {int(row['pausa_s'])}s"


# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD DATA & PCA
# ══════════════════════════════════════════════════════════════════════════════
df_p = pd.read_csv(PATH_ROI_PAUSA)
df_s_full = pd.read_csv(PATH_ROI_STD)
df_s = df_s_full[df_s_full["specimen"] != "STD_OLS_global"].copy()
df_mech = pd.read_csv(PATH_MECH)


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
df_s["slope_OLS"] = std_slopes
df_s["intercept_OLS"] = std_intercepts
df_s["heating_rate_t0"] = df_s["slope_OLS"] / DT_LAYER_SECONDS

FEAT_COLS = [
    "temp_0", "temp_1", "temp_2", "temp_3", "temp_4", "temp_5", "temp_6",
    "heating_rate", "roi_std", "bed_temp"
]

rows_p = [
    {
        "specimen": r["specimen"],
        "group": f"{int(r['pausa_s'])}s",
        "pausa_s": int(r["pausa_s"]),
        "temp_0": r["T0"],
        "temp_1": r["T1"],
        "temp_2": r["T2"],
        "temp_3": r["T3"],
        "temp_4": r["T4"],
        "temp_5": r["T5"],
        "temp_6": r["T6"],
        "heating_rate": r["heating_rate_t0"],
        "roi_std": r["roi_std_spaziale_mean"],
        "bed_temp": r["bed_temp_C"],
    }
    for _, r in df_p.iterrows()
]

rows_s = [
    {
        "specimen": r["specimen"],
        "group": "STD",
        "pausa_s": 0,
        "temp_0": r["intercept_OLS"],
        "temp_1": r["T1"],
        "temp_2": r["T2"],
        "temp_3": r["T3"],
        "temp_4": r["T4"],
        "temp_5": r["T5"],
        "temp_6": r["T6"],
        "heating_rate": r["heating_rate_t0"],
        "roi_std": r["roi_std_spaziale_mean"],
        "bed_temp": r["intercept_OLS"],
    }
    for _, r in df_s.iterrows()
]

df_all = pd.DataFrame(rows_p + rows_s)
df_all = df_all.merge(df_mech[["specimen", "peso_rottura_kg"]], on="specimen", how="left")

# Fit PCA on paused only
df_pausa = df_all[df_all["group"] != "STD"]
scaler = StandardScaler()
X_pausa_s = scaler.fit_transform(df_pausa[FEAT_COLS].values)
pca = PCA(n_components=2)
pca.fit(X_pausa_s)

var_exp = pca.explained_variance_ratio_
print(f"PCA variance explained: PC1={var_exp[0]:.1%}  PC2={var_exp[1]:.1%}")

# Project all points
df_all[["PC1", "PC2"]] = pca.transform(scaler.transform(df_all[FEAT_COLS].values))
print(f"Total specimens used for modeling: {len(df_all)}")

# ══════════════════════════════════════════════════════════════════════════════
# 2. LOOCV
# ══════════════════════════════════════════════════════════════════════════════
X_pc1 = df_all[["PC1"]].values
X_pc12 = df_all[["PC1", "PC2"]].values
y = df_all["peso_rottura_kg"].values

slope_pc1, intercept_pc1, r_pc1, p_pc1, _ = stats.linregress(df_all["PC1"].values, y)
r2_pc1 = r_pc1**2

gpr_kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 10.0)) + WhiteKernel(
    noise_level=0.1, noise_level_bounds=(1e-5, 1.0)
)

loo = LeaveOneOut()
y_true, y_pred_pc1, y_pred_pc12, y_pred_poly2, y_pred_svr, y_pred_gpr = [], [], [], [], [], []

for train_idx, test_idx in loo.split(X_pc1):
    lr1 = LinearRegression()
    lr1.fit(X_pc1[train_idx], y[train_idx])
    y_pred_pc1.append(lr1.predict(X_pc1[test_idx])[0])

    lr2 = LinearRegression()
    lr2.fit(X_pc12[train_idx], y[train_idx])
    y_pred_pc12.append(lr2.predict(X_pc12[test_idx])[0])

    poly_pipe = Pipeline([
        ("poly", PolynomialFeatures(degree=2, include_bias=True)),
        ("lr", LinearRegression())
    ])
    poly_pipe.fit(X_pc1[train_idx], y[train_idx])
    y_pred_poly2.append(poly_pipe.predict(X_pc1[test_idx])[0])

    svr_sc = StandardScaler()
    svr = SVR(kernel="rbf", C=1.0, epsilon=0.1, gamma="scale")
    svr.fit(svr_sc.fit_transform(X_pc12[train_idx]), y[train_idx])
    y_pred_svr.append(svr.predict(svr_sc.transform(X_pc12[test_idx]))[0])

    gpr_sc = StandardScaler()
    gpr = GaussianProcessRegressor(
        kernel=gpr_kernel,
        n_restarts_optimizer=5,
        normalize_y=True,
        random_state=42,
    )
    gpr.fit(gpr_sc.fit_transform(X_pc12[train_idx]), y[train_idx])
    y_pred_gpr.append(gpr.predict(gpr_sc.transform(X_pc12[test_idx]))[0])

    y_true.append(y[test_idx][0])

y_true = np.array(y_true)
y_pred_pc1 = np.array(y_pred_pc1)
y_pred_pc12 = np.array(y_pred_pc12)
y_pred_poly2 = np.array(y_pred_poly2)
y_pred_svr = np.array(y_pred_svr)
y_pred_gpr = np.array(y_pred_gpr)


def loocv_metrics(y_t, y_p):
    return np.sqrt(mean_squared_error(y_t, y_p)), r2_score(y_t, y_p)


# Fully trained estimators
m1_est = LinearRegression().fit(X_pc1, y)
m2_est = LinearRegression().fit(X_pc12, y)
m3_est = Pipeline([
    ("poly", PolynomialFeatures(degree=2, include_bias=True)),
    ("lr", LinearRegression())
]).fit(X_pc1, y)
m4_est = Pipeline([
    ("scaler", StandardScaler()),
    ("svr", SVR(kernel="rbf", C=1.0, epsilon=0.1, gamma="scale"))
]).fit(X_pc12, y)
m5_est = Pipeline([
    ("scaler", StandardScaler()),
    ("gpr", GaussianProcessRegressor(
        kernel=gpr_kernel,
        n_restarts_optimizer=5,
        normalize_y=True,
        random_state=42,
    ))
]).fit(X_pc12, y)

models_data = {
    "m1": {"name": "Linear PC1", "preds": y_pred_pc1, "dir": DIRS["m1"], "est": m1_est, "features": "PC1"},
    "m2": {"name": "Linear PC1+PC2", "preds": y_pred_pc12, "dir": DIRS["m2"], "est": m2_est, "features": "PC12"},
    "m3": {"name": "Polynomial deg-2", "preds": y_pred_poly2, "dir": DIRS["m3"], "est": m3_est, "features": "PC1"},
    "m4": {"name": "SVR RBF", "preds": y_pred_svr, "dir": DIRS["m4"], "est": m4_est, "features": "PC12"},
    "m5": {"name": "GPR RBF", "preds": y_pred_gpr, "dir": DIRS["m5"], "est": m5_est, "features": "PC12"},
}

for k in models_data:
    rmse, r2 = loocv_metrics(y_true, models_data[k]["preds"])
    models_data[k]["rmse"] = rmse
    models_data[k]["r2"] = r2
    df_all[f"pred_{k}"] = models_data[k]["preds"]
    df_all[f"resid_{k}"] = y_true - models_data[k]["preds"]

# ══════════════════════════════════════════════════════════════════════════════
# 3. SAVE SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
with open(os.path.join(DIRS["sum"], "pca_prediction_summary.txt"), "w", encoding="utf-8") as f:
    f.write("PCA-BASED PREDICTION OF BREAKING LOAD — ALL SPECIMENS (LOOCV)\n")
    f.write("=" * 65 + "\n\n")
    for k, v in models_data.items():
        f.write(f"Model: {v['name']}\n")
        f.write(f"  LOOCV RMSE : {v['rmse']:.4f} kg\n")
        f.write(f"  LOOCV R²   : {v['r2']:.4f}\n\n")
    best = min(models_data.values(), key=lambda x: x["rmse"])
    f.write(f"Best by LOOCV RMSE: {best['name']} ({best['rmse']:.4f} kg)\n")
print(f"Saved: {DIRS['sum']}/pca_prediction_summary.txt")

# ══════════════════════════════════════════════════════════════════════════════
# 4. PLOTS — PCA Analysis
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 7))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")
scatter = ax.scatter(
    df_all["PC1"], df_all["PC2"], c=df_all["peso_rottura_kg"],
    cmap="viridis", vmin=y.min(), vmax=y.max(), s=150, zorder=3,
    edgecolors="#333", linewidths=1
)
plt.colorbar(scatter, ax=ax).set_label("Breaking Load (kg)", fontsize=12)

df_std = df_all[df_all["group"] == "STD"]
ax.scatter(
    df_std["PC1"], df_std["PC2"], c=df_std["peso_rottura_kg"], cmap="viridis",
    marker="*", s=400, zorder=4, edgecolors="#333", linewidths=1,
    vmin=y.min(), vmax=y.max()
)

for _, row in df_all.iterrows():
    fw = "bold" if row["group"] == "STD" else "normal"
    ax.annotate(
        row["specimen"].replace("Rec-", "").replace("_", "\n"),
        (row["PC1"], row["PC2"]),
        textcoords="offset points",
        xytext=(6, 5),
        fontsize=8,
        fontweight=fw,
    )

ax.axhline(0, color="#bbb", linewidth=0.8)
ax.axvline(0, color="#bbb", linewidth=0.8)
ax.set_xlabel(f"PC1 ({var_exp[0]:.1%} variance)", fontsize=12)
ax.set_ylabel(f"PC2 ({var_exp[1]:.1%} variance)", fontsize=12)
ax.set_title("PCA Thermal Feature Space Colored by Breaking Load\nStars = STD specimens", fontsize=14, pad=10)
ax.grid(True, linestyle="--", alpha=0.4)
fig.savefig(os.path.join(DIRS["pca"], "pca_breaking_scatter.png"), dpi=150, bbox_inches="tight")
plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# 5. PER-MODEL PLOTS
# ══════════════════════════════════════════════════════════════════════════════
legend_handles = [mpatches.Patch(color=PAUSE_COLORS[p], label=f"Pause {p}s") for p in [10, 30, 60, 90]]
legend_handles.append(mpatches.Patch(color=STD_COLOR, label="STD"))

for k, m in models_data.items():
    fig, ax = plt.subplots(figsize=(7, 7))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for _, row in df_all.iterrows():
        ax.scatter(
            row["peso_rottura_kg"], row[f"pred_{k}"],
            color=get_color(row),
            marker=get_marker(row),
            s=150 if row["group"] == "STD" else 120,
            edgecolors="white",
            zorder=3,
        )

    min_val = min(df_all["peso_rottura_kg"].min(), df_all[f"pred_{k}"].min()) - 0.2
    max_val = max(df_all["peso_rottura_kg"].max(), df_all[f"pred_{k}"].max()) + 0.2
    ax.plot([min_val, max_val], [min_val, max_val], color="#333", linestyle="--", linewidth=1.5, label="Perfect prediction")
    ax.set_xlabel("Actual Breaking Load (kg)", fontsize=12)
    ax.set_ylabel("Predicted (LOOCV) kg", fontsize=12)
    ax.set_title(
        f"LOOCV Prediction vs Actual — {m['name']}\nRMSE={m['rmse']:.3f} kg, $R^2$={m['r2']:.3f}",
        fontsize=13,
        pad=10,
    )
    ax.legend(
        handles=legend_handles + [plt.Line2D([0], [0], color="#333", linestyle="--")],
        labels=[h.get_label() for h in legend_handles] + ["Perfect prediction"],
        fontsize=10,
    )
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    fig.savefig(os.path.join(m["dir"], "prediction_vs_actual.png"), dpi=150, bbox_inches="tight")
    plt.close()

    df_all_sorted = df_all.sort_values("peso_rottura_kg").reset_index(drop=True)
    x_pos = np.arange(len(df_all_sorted))
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.bar(
        x_pos,
        df_all_sorted[f"resid_{k}"].values,
        color=[get_color(r) for _, r in df_all_sorted.iterrows()],
        edgecolor="white",
    )
    ax.axhline(0, color="#333", linewidth=1.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(
        df_all_sorted["specimen"].str.replace("Rec-", "").str.replace("_", "\n"),
        rotation=45,
        ha="right",
        fontsize=9,
    )
    ax.set_ylabel("Residual (Actual - Predicted) kg", fontsize=12)
    ax.set_title(f"LOOCV Residuals — {m['name']}", fontsize=14, pad=10)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax.legend(handles=legend_handles, fontsize=10)
    fig.savefig(os.path.join(m["dir"], "loocv_residuals.png"), dpi=150, bbox_inches="tight")
    plt.close()

# Specific plot for Model 1
fig, ax = plt.subplots(figsize=(8, 6))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")
for _, row in df_all.iterrows():
    ax.scatter(
        row["PC1"], row["peso_rottura_kg"],
        color=get_color(row),
        marker=get_marker(row),
        s=150 if row["group"] == "STD" else 120,
        edgecolors="white",
        zorder=3,
    )
x_line = np.linspace(df_all["PC1"].min(), df_all["PC1"].max(), 100)
ax.plot(x_line, intercept_pc1 + slope_pc1 * x_line, color="#333", linestyle="--", linewidth=2)
ax.set_xlabel(f"PC1 ({var_exp[0]:.1%} variance)", fontsize=12)
ax.set_ylabel("Breaking Load (kg)", fontsize=12)
ax.set_title(f"Breaking Load vs PC1 (All Specimens)\n$R^2$ = {r2_pc1:.3f}, $p$ = {p_pc1:.4f}", fontsize=14, pad=10)
ax.grid(True, linestyle="--", alpha=0.4)
ax.legend(handles=legend_handles, fontsize=10)
fig.savefig(os.path.join(DIRS["m1"], "pc1_vs_breaking_regression.png"), dpi=150, bbox_inches="tight")
plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# 6. MODEL COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
model_keys = ["m1", "m2", "m3", "m4", "m5"]
model_names_short = ["Linear\nPC1", "Linear\nPC1+PC2", "Poly-2\nPC1", "SVR\nRBF", "GPR\nRBF"]
rmse_vals = [models_data[k]["rmse"] for k in model_keys]
r2_vals = [models_data[k]["r2"] for k in model_keys]
bar_clrs = ["#636EFA", "#00CC96", "#FFA15A", "#EF553B", "#AB63FA"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
fig.patch.set_facecolor("white")
for ax_, vals, ylabel, title in [
    (ax1, rmse_vals, "LOOCV RMSE (kg)", "LOOCV RMSE — All Models"),
    (ax2, r2_vals, "LOOCV R²", "LOOCV R² — All Models"),
]:
    ax_.set_facecolor("white")
    bars = ax_.bar(model_names_short, vals, color=bar_clrs, edgecolor="white", zorder=3)
    for bar, val in zip(bars, vals):
        ax_.text(
            bar.get_x() + bar.get_width() / 2,
            val + (0.002 if ax_ is ax1 else 0.01),
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    ax_.set_ylabel(ylabel, fontsize=12)
    ax_.set_title(title, fontsize=13)
    ax_.grid(True, axis="y", linestyle="--", alpha=0.4)
ax2.axhline(0, color="#555", linewidth=1)
fig.suptitle("Model Comparison — LOOCV on ALL Specimens", fontsize=14, y=1.02)
plt.tight_layout()
fig.savefig(os.path.join(DIRS["comp"], "model_comparison.png"), dpi=150, bbox_inches="tight")
plt.close()

# ══════════════════════��═══════════════════════════════════════════════════════
# 7. PREDICTION SURFACES
# ══════════════════════════════════════════════════════════════════════════════
grid_pc1 = np.linspace(df_all["PC1"].min() - 0.5, df_all["PC1"].max() + 0.5, 100)
grid_pc2 = np.linspace(df_all["PC2"].min() - 0.5, df_all["PC2"].max() + 0.5, 100)
PC1_grid, PC2_grid = np.meshgrid(grid_pc1, grid_pc2)

def get_grid_predictions(est, features):
    if features == "PC1":
        return est.predict(PC1_grid.ravel().reshape(-1, 1)).reshape(PC1_grid.shape)
    return est.predict(np.c_[PC1_grid.ravel(), PC2_grid.ravel()]).reshape(PC1_grid.shape)

groups = df_all["group"].unique()

for k, m in models_data.items():
    est = m["est"]
    feat_type = m["features"]
    Z_pred = get_grid_predictions(est, feat_type)

    if k == "m5":
        gpr_pipeline = m5_est
        gpr_scaler = gpr_pipeline.named_steps["scaler"]
        gpr_model = gpr_pipeline.named_steps["gpr"]

        X_grid = np.c_[PC1_grid.ravel(), PC2_grid.ravel()]
        X_grid_s = gpr_scaler.transform(X_grid)

        pred_mean, pred_std = gpr_model.predict(X_grid_s, return_std=True)
        Z_pred = pred_mean.reshape(PC1_grid.shape)
        Z_low = (pred_mean - 1.96 * pred_std).reshape(PC1_grid.shape)
        Z_high = (pred_mean + 1.96 * pred_std).reshape(PC1_grid.shape)

    if k != "m5":
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection="3d")
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")

        surf = ax.plot_surface(
            PC1_grid, PC2_grid, Z_pred,
            cmap="viridis", alpha=0.55,
            linewidth=0, antialiased=True,
            vmin=y.min(), vmax=y.max()
        )

        for _, row in df_all.iterrows():
            ax.scatter(
                row["PC1"], row["PC2"], row["peso_rottura_kg"],
                color=get_color(row), marker=get_marker(row),
                s=200 if row["group"] == "STD" else 120,
                edgecolors="#333", linewidths=0.8, zorder=10,
            )

        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("Breaking Load (kg)")
        ax.set_title(
            f"3D Prediction Surface — {m['name']}\n"
            f"(Actual points plotted at their true coordinates)",
            pad=15,
        )
        ax.view_init(elev=20, azim=135)
        fig.colorbar(surf, shrink=0.5, aspect=10, label="Predicted Breaking Load (kg)")
        plt.tight_layout()
        fig.savefig(os.path.join(m["dir"], "prediction_surface_3d.png"), dpi=150, bbox_inches="tight")
        plt.close()

    else:
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection="3d")
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")

        surf_mean = ax.plot_surface(
            PC1_grid, PC2_grid, Z_pred,
            cmap="viridis", alpha=0.78,
            linewidth=0, antialiased=True,
            vmin=y.min(), vmax=y.max()
        )
        ax.plot_surface(PC1_grid, PC2_grid, Z_low, cmap="Blues", alpha=0.22, linewidth=0, antialiased=True)
        ax.plot_surface(PC1_grid, PC2_grid, Z_high, cmap="Reds", alpha=0.22, linewidth=0, antialiased=True)

        for _, row in df_all.iterrows():
            ax.scatter(
                row["PC1"], row["PC2"], row["peso_rottura_kg"],
                color=get_color(row), marker=get_marker(row),
                s=200 if row["group"] == "STD" else 120,
                edgecolors="#333", linewidths=0.8, zorder=10,
            )

        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("Breaking Load (kg)")
        ax.set_title(
            "3D GPR Prediction Surface with 95% Confidence Limits\n"
            "(Mean surface + lower/upper confidence surfaces)",
            pad=15,
        )
        ax.view_init(elev=20, azim=135)
        fig.colorbar(surf_mean, shrink=0.5, aspect=10, label="Predicted Breaking Load (kg)")
        plt.tight_layout()
        fig.savefig(os.path.join(m["dir"], "prediction_surface_3d_ci.png"), dpi=150, bbox_inches="tight")
        plt.close()

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    heatmap = ax.contourf(
        PC1_grid, PC2_grid, Z_pred, levels=50,
        cmap="viridis", vmin=y.min(), vmax=y.max(), alpha=0.8
    )

    for gp in groups:
        sub = df_all[df_all["group"] == gp]
        m_char = "*" if gp == "STD" else PAUSE_MARKERS[int(gp.replace("s", ""))]
        size = 350 if gp == "STD" else 150
        ax.scatter(
            sub["PC1"], sub["PC2"], c=sub["peso_rottura_kg"],
            cmap="viridis", marker=m_char, s=size,
            edgecolors="black", linewidths=1.5,
            vmin=y.min(), vmax=y.max(), zorder=5,
            label=f"{gp} (Actual)"
        )

    ax.set_xlabel(f"PC1 ({var_exp[0]:.1%} variance)", fontsize=12)
    ax.set_ylabel(f"PC2 ({var_exp[1]:.1%} variance)", fontsize=12)

    if k != "m5":
        ax.set_title(
            f"2D Prediction Landscape — {m['name']}\n"
            f"Background: Predicted breaking load | Points: Actual breaking load",
            fontsize=13, pad=10
        )
    else:
        ax.contour(
            PC1_grid, PC2_grid, Z_low,
            levels=12, colors="white", linewidths=0.8, linestyles="dashed", alpha=0.7
        )
        ax.contour(
            PC1_grid, PC2_grid, Z_high,
            levels=12, colors="red", linewidths=0.8, linestyles="dotted", alpha=0.7
        )
        ax.set_title(
            "2D GPR Prediction Landscape with 95% Confidence Limits\n"
            "Background: Mean prediction | White dashed: lower CI | Red dotted: upper CI",
            fontsize=13, pad=10
        )

    ax.grid(True, linestyle="--", alpha=0.3, color="#fff")
    cbar = fig.colorbar(heatmap, ax=ax)
    cbar.set_label("Breaking Load (kg)", fontsize=12)

    legend_elements = []
    for gp in sorted(groups, key=lambda g: 0 if g == "STD" else int(g.replace("s", ""))):
        m_char = "*" if gp == "STD" else PAUSE_MARKERS[int(gp.replace("s", ""))]
        legend_elements.append(
            plt.Line2D(
                [0], [0], marker=m_char, color="w",
                markerfacecolor="gray", markeredgecolor="black",
                markersize=10,
                label=f"Pause {gp}" if gp != "STD" else "STD"
            )
        )

    ax.legend(handles=legend_elements, loc="upper right", framealpha=0.9, facecolor="white")

    if k != "m5":
        fig.savefig(os.path.join(m["dir"], "prediction_landscape_2d.png"), dpi=150, bbox_inches="tight")
    else:
        fig.savefig(os.path.join(m["dir"], "prediction_landscape_2d_ci.png"), dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved 3D/2D surfaces in: {m['dir']}")

print(f"\nDone. All structured outputs generated in: {ROOT_OUT}")