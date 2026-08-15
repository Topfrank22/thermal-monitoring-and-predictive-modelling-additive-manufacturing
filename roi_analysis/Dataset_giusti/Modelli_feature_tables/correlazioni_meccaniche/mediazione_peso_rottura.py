"""
mediazione_peso_rottura.py
==========================
Obiettivo: trovare la variabile termica X tale che:

    pausa_s  -->  X  -->  peso_rottura_kg

Se X "media" l'effetto della pausa sul peso, allora X rappresenta
il meccanismo fisico sottostante all'indebolimento del pezzo,
scollegando il fattore tempo dal fattore temperatura/dinamica termica.

Note sulla classificazione delle feature
-----------------------------------------
- FEAT_DIRECT: misure IR dirette dalla camera termica (T1-T6, bed_temp_C)
  bed_temp_C è la temperatura media del ROI misurata PRIMA della deposizione
  del nuovo layer — è una misura diretta, non un parametro di modello.
- FEAT_EXTRAP: feature ricavate dal fit WLS del profilo termico
  (T0 estrapolato, heating rates, A, alpha, delta_T, t_star, roi_std)

Analisi eseguite
----------------
1. Correlazioni univariate: corr(feature, peso) + p-value per ogni feature
2. Confronto R² (OLS semplice): peso ~ pausa_s vs peso ~ X per ogni X
3. Test di mediazione (Baron & Kenny step 2+3)
4. Proxy check specifico: corr(bed_temp_C, pausa_s) + variabilità intra-pausa
5. Scatter plot peso vs ogni feature, colorati per pausa

Input
-----
feature_table_roi_ROI_wide_2_6_depth_1_2.csv  (solo provini con pausa)

Output (salvati in output/)
---------------------------
  scatter_peso_vs_features.png
  r2_confronto.png
  mediazione_test.png
  bedtemp_vs_pause.png
  mediazione_summary.txt
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
from scipy.stats import pearsonr, spearmanr

# ══ CONFIG ══════════════════════════════════════════════════════════════════════
BASE = r"C:\Users\tomga\Desktop\Lab Data science\Dataset_giusti\feature_tables\output"
PATH_ROI = os.path.join(BASE, "feature_table_roi_ROI_wide_2_6_depth_1_2.csv")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR    = os.path.join(SCRIPT_DIR, "output")
os.makedirs(OUT_DIR, exist_ok=True)

PAUSE_COLORS = {10: "#636EFA", 30: "#00CC96", 60: "#FFA15A", 90: "#EF553B"}

# ══ 1. LOAD & PREPARE ═════════════════════════════════════════════════════════════════
df = pd.read_csv(PATH_ROI)
print(f"Loaded {len(df)} specimens")
print(f"Columns: {list(df.columns)}")

Y = "peso_rottura_kg"

# bed_temp_C is a DIRECT IR measurement (temperature of ROI before restart)
# — not a model-fitted parameter — so it belongs in FEAT_DIRECT.
FEAT_DIRECT = ["T1", "T2", "T3", "T4", "T5", "T6", "bed_temp_C"]
FEAT_EXTRAP = ["T0", "heating_rate_t0", "hr_t1", "hr_t2", "hr_t3",
               "delta_T", "A", "alpha", "t_star", "roi_std_spaziale_mean"]
FEAT_CTRL   = ["pausa_s"]
ALL_FEAT    = FEAT_CTRL + FEAT_DIRECT + FEAT_EXTRAP

df = df.loc[:, ~df.columns.duplicated()]

# ══ 2. CORRELAZIONI UNIVARIATE ═════════════════════════════════════════════════════
print("\n" + "="*70)
print("UNIVARIATE CORRELATIONS  (target: peso_rottura_kg)")
print("="*70)
print(f"{'Feature':<25} {'Pearson r':>10} {'p-val':>8} {'Spearman rho':>12} {'p-val':>8}")
print("-"*65)

corr_results = []
for feat in ALL_FEAT:
    if feat not in df.columns:
        print(f"  [SKIP] {feat} not in columns")
        continue
    x = df[feat].values
    y = df[Y].values
    mask = ~(np.isnan(x) | np.isnan(y))
    if mask.sum() < 4:
        continue
    r_p, p_p = pearsonr(x[mask], y[mask])
    r_s, p_s = spearmanr(x[mask], y[mask])
    sig = "***" if p_p < 0.01 else ("**" if p_p < 0.05 else ("*" if p_p < 0.1 else ""))
    print(f"  {feat:<23} {r_p:>10.3f} {p_p:>8.4f} {r_s:>12.3f} {p_s:>8.4f}  {sig}")
    corr_results.append({
        "feature": feat, "pearson_r": r_p, "pearson_p": p_p,
        "spearman_r": r_s, "spearman_p": p_s, "n": int(mask.sum()),
        "category": "ctrl" if feat in FEAT_CTRL else ("direct" if feat in FEAT_DIRECT else "extrap")
    })

df_corr = pd.DataFrame(corr_results).sort_values("pearson_r", key=abs, ascending=False)

# ══ 3. R² CONFRONTO ═══════════════════════════════════════════════════════════════════
def ols_r2(x, y):
    mask = ~(np.isnan(x) | np.isnan(y))
    xm, ym = x[mask], y[mask]
    if len(xm) < 3:
        return np.nan, np.nan, np.nan, np.nan
    slope, intercept, r, p, se = stats.linregress(xm, ym)
    return r**2, intercept, slope, p

def quad_r2(x, y):
    mask = ~(np.isnan(x) | np.isnan(y))
    xm, ym = x[mask], y[mask]
    if len(xm) < 4:
        return np.nan, (np.nan, np.nan, np.nan), np.nan
    
    # Fit quadratic model: y = c2*x^2 + c1*x + c0
    coeffs = np.polyfit(xm, ym, 2)
    y_pred = np.polyval(coeffs, xm)
    
    # Calculate R2
    ss_res = np.sum((ym - y_pred)**2)
    ss_tot = np.sum((ym - np.mean(ym))**2)
    r2 = 1 - ss_res/ss_tot if ss_tot > 0 else np.nan
    
    # Calculate p-value via F-test comparing to mean model
    n = len(xm)
    p_num = 3 # 3 parameters
    df_model = p_num - 1
    df_error = n - p_num
    
    ms_model = (ss_tot - ss_res) / df_model if df_model > 0 else 0
    ms_error = ss_res / df_error if df_error > 0 else 0
    
    if ms_error > 0:
        f_stat = ms_model / ms_error
        p_val = 1 - stats.f.cdf(f_stat, df_model, df_error)
    else:
        p_val = np.nan
        
    return r2, coeffs, p_val

print("\n" + "="*70)
print("R² UNIVARIATO  (OLS & Quad: peso ~ feature)")
print("="*70)
print(f"  {'Feature':<25} {'R²(Lin)':>8} {'p(Lin)':>10} {'R²(Quad)':>10} {'p(Quad)':>10}")
print("-"*80)

r2_results = []
for feat in ALL_FEAT:
    if feat not in df.columns: continue
    r2_lin, _, _, p_lin = ols_r2(df[feat].values, df[Y].values)
    r2_quad, coeffs_quad, p_quad = quad_r2(df[feat].values, df[Y].values)
    if np.isnan(r2_lin): continue
    sig_lin = "***" if p_lin < 0.01 else ("**" if p_lin < 0.05 else ("*" if p_lin < 0.1 else ""))
    sig_quad = "***" if p_quad < 0.01 else ("**" if p_quad < 0.05 else ("*" if p_quad < 0.1 else ""))
    print(f"  {feat:<25} {r2_lin:>8.3f} {p_lin:>10.4f}  {sig_lin:<3} | {r2_quad:>8.3f} {p_quad:>10.4f} {sig_quad}")
    r2_results.append({
        "feature": feat, "R2": r2_lin, "slope_p": p_lin, "R2_quad": r2_quad, "p_quad": p_quad, "coeffs_quad": coeffs_quad,
        "category": "ctrl" if feat in FEAT_CTRL else ("direct" if feat in FEAT_DIRECT else "extrap")
    })

df_r2 = pd.DataFrame(r2_results).sort_values("R2", ascending=False)

# ══ 4. MEDIATION TEST ═══════════════════════════════════════════════════════════════════
def ols_r2_multi(X_mat, y):
    mask = ~np.isnan(y)
    for col in range(X_mat.shape[1]):
        mask &= ~np.isnan(X_mat[:, col])
    Xm, ym = X_mat[mask], y[mask]
    if len(ym) < X_mat.shape[1] + 2: return np.nan
    Xc = np.column_stack([np.ones(len(Xm)), Xm])
    coeffs, _, _, _ = np.linalg.lstsq(Xc, ym, rcond=None)
    y_hat = Xc @ coeffs
    ss_res = np.sum((ym - y_hat)**2)
    ss_tot = np.sum((ym - ym.mean())**2)
    return 1 - ss_res/ss_tot if ss_tot > 0 else np.nan

y_arr     = df[Y].values
pausa_arr = df["pausa_s"].values
r2_pausa  = ols_r2(pausa_arr, y_arr)[0]

print("\n" + "="*70)
print(f"MEDIATION TEST  (baseline R2(peso~pausa_s) = {r2_pausa:.3f})")
print("="*70)
print(f"  {'Feature':<25} {'R2(X)':>7} {'R2(pausa+X)':>13} {'Delta R2':>8} {'Med.':>6}")
print("-"*65)

med_results = []
for feat in FEAT_DIRECT + FEAT_EXTRAP:
    if feat not in df.columns: continue
    r2_x, _, _, _ = ols_r2(df[feat].values, y_arr)
    if np.isnan(r2_x): continue
    r2_both = ols_r2_multi(np.column_stack([pausa_arr, df[feat].values]), y_arr)
    delta   = r2_both - r2_x if not np.isnan(r2_both) else np.nan
    mediator = (r2_x >= r2_pausa * 0.9) and (not np.isnan(delta)) and (delta < 0.05)
    print(f"  {feat:<25} {r2_x:>7.3f} {r2_both:>13.3f} {delta:>8.3f}  {'YES' if mediator else ''}")
    med_results.append({
        "feature": feat, "R2_X": r2_x, "R2_pausa_X": r2_both,
        "delta_pausa": delta, "mediator": mediator,
        "category": "direct" if feat in FEAT_DIRECT else "extrap"
    })

df_med = pd.DataFrame(med_results).sort_values("R2_X", ascending=False)

# ══ 5. PROXY CHECK: bed_temp_C vs pausa_s ════════════════════════════════════════
print("\n" + "="*70)
print("PROXY CHECK: bed_temp_C vs pausa_s")
print("="*70)

bed_r_p, bed_p_p = pearsonr(df["bed_temp_C"].values, df["pausa_s"].values)
bed_r_s, bed_p_s = spearmanr(df["bed_temp_C"].values, df["pausa_s"].values)
r2_bed_pause, _, _, _ = ols_r2(df["pausa_s"].values, df["bed_temp_C"].values)

print(f"Pearson r(bed_temp_C, pausa_s)    = {bed_r_p:.3f}   p = {bed_p_p:.4f}")
print(f"Spearman rho(bed_temp_C, pausa_s) = {bed_r_s:.3f}   p = {bed_p_s:.4f}")
print(f"R²(bed_temp_C ~ pausa_s)          = {r2_bed_pause:.3f}")

intra_stats = (df.groupby("pausa_s")
                 .agg(n=("bed_temp_C","size"), bed_mean=("bed_temp_C","mean"),
                      bed_std=("bed_temp_C","std"), bed_min=("bed_temp_C","min"),
                      bed_max=("bed_temp_C","max")).reset_index())
intra_stats["cv_percent"] = 100 * intra_stats["bed_std"] / intra_stats["bed_mean"]
print("\nWithin-pause variability of bed_temp_C:")
print(intra_stats.round(3).to_string(index=False))

# ══ 6. SAVE SUMMARY TXT ═════════════════════════════════════════════════════════════════
with open(os.path.join(OUT_DIR, "mediazione_summary.txt"), "w") as f:
    f.write("MEDIATION ANALYSIS SUMMARY\n")
    f.write("Target: peso_rottura_kg\n")
    f.write(f"n = {len(df)} specimens (paused only)\n")
    f.write(f"Baseline R²(peso ~ pausa_s) = {r2_pausa:.3f}\n")
    f.write("Feature classification: bed_temp_C = DIRECT IR measure\n\n")
    f.write("=== CORRELATIONS (sorted by |Pearson r|) ===\n")
    f.write(df_corr[["feature","pearson_r","pearson_p","spearman_r","spearman_p"]].to_string(index=False))
    f.write("\n\n=== R² UNIVARIATO (sorted by linear R²) ===\n")
    f.write(df_r2[["feature","R2","slope_p","R2_quad","p_quad","category"]].to_string(index=False))
    f.write("\n\n=== MEDIATION TEST (sorted by R²_X) ===\n")
    f.write(df_med[["feature","R2_X","R2_pausa_X","delta_pausa","mediator","category"]].to_string(index=False))
    f.write("\n\n=== PROXY CHECK: bed_temp_C vs pausa_s ===\n")
    f.write(f"Pearson r = {bed_r_p:.4f}   p = {bed_p_p:.6f}\n")
    f.write(f"Spearman rho = {bed_r_s:.4f}   p = {bed_p_s:.6f}\n")
    f.write(f"R²(bed_temp_C ~ pausa_s) = {r2_bed_pause:.4f}\n\n")
    f.write("Within-pause variability:\n")
    f.write(intra_stats.round(4).to_string(index=False))
    f.write("\n\nInterpretation guide:\n")
    f.write("- R² > 0.90: bed_temp_C is essentially a proxy for pause duration.\n")
    f.write("- R² 0.60-0.90: strong but incomplete; bed_temp_C carries extra specimen-level variance.\n")
    f.write("- R² < 0.60: bed_temp_C is substantially independent of nominal pause duration.\n")
print("\nSaved: mediazione_summary.txt")

# ══ 7. PLOT 1 — R² BARPLOT ═════════════════════════════════════════════════════════════
df_r2_plot = df_r2.sort_values("R2", ascending=True)
cat_colors = {"ctrl": "#888888", "direct": "#636EFA", "extrap": "#EF553B"}
colors_r2  = [cat_colors[c] for c in df_r2_plot["category"]]

fig, ax = plt.subplots(figsize=(10, 7))
fig.patch.set_facecolor("white"); ax.set_facecolor("white")
bars = ax.barh(df_r2_plot["feature"], df_r2_plot["R2"],
               color=colors_r2, edgecolor="white", zorder=3)
for bar, p in zip(bars, df_r2_plot["slope_p"]):
    sig = "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.1 else ""))
    if sig:
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                sig, va="center", fontsize=10, color="#222")
ax.axvline(r2_pausa, color="#222", linewidth=1.8, linestyle="--", zorder=4)
ax.text(r2_pausa + 0.005, len(df_r2_plot) - 0.5,
        f"pausa_s baseline\nR²={r2_pausa:.3f}", fontsize=8, color="#222")
legend_patches = [
    mpatches.Patch(color="#888888", label="Pause duration (baseline)"),
    mpatches.Patch(color="#636EFA", label="Direct IR measurement"),
    mpatches.Patch(color="#EF553B", label="Extrapolated / WLS-fitted feature"),
]
ax.legend(handles=legend_patches, fontsize=9, loc="lower right")
ax.set_xlabel("R²  (OLS: peso_rottura_kg ~ feature)", fontsize=12)
ax.set_title(
    "Which thermal feature best explains breaking load?\n"
    "Features right of the dashed line outperform pause duration alone",
    fontsize=13, pad=10)
ax.grid(True, axis="x", linestyle="--", alpha=0.4)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "r2_confronto_lineare.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved: r2_confronto_lineare.png")

# ══ 7 BIS. PLOT 1 BIS — R² QUADRATICO BARPLOT ════════════════════════════════════
df_r2_plot_quad = df_r2.copy()
df_r2_plot_quad["Max_R2"] = df_r2_plot_quad[["R2", "R2_quad"]].max(axis=1)
df_r2_plot_quad = df_r2_plot_quad.sort_values("Max_R2", ascending=True)

fig, ax = plt.subplots(figsize=(10, 8))
fig.patch.set_facecolor("white"); ax.set_facecolor("white")
y_pos = np.arange(len(df_r2_plot_quad))
height = 0.35

ax.barh(y_pos - height/2, df_r2_plot_quad["R2"], height, label="Linear R²", color="#636EFA", edgecolor="white")
ax.barh(y_pos + height/2, df_r2_plot_quad["R2_quad"], height, label="Quadratic R²", color="#FFA15A", edgecolor="white")

ax.axvline(r2_pausa, color="#222", linewidth=1.8, linestyle="--", zorder=4)
ax.text(r2_pausa + 0.005, len(df_r2_plot_quad) - 0.5,
        f"pausa_s baseline (linear)\nR²={r2_pausa:.3f}", fontsize=8, color="#222")

ax.set_yticks(y_pos)
ax.set_yticklabels(df_r2_plot_quad["feature"], fontsize=10)
ax.legend(fontsize=10, loc="lower right")
ax.set_xlabel("R²", fontsize=12)
ax.set_title(
    "Linear vs Quadratic Regression R²\n"
    "Does a non-linear fit better capture the mechanical behavior?",
    fontsize=13, pad=10)
ax.grid(True, axis="x", linestyle="--", alpha=0.4)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "r2_confronto_quadratico.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved: r2_confronto_quadratico.png")

# ══ 8. PLOT 2 — SCATTER MATRIX (top 6) ═══════════════════════════════════════════════
top_feats = df_r2.head(6)["feature"].tolist()

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.patch.set_facecolor("white")
axes_flat = axes.flatten()

for i, feat in enumerate(top_feats):
    ax = axes_flat[i]
    ax.set_facecolor("white")
    for _, row in df.iterrows():
        p = int(row["pausa_s"])
        ax.scatter(row[feat], row[Y], color=PAUSE_COLORS.get(p, "#888"),
                   s=90, zorder=3, edgecolors="white", linewidths=0.6)
        ax.annotate(row["specimen"].replace("Rec-", "").split("_")[0],
                    (row[feat], row[Y]), textcoords="offset points",
                    xytext=(5, 3), fontsize=7, color="#444")
    xv = df[feat].dropna().values
    yv = df.loc[df[feat].notna(), Y].values
    if len(xv) >= 3:
        slope, intercept, r, p_val, _ = stats.linregress(xv, yv)
        x_line = np.linspace(xv.min(), xv.max(), 100)
        ax.plot(x_line, intercept + slope * x_line, color="#333",
                linewidth=1.5, linestyle="--", alpha=0.7, label=f"Lin R²={r**2:.2f}")
        
        # Add quadratic fit if it improves R2 by at least 0.02
        r2_lin = r**2
        row_r2 = df_r2[df_r2["feature"] == feat].iloc[0]
        r2_quad = row_r2["R2_quad"]
        coeffs_quad = row_r2["coeffs_quad"]
        
        if not np.isnan(r2_quad) and r2_quad > r2_lin + 0.02:
            y_quad = np.polyval(coeffs_quad, x_line)
            ax.plot(x_line, y_quad, color="#EF553B",
                    linewidth=2, linestyle="-", alpha=0.8, label=f"Quad R²={r2_quad:.2f}")
            
        sig = "***" if p_val < 0.01 else ("**" if p_val < 0.05 else ("*" if p_val < 0.1 else ""))
        title_text = f"{feat}\nLin R²={r**2:.3f} p={p_val:.4f} {sig}"
        if not np.isnan(r2_quad):
            title_text += f" | Quad R²={r2_quad:.3f}"
        ax.set_title(title_text, fontsize=9)
        ax.legend(fontsize=8, loc="best")
    ax.set_xlabel(feat, fontsize=9)
    ax.set_ylabel("Breaking load (kg)", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.3)

fig.legend(handles=[mpatches.Patch(color=PAUSE_COLORS[p], label=f"Pause {p}s")
                    for p in [10, 30, 60, 90]],
           loc="lower center", ncol=4, fontsize=10, bbox_to_anchor=(0.5, -0.01))
fig.suptitle("Breaking load vs. top-6 thermal features", fontsize=13, y=1.01)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "scatter_peso_vs_features.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved: scatter_peso_vs_features.png")

# ══ 9. PLOT 3 — MEDIATION CHART ════════════════════════════════════════════════════════
df_mp   = df_med.sort_values("R2_X", ascending=False).head(10)
x_pos   = np.arange(len(df_mp)); width = 0.35

fig, ax = plt.subplots(figsize=(12, 6))
fig.patch.set_facecolor("white"); ax.set_facecolor("white")
ax.bar(x_pos - width/2, df_mp["R2_X"],       width, label="R²(peso ~ X)",
       color="#636EFA", edgecolor="white", zorder=3)
ax.bar(x_pos + width/2, df_mp["R2_pausa_X"], width, label="R²(peso ~ pausa + X)",
       color="#FFA15A", edgecolor="white", zorder=3, alpha=0.85)
ax.axhline(r2_pausa, color="#888", linewidth=1.5, linestyle="--")
ax.text(len(df_mp) - 0.3, r2_pausa + 0.01,
        f"R²(pausa alone) = {r2_pausa:.3f}", fontsize=8, color="#555")
ax.set_xticks(x_pos)
ax.set_xticklabels(df_mp["feature"], rotation=30, ha="right", fontsize=9)
ax.set_ylabel("R²", fontsize=12)
ax.set_title(
    "Mediation test: if orange ≈ blue → pause adds nothing → X is the mediator",
    fontsize=12, pad=10)
ax.legend(fontsize=10); ax.set_ylim(0, 1.05)
ax.grid(True, axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "mediazione_test.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved: mediazione_test.png")

# ══ 10. PLOT 4 — bed_temp_C vs pausa_s ═══════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 6))
fig.patch.set_facecolor("white"); ax.set_facecolor("white")
for _, row in df.iterrows():
    p = int(row["pausa_s"])
    ax.scatter(row["pausa_s"], row["bed_temp_C"],
               color=PAUSE_COLORS.get(p, "#888"), s=100,
               edgecolors="white", linewidths=0.7, zorder=3)
    ax.annotate(row["specimen"].replace("Rec-", "").split("_")[0],
                (row["pausa_s"], row["bed_temp_C"]),
                textcoords="offset points", xytext=(5, 3), fontsize=8, color="#444")
slope_bp, int_bp, _, _, _ = stats.linregress(df["pausa_s"], df["bed_temp_C"])
xl = np.linspace(df["pausa_s"].min(), df["pausa_s"].max(), 100)
ax.plot(xl, int_bp + slope_bp * xl, "--", color="#333", linewidth=1.5)
ax.set_xlabel("Pause duration (s)", fontsize=12)
ax.set_ylabel("bed_temp_C (°C)", fontsize=12)
ax.set_title(
    f"bed_temp_C (direct IR) vs pause duration\n"
    f"Pearson r={bed_r_p:.3f}, Spearman rho={bed_r_s:.3f}, R²={r2_bed_pause:.3f}",
    fontsize=12, pad=10)
ax.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "bedtemp_vs_pause.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved: bedtemp_vs_pause.png")

print(f"\nDone. All outputs in: {OUT_DIR}")
