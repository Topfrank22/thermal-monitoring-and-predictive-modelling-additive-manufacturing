"""
Test di indipendenza sui dataset STD
=====================================
Ipotesi 1: i layer sono indipendenti tra loro
  -> ACF + Ljung-Box sulla serie T_mean_layer (una per ogni provino STD)

Ipotesi 2: i frame dentro ogni layer sono indipendenti tra loro
  -> ACF + Ljung-Box su T_ROI(t) frame-per-frame, per ogni (provino, layer)
  -> Output paginato: se i layer sono > MAX_COLS_PER_PAGE vengono suddivisi
     in piu' figure (es. page1, page2, ...) per mantenere la leggibilita'

Filtri applicati al caricamento:
  - solo frame_type == "core"
  - solo roi_complete == True  (ROI completamente dentro il pezzo)

Output salvati in: test_ipotesi/risultati/
  - acf_layer_<dataset>.png                    : ACF tra layer per ogni dataset STD
  - acf_frames_<dataset>_p<N>.png              : ACF intra-layer paginata
  - heatmap_pvalue_<dataset>.png               : heatmap p-value Ljung-Box intra-layer
  - ljung_box_risultati.txt                    : tutti i p-value Ljung-Box

Uso:
    python test_ipotesi/test_indipendenza.py
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIGURAZIONE PATHS
# ─────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(
    SCRIPT_DIR, "..", "..", "creazione_dataset", "datasets"
)
RISULTATI_DIR = os.path.join(SCRIPT_DIR, "risultati")
os.makedirs(RISULTATI_DIR, exist_ok=True)

# Dataset STD disponibili (aggiungere qui nuovi file _STD.csv)
STD_FILES = {
    "ROI_wide_2_6_depth_1_2":  "ROI_wide_2_6_depth_1_2_STD.csv",
    "ROI_wide_3_7_depth_1_2":  "ROI_wide_3_7_depth_1_2_STD.csv",
    "ROI_wide_3_10_depth_1_3": "ROI_wide_3_10_depth_1_3_STD.csv",
}

# Colonna temperatura da usare
TEMP_COL = "roi_mean_C"

# Numero massimo di lag per ACF
MAX_LAGS_LAYER = 10   # tra layer
MAX_LAGS_FRAME = 20   # dentro un layer

# Soglia significativita'
ALPHA = 0.05

# Numero massimo di colonne (layer) per pagina nella griglia ACF intra-layer.
# Con dataset STD che hanno fino a 29 layer, imposta un valore ragionevole
# per mantenere i subplot leggibili (es. 8 layer per pagina).
MAX_COLS_PER_PAGE = 8


# ─────────────────────────────────────────────
# UTILITA'
# ─────────────────────────────────────────────
def safe_lags(n, max_lags):
    """Restituisce n_lags sicuro: min(max_lags, n//2 - 1, n-2)."""
    return max(1, min(max_lags, n // 2 - 1, n - 2))


def ljung_box_test(series, lags):
    """
    Esegue Ljung-Box e restituisce il p-value al lag specificato.
    Ritorna NaN se la serie e' troppo corta o costante.
    """
    series = np.array(series, dtype=float)
    if len(series) < 4 or np.std(series) < 1e-10:
        return np.nan
    lags = safe_lags(len(series), lags)
    try:
        res = acorr_ljungbox(series, lags=[lags], return_df=True)
        return float(res["lb_pvalue"].iloc[-1])
    except Exception:
        return np.nan


def load_std_dataset(filepath, txt_lines):
    """
    Carica un CSV STD, applica i filtri obbligatori e logga le righe scartate.
    Filtri:
      1. frame_type == "core"
      2. roi_complete == True
    """
    df = pd.read_csv(filepath)
    n_totale = len(df)
    log = [f"  Righe totali nel CSV: {n_totale}"]

    if "frame_type" in df.columns:
        df = df[df["frame_type"] == "core"].copy()
        log.append(f"  Dopo filtro frame_type='core': {len(df)} righe "
                   f"(scartate: {n_totale - len(df)})")
    else:
        log.append("  [WARN] Colonna 'frame_type' assente, nessun filtro applicato.")

    n_dopo_core = len(df)

    if "roi_complete" in df.columns:
        df = df[df["roi_complete"] == True].copy()
        scartate = n_dopo_core - len(df)
        log.append(f"  Dopo filtro roi_complete=True: {len(df)} righe "
                   f"(scartate: {scartate})")
        if scartate > 0:
            pct = scartate / n_dopo_core * 100
            log.append(f"  -> {pct:.1f}% dei frame core aveva ROI parziale")
    else:
        log.append("  [WARN] Colonna 'roi_complete' assente.")

    log.append(f"  Righe usate per i test: {len(df)}")
    for line in log:
        print(line)
        txt_lines.append(line)

    return df


# ─────────────────────────────────────────────
# IPOTESI 1: LAYER INDIPENDENTI
# ─────────────────────────────────────────────
def test_ipotesi1_layer(df, dataset_name, txt_lines):
    """
    Per ogni provino STD, costruisce la serie T_mean per layer
    e applica ACF + Ljung-Box.
    Funziona con qualsiasi numero di layer (dinamico).
    """
    print(f"\n{'='*60}")
    print(f"IPOTESI 1 — Layer indipendenti | {dataset_name}")
    print(f"{'='*60}")

    txt_lines.append(f"\n{'='*60}")
    txt_lines.append(f"IPOTESI 1 — Layer indipendenti | {dataset_name}")
    txt_lines.append(f"{'='*60}")

    specimens = sorted(df["specimen"].unique())
    n_cols = len(specimens)
    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 4), squeeze=False)
    fig.suptitle(
        f"ACF tra layer — {dataset_name}\n"
        f"Ipotesi 1: layer indipendenti  |  filtro: core + roi_complete",
        fontsize=11, y=1.02
    )

    for col_idx, spec in enumerate(specimens):
        sub = df[df["specimen"] == spec]
        layer_means = (
            sub.groupby("layer_index")[TEMP_COL]
            .mean()
            .sort_index()
        )
        series = layer_means.values
        n = len(series)
        lags = safe_lags(n, MAX_LAGS_LAYER)

        ax = axes[0][col_idx]
        plot_acf(series, lags=lags, ax=ax, alpha=ALPHA, title="")
        ax.set_title(f"{spec}\n(N layer = {n})", fontsize=9)
        ax.set_ylabel("ACF")
        ax.axhline(0, color="black", linewidth=0.5)

        pval = ljung_box_test(series, lags)
        conclusione = (
            "NON RIFIUTA H0 (indipendenza ragionevole)"
            if pval > ALPHA else
            "RIFIUTA H0 (autocorrelazione significativa!)"
        )
        msg = f"  {spec}: N={n}, Ljung-Box p={pval:.4f} -> {conclusione}"
        print(msg)
        txt_lines.append(msg)
        ax.set_xlabel(f"Lag (layer) | LB p={pval:.3f}")

    plt.tight_layout()
    out_path = os.path.join(RISULTATI_DIR, f"acf_layer_{dataset_name}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Grafico salvato: {out_path}")


# ─────────────────────────────────────────────
# IPOTESI 2: FRAME INDIPENDENTI DENTRO IL LAYER
# ─────────────────────────────────────────────
def test_ipotesi2_frame(df, dataset_name, txt_lines):
    """
    Per ogni (provino, layer), costruisce la serie T_ROI(t) frame per frame
    e applica ACF + Ljung-Box.

    Output paginato: se il numero di layer supera MAX_COLS_PER_PAGE,
    la griglia ACF viene suddivisa in piu' file PNG:
        acf_frames_<dataset>_p1.png, _p2.png, ...
    Questo evita figure illeggibili per provini STD con 20+ layer.

    La heatmap p-value mostra sempre tutti i layer su un'unica figura
    con altezza adattiva.
    """
    print(f"\n{'='*60}")
    print(f"IPOTESI 2 — Frame indipendenti dentro layer | {dataset_name}")
    print(f"{'='*60}")

    txt_lines.append(f"\n{'='*60}")
    txt_lines.append(f"IPOTESI 2 — Frame indipendenti dentro layer | {dataset_name}")
    txt_lines.append(f"{'='*60}")

    specimens = sorted(df["specimen"].unique())
    layers    = sorted(df["layer_index"].unique())
    n_rows    = len(specimens)
    n_layers  = len(layers)

    print(f"  Provini: {specimens}")
    print(f"  Layer totali: {n_layers} -> {layers}")

    # Raccoglie tutti i p-value per la heatmap e il txt
    pval_matrix = np.full((n_rows, n_layers), np.nan)

    # ── Griglia ACF paginata ──────────────────────────────────────
    # Suddivide i layer in gruppi di MAX_COLS_PER_PAGE colonne per pagina.
    layer_chunks = [
        layers[i : i + MAX_COLS_PER_PAGE]
        for i in range(0, n_layers, MAX_COLS_PER_PAGE)
    ]
    n_pages = len(layer_chunks)

    for page_idx, layer_chunk in enumerate(layer_chunks):
        n_cols_page = len(layer_chunk)
        page_num    = page_idx + 1

        # Altezza adattiva: 3.5 inch per riga, min 4 inch totale
        fig_h = max(4, 3.5 * n_rows)
        fig_w = max(6, 4.5 * n_cols_page)

        fig, axes = plt.subplots(
            n_rows, n_cols_page,
            figsize=(fig_w, fig_h),
            squeeze=False
        )

        layer_range_str = f"L{layer_chunk[0]}–L{layer_chunk[-1]}"
        page_str = f"pagina {page_num}/{n_pages}  ({layer_range_str})"

        fig.suptitle(
            f"ACF intra-layer — {dataset_name}\n"
            f"Ipotesi 2: frame indipendenti  |  filtro: core + roi_complete\n"
            f"{page_str}",
            fontsize=10, y=1.02
        )

        for r, spec in enumerate(specimens):
            for c, layer in enumerate(layer_chunk):
                # indice globale del layer per pval_matrix
                global_c = layers.index(layer)

                sub = df[
                    (df["specimen"] == spec) &
                    (df["layer_index"] == layer)
                ].sort_values("frame_idx")

                series = sub[TEMP_COL].values
                ax = axes[r][c]

                if len(series) < 4:
                    ax.set_title(
                        f"{spec}\nL{layer}\n"
                        f"({'insuff.' if len(series) > 0 else 'nessun dato'})",
                        fontsize=7
                    )
                    ax.axis("off")
                    msg = (f"  {spec} Layer {layer}: N={len(series)} "
                           f"— skippato (troppo pochi frame)")
                    print(msg)
                    txt_lines.append(msg)
                    continue

                lags = safe_lags(len(series), MAX_LAGS_FRAME)
                plot_acf(series, lags=lags, ax=ax, alpha=ALPHA, title="")
                ax.set_title(f"{spec} | L{layer} (N={len(series)})", fontsize=7)
                ax.set_ylabel("ACF", fontsize=7)

                pval = ljung_box_test(series, lags)
                pval_matrix[r, global_c] = pval
                conclusione = "OK" if pval > ALPHA else "AUTOCORR!"
                ax.set_xlabel(f"Lag | LB p={pval:.3f} [{conclusione}]", fontsize=7)
                ax.tick_params(labelsize=6)

                msg = (
                    f"  {spec} Layer {layer}: N={len(series)}, "
                    f"Ljung-Box p={pval:.4f} -> "
                    f"{'NON RIFIUTA H0' if pval > ALPHA else 'RIFIUTA H0!'}"
                )
                print(msg)
                txt_lines.append(msg)

        plt.tight_layout()
        suffix = f"_p{page_num}" if n_pages > 1 else ""
        out_acf = os.path.join(
            RISULTATI_DIR, f"acf_frames_{dataset_name}{suffix}.png"
        )
        fig.savefig(out_acf, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Grafico ACF salvato ({page_str}): {out_acf}")

    # ── Heatmap p-value (tutti i layer, altezza adattiva) ─────────
    # Con molti layer la heatmap viene allargata orizzontalmente;
    # con molti provini viene allungata verticalmente.
    fig_w_hm = max(8, n_layers * 1.1)
    fig_h_hm = max(3, n_rows * 1.0 + 2)

    fig2, ax2 = plt.subplots(figsize=(fig_w_hm, fig_h_hm))
    im = ax2.imshow(
        pval_matrix, aspect="auto",
        cmap="RdYlGn", vmin=0, vmax=1
    )
    plt.colorbar(im, ax=ax2, label="p-value Ljung-Box")
    ax2.set_xticks(range(n_layers))
    ax2.set_xticklabels([f"L{l}" for l in layers], rotation=45, ha="right", fontsize=7)
    ax2.set_yticks(range(n_rows))
    ax2.set_yticklabels(specimens, fontsize=8)
    ax2.set_title(
        f"Heatmap p-value Ljung-Box intra-layer — {dataset_name}  "
        f"({n_layers} layer)\n"
        f"Verde p>{ALPHA} (indipendenza ok) | Rosso p<{ALPHA} (autocorrelazione)\n"
        f"Grigio = layer senza dati dopo filtro roi_complete",
        fontsize=9
    )

    # Annota i valori nelle celle solo se la heatmap non e' troppo densa
    if n_layers <= 30:
        for r in range(n_rows):
            for c in range(n_layers):
                if not np.isnan(pval_matrix[r, c]):
                    ax2.text(
                        c, r, f"{pval_matrix[r, c]:.2f}",
                        ha="center", va="center",
                        fontsize=6 if n_layers > 15 else 7,
                        color="black" if pval_matrix[r, c] > 0.15 else "white"
                    )

    plt.tight_layout()
    out_heatmap = os.path.join(
        RISULTATI_DIR, f"heatmap_pvalue_{dataset_name}.png"
    )
    fig2.savefig(out_heatmap, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"  Heatmap p-value salvata: {out_heatmap}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    txt_lines = [
        "RISULTATI TEST DI INDIPENDENZA — DATASET STD",
        "=" * 60,
        f"Colonna analizzata: {TEMP_COL}",
        f"Soglia significativita': alpha = {ALPHA}",
        f"Max layer per pagina ACF: MAX_COLS_PER_PAGE = {MAX_COLS_PER_PAGE}",
        "",
        "FILTRI APPLICATI:",
        "  1. frame_type == 'core'   (scarta frame pre/post layer)",
        "  2. roi_complete == True   (scarta ROI parziali fuori dal pezzo)",
        "",
        "STRUTTURA DEI TEST:",
        "  Ipotesi 1: layer indipendenti -> ACF + Ljung-Box su T_mean per layer",
        "  Ipotesi 2: frame indipendenti dentro layer -> ACF + Ljung-Box su T_ROI(t)",
        "",
        "LETTURA DEI RISULTATI:",
        "  p-value > 0.05 -> NON si rifiuta H0 (indipendenza ragionevole)",
        "  p-value < 0.05 -> SI rifiuta H0 (autocorrelazione significativa)",
        "",
    ]

    for dataset_name, filename in STD_FILES.items():
        filepath = os.path.join(DATASET_DIR, filename)
        if not os.path.exists(filepath):
            msg = f"[SKIP] File non trovato: {filepath}"
            print(msg)
            txt_lines.append(msg)
            continue

        print(f"\n{'='*60}")
        print(f"Caricamento: {filename}")
        txt_lines.append(f"\n{'='*60}")
        txt_lines.append(f"Dataset: {dataset_name}")

        df = load_std_dataset(filepath, txt_lines)

        if df.empty:
            msg = "  [ERRORE] Nessuna riga rimasta dopo i filtri."
            print(msg)
            txt_lines.append(msg)
            continue

        print(f"  Provini: {list(df['specimen'].unique())}")
        print(f"  Layer disponibili: {sorted(df['layer_index'].unique())}")

        test_ipotesi1_layer(df, dataset_name, txt_lines)
        test_ipotesi2_frame(df, dataset_name, txt_lines)

    out_txt = os.path.join(RISULTATI_DIR, "ljung_box_risultati.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(txt_lines))
    print(f"\nRisultati testuali salvati: {out_txt}")
    print("\nDone. Tutti i risultati sono in:", RISULTATI_DIR)


if __name__ == "__main__":
    main()
