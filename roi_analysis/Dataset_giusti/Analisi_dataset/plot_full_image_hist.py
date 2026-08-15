"""
plot_full_image_hist.py
------------------------
Per ogni provino genera un istogramma della distribuzione di temperatura
che usa TUTTI I PIXEL di TUTTI I FRAME CORE (intera immagine, non ROI).

FLUSSO:
  Per ogni provino:
    1. Prende tutti i frame core di tutti i layer (tramite frame_selector).
    2. Carica ogni frame come PNG 8-bit e converte in gradi Celsius con
       T = T_MIN + (px / 255) * (T_MAX - T_MIN).
    3. Concatena TUTTI i valori pixel di tutti i frame in un unico array.
    4. Costruisce l'istogramma di questa distribuzione.
    5. Traccia linee verticali per: minimo globale, P01, P05, P10, P25.

Differenza rispetto a plot_ambient_temp.py:
  - plot_ambient_temp.py  => raccoglie UN valore per frame (il minimo globale),
    poi fa l'istogramma di quei minimi. Utile per stimare T_env.
  - plot_full_image_hist.py => raccoglie TUTTI i pixel di tutti i frame,
    poi fa l'istogramma. Mostra la distribuzione termica completa del provino.

Output:
  Analisi_dataset/Full_Image_Hist/<specimen>_full_hist.png   (uno per provino)
  Analisi_dataset/Full_Image_Hist/all_specimens_kde.png      (KDE sovrapposti)

USO:
    python plot_full_image_hist.py
"""

import sys
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from pathlib import Path
from scipy.stats import gaussian_kde
from tqdm import tqdm

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

from config import DATA_DIR          # noqa: E402
from frame_selector import SPECIMENS, get_all_layers  # noqa: E402


# ===========================================================================
#  CONFIG
# ===========================================================================

# Granularita' bin istogramma [gradi C]
BIN_WIDTH = 0.5

# Range X asse temperatura: None = automatico (da tutti i dati)
X_RANGE = None  # es. (40, 200) per forzare un range fisso

# Percentili da mostrare come linee verticali nel grafico per provino
# e da stampare nella tabella riassuntiva.
# P25 = primo quartile, usabile come stima conservativa di T_env.
MARKER_PERCENTILES = [1, 5, 10, 25]

# Provini da escludere sempre
ALWAYS_EXCLUDE = {"Rec-023"}

# Numero massimo di frame per provino da caricare (None = tutti).
# Aumenta il tempo ma da' la distribuzione piu' accurata.
# Suggerimento: None per analisi finale, 50 per test rapido.
MAX_FRAMES_PER_SPECIMEN = None

# ===========================================================================

DATA_PATH   = Path(DATA_DIR)
OUTPUT_BASE = SCRIPT_DIR / "Full_Image_Hist"
OUTPUT_BASE.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Range termico (identico a roi_dataset_builder e plot_ambient_temp)
# ---------------------------------------------------------------------------

def _load_thermal_range() -> tuple[float, float]:
    cache_dir = DATA_PATH / "_cache"
    candidates = list(cache_dir.glob("*meta*.json")) if cache_dir.exists() else []
    for meta_path in candidates:
        try:
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
            t_min = meta.get("temp_min") or meta.get("min_temp") or meta.get("tmin")
            t_max = meta.get("temp_max") or meta.get("max_temp") or meta.get("tmax")
            if t_min is not None and t_max is not None:
                print(f"[INFO] Range termico: T_MIN={t_min}  T_MAX={t_max}")
                return float(t_min), float(t_max)
        except Exception:
            pass
    print("[WARN] global_meta.json non trovato. Fallback T_MIN=41.8  T_MAX=180.2")
    return 41.8, 180.2


T_MIN, T_MAX = _load_thermal_range()


def pixels_to_celsius(px_array: np.ndarray) -> np.ndarray:
    """Converte un array 8-bit in gradi Celsius."""
    return T_MIN + (px_array.astype(np.float32) / 255.0) * (T_MAX - T_MIN)


# ---------------------------------------------------------------------------
# Raccolta pixel: tutti i pixel di tutti i frame core
# ---------------------------------------------------------------------------

def collect_all_pixels(specimen: str) -> np.ndarray:
    """
    Restituisce un array 1-D con i valori di temperatura [°C] di TUTTI i
    pixel di TUTTI i frame core dell'intera immagine (non solo ROI).

    Parametri
    ---------
    specimen : nome del provino (chiave in SPECIMENS)

    Ritorna
    -------
    np.ndarray 1-D float32 — puo' contenere milioni di valori
    """
    try:
        layers = get_all_layers(specimen, margin=0)
    except Exception as e:
        print(f"  [WARN] {specimen}: {e}")
        return np.array([], dtype=np.float32)

    pixel_chunks = []
    loaded = 0

    for layer_info in layers:
        frames_dir = Path(layer_info["frames_dir"])
        for frame_idx in layer_info["frames_core"]:
            if MAX_FRAMES_PER_SPECIMEN is not None and loaded >= MAX_FRAMES_PER_SPECIMEN:
                break

            path = frames_dir / f"{frame_idx:04d}.png"
            if not path.exists():
                continue
            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            pixel_chunks.append(pixels_to_celsius(img).ravel())
            loaded += 1

        if MAX_FRAMES_PER_SPECIMEN is not None and loaded >= MAX_FRAMES_PER_SPECIMEN:
            break

    if not pixel_chunks:
        return np.array([], dtype=np.float32)

    return np.concatenate(pixel_chunks)


# ---------------------------------------------------------------------------
# Palette colori famiglie di pausa
# ---------------------------------------------------------------------------

PAUSE_COLORS = {
    "10s": "#e74c3c",
    "30s": "#27ae60",
    "60s": "#2980b9",
    "90s": "#8e44ad",
    "std": "#7f8c8d",
}


def get_pause_group(name: str) -> str:
    low = name.lower()
    for pause in ["10s", "30s", "60s", "90s"]:
        if pause in low:
            return pause
    return "std"


# ---------------------------------------------------------------------------
# Grafico singolo per provino
# ---------------------------------------------------------------------------

def plot_specimen_hist(
    specimen: str,
    pixels_c: np.ndarray,
    bins: np.ndarray,
    x_kde: np.ndarray,
    color: str,
    n_frames: int,
) -> None:
    """
    Genera e salva l'istogramma per un singolo provino.

    Mostra:
      - Istogramma di densita' di tutti i pixel
      - Curva KDE sovrapposta
      - Linee verticali per minimo globale e percentili di MARKER_PERCENTILES
    """
    fig, ax = plt.subplots(figsize=(11, 6))

    # Istogramma
    ax.hist(
        pixels_c, bins=bins, density=True,
        color=color, alpha=0.30, edgecolor="none",
        label="Istogramma pixel"
    )

    # KDE (downsampling se troppi punti per velocita')
    kde_data = pixels_c
    if len(pixels_c) > 200_000:
        rng = np.random.default_rng(42)
        kde_data = rng.choice(pixels_c, size=200_000, replace=False)
    if len(kde_data) > 5:
        try:
            kde = gaussian_kde(kde_data, bw_method="scott")
            ax.plot(x_kde, kde(x_kde), color=color, linewidth=2.2, label="KDE")
        except Exception:
            pass

    # Linea minimo globale
    t_min_val = float(pixels_c.min())
    ax.axvline(
        t_min_val, color="#1a1a1a", linewidth=1.4,
        linestyle="-", alpha=0.8, label=f"Min = {t_min_val:.1f}\u00b0C"
    )
    ax.text(
        t_min_val + 0.15, 0,
        f"Min\n{t_min_val:.1f}\u00b0C",
        fontsize=7.5, color="#1a1a1a", va="bottom", alpha=0.85,
        transform=ax.get_xaxis_transform()
    )

    # Linee percentili
    linestyles = {1: ":", 5: "--", 10: "-.", 25: "--"}
    alphas     = {1: 0.55, 5: 0.75, 10: 0.65, 25: 0.90}
    colors_p   = {1: "#888", 5: "#555", 10: "#444", 25: "#c0392b"}

    for p in MARKER_PERCENTILES:
        p_val = float(np.percentile(pixels_c, p))
        ls    = linestyles.get(p, "--")
        alp   = alphas.get(p, 0.65)
        cp    = colors_p.get(p, "#555")

        ax.axvline(
            p_val, color=cp, linewidth=1.2,
            linestyle=ls, alpha=alp,
            label=f"P{p:02d} = {p_val:.1f}\u00b0C"
        )
        ax.text(
            p_val + 0.15, 0.05 + 0.08 * MARKER_PERCENTILES.index(p),
            f"P{p:02d}\n{p_val:.1f}\u00b0C",
            fontsize=7, color=cp, va="bottom", alpha=alp,
            transform=ax.get_xaxis_transform()
        )

    ax.set_xlabel("Temperatura [\u00b0C]", fontsize=11)
    ax.set_ylabel(f"Densit\u00e0  (bin = {BIN_WIDTH}\u00b0C)", fontsize=11)
    ax.set_title(
        f"{specimen}  \u2014  distribuzione termica intera immagine\n"
        f"{n_frames} frame core  |  {len(pixels_c):,} pixel  |  "
        f"bin = {BIN_WIDTH}\u00b0C",
        fontsize=11, pad=10
    )
    ax.grid(True, linestyle="--", alpha=0.3, linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)

    ax.legend(
        fontsize=9, loc="upper right",
        framealpha=0.9, edgecolor="#cccccc"
    )

    plt.tight_layout()
    out_path = OUTPUT_BASE / f"{specimen}_full_hist.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {out_path.name}")


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    all_specimens = [
        s for s, info in SPECIMENS.items()
        if info["valid"]
        and info.get("layers_front") is not None
        and s not in ALWAYS_EXCLUDE
    ]

    print(f"Provini da analizzare ({len(all_specimens)}):")
    for s in all_specimens:
        print(f"  - {s}")
    print()

    # ------------------------------------------------------------------
    # Raccolta pixel per ogni provino
    # ------------------------------------------------------------------
    specimen_pixels: dict[str, np.ndarray] = {}
    specimen_nframes: dict[str, int]       = {}

    for spec in tqdm(all_specimens, desc="Caricamento frame"):
        layers = []
        try:
            layers = get_all_layers(spec, margin=0)
        except Exception:
            pass
        n_frames_expected = sum(len(l["frames_core"]) for l in layers)

        px = collect_all_pixels(spec)
        if len(px) > 0:
            # stima frame caricati (ogni frame = H*W pixel, es. 320*240=76800)
            # usiamo la lunghezza dell'array / pixels per frame
            specimen_pixels[spec]  = px
            specimen_nframes[spec] = n_frames_expected
            print(
                f"  {spec:<25}  {n_frames_expected:>4} frame  "
                f"{len(px):>12,} pixel  "
                f"min={px.min():.1f}  P25={np.percentile(px, 25):.1f}  "
                f"median={np.median(px):.1f}\u00b0C"
            )
        else:
            print(f"  {spec:<25}  [NESSUN FRAME]")

    if not specimen_pixels:
        print("[ERR] Nessun dato raccolto. Controlla DATA_DIR e i path dei frame.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Tabella percentili riassuntiva
    # ------------------------------------------------------------------
    SUMMARY_P = [1, 5, 10, 25, 50, 75]
    print(f"\n{'=' * 78}")
    print("  Percentili distribuzione pixel intera immagine per provino [\u00b0C]")
    print(f"{'=' * 78}")
    header = f"  {'Provino':<25}" + "".join(f"  P{p:02d} " for p in SUMMARY_P)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for spec, px in specimen_pixels.items():
        row = f"  {spec:<25}"
        for p in SUMMARY_P:
            row += f"  {np.percentile(px, p):>5.1f}"
        print(row)

    # ------------------------------------------------------------------
    # Range globale per assi uniformi
    # ------------------------------------------------------------------
    all_px_global = np.concatenate(list(specimen_pixels.values()))
    x_lo = X_RANGE[0] if X_RANGE else float(np.floor(all_px_global.min()) - 0.5)
    x_hi = X_RANGE[1] if X_RANGE else float(np.ceil(all_px_global.max())  + 0.5)
    bins  = np.arange(x_lo, x_hi + BIN_WIDTH, BIN_WIDTH)
    x_kde = np.linspace(x_lo, x_hi, 800)

    # ------------------------------------------------------------------
    # Grafici singoli per provino
    # ------------------------------------------------------------------
    print("\nGenerazione grafici singoli...")
    for spec, px in specimen_pixels.items():
        pg    = get_pause_group(spec)
        color = PAUSE_COLORS.get(pg, "#555555")
        plot_specimen_hist(
            spec, px, bins, x_kde,
            color=color,
            n_frames=specimen_nframes.get(spec, 0)
        )

    # ------------------------------------------------------------------
    # Grafico KDE sovrapposto (tutti i provini)
    # ------------------------------------------------------------------
    print("\nGenerazione grafico KDE sovrapposto...")
    fig_all, ax_all = plt.subplots(figsize=(14, 7))
    legend_handles = []
    kde_sample_size = 300_000

    for spec, px in specimen_pixels.items():
        pg    = get_pause_group(spec)
        color = PAUSE_COLORS.get(pg, "#555555")

        kde_data = px
        if len(px) > kde_sample_size:
            rng      = np.random.default_rng(42)
            kde_data = rng.choice(px, size=kde_sample_size, replace=False)

        try:
            kde = gaussian_kde(kde_data, bw_method="scott")
            ax_all.plot(
                x_kde, kde(x_kde),
                color=color, linewidth=1.8, alpha=0.80
            )
        except Exception:
            pass

        p25 = float(np.percentile(px, 25))
        ax_all.axvline(p25, color=color, linewidth=0.8,
                       linestyle="--", alpha=0.45)

        handle = mlines.Line2D(
            [], [], color=color, linewidth=2.0,
            label=f"{spec}  [P25={p25:.1f}\u00b0C]"
        )
        legend_handles.append(handle)

    ax_all.set_xlim(x_lo, x_hi)
    ax_all.set_xlabel("Temperatura [\u00b0C]", fontsize=11)
    ax_all.set_ylabel("Densit\u00e0 KDE", fontsize=11)
    ax_all.set_title(
        "Distribuzione KDE temperatura intera immagine — tutti i provini sovrapposti\n"
        "(tutti i pixel di tutti i frame core)  \u2014  linee tratteggiate = P25",
        fontsize=12, pad=12
    )
    ax_all.grid(True, linestyle="--", alpha=0.35, linewidth=0.6)
    ax_all.spines[["top", "right"]].set_visible(False)
    ax_all.legend(
        handles=legend_handles, loc="upper right",
        fontsize=8, framealpha=0.85, edgecolor="#cccccc"
    )
    plt.tight_layout()
    out_all = OUTPUT_BASE / "all_specimens_kde.png"
    fig_all.savefig(out_all, dpi=150, bbox_inches="tight")
    plt.close(fig_all)
    print(f"[OK] KDE sovrapposto salvato in: {out_all}")

    print("\nDone. Controlla Full_Image_Hist/ per i grafici.")
