"""
bed_temp_at_pause_end.py

Estrae la temperatura media e la deviazione standard spaziale della
superficie di stampa alla fine della pausa per tutti i provini NON-std
(quelli che hanno restart_frame != None in SPECIMENS).

I dati vengono letti direttamente da:
  - frame_selector.SPECIMENS  -> lista provini, stop_frame, restart_frame, seq_name
  - config.DATA_DIR            -> cartella radice con le sottocartelle {seq_name}_frames/

  Struttura cartelle:
    Lab Data science/          <- root del progetto (qui sta config.py)
      Dataset_giusti/
        creazione_dataset/     <- questo script
          frame_selector.py

Logica ROI:
  - Provini con "G3" nel nome  -> usano il poligono definito su Rec-G3_S60_1
  - Tutti gli altri             -> usano il poligono definito su Rec-022_10s_1
  Il poligono viene selezionato interattivamente UNA VOLTA per tipo,
  salvato in datasets/roi_bed_definitions.json e riusato per tutti.

Range termico:
  Letto da DATA_DIR/_cache/*meta*.json
  Fallback: 41.8 / 180.2

Selezione poligono (finestra interattiva):
  Click sinistro  -> aggiungi punto (cerchio giallo visibile)
  C               -> chiudi il poligono (unisce ultimo -> primo)
  U               -> undo ultimo punto (o riapre se gia' chiuso)
  R               -> reset completo
  Scroll mouse    -> zoom in/out centrato sul cursore
  Enter           -> conferma (solo se gia' chiuso)
  ESC             -> annulla

Output:
  creazione_dataset/datasets/bed_temp_end_pause.csv
    Colonne: provino | group | restart_frame | frame_used |
             bed_temp_C | bed_temp_std_C | polygon_pts
    bed_temp_C     : temperatura media [°C] della ROI bed al frame_used
    bed_temp_std_C : deviazione standard spaziale [°C] dei pixel della ROI bed
  creazione_dataset/datasets/roi_bed_definitions.json
  creazione_dataset/datasets/Bed_temp_distribution/  (istogrammi per provino + riepilogo)

Uso:
  python Dataset_giusti/creazione_dataset/bed_temp_at_pause_end.py
  python Dataset_giusti/creazione_dataset/bed_temp_at_pause_end.py --redefine-roi
  python Dataset_giusti/creazione_dataset/bed_temp_at_pause_end.py --preview-only
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

SCRIPT_DIR   = Path(__file__).resolve().parent        # .../Dataset_giusti/creazione_dataset/
DATASET_DIR  = SCRIPT_DIR.parent                      # .../Dataset_giusti/
PROJECT_ROOT = DATASET_DIR.parent                     # .../Lab Data science/  <- qui sta config.py

for _p in (str(PROJECT_ROOT), str(DATASET_DIR), str(SCRIPT_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from config import DATA_DIR           # Lab Data science/config.py
from frame_selector import SPECIMENS  # Dataset_giusti/creazione_dataset/frame_selector.py

DATA_DIR = Path(DATA_DIR)

# =============================================================================
# PERCORSI OUTPUT
# =============================================================================

DATASETS_DIR = SCRIPT_DIR / "datasets"
OUTPUT_CSV   = DATASETS_DIR / "bed_temp_end_pause.csv"
ROI_JSON     = DATASETS_DIR / "roi_bed_definitions.json"
HIST_DIR     = DATASETS_DIR / "Bed_temp_distribution"

ROI_REF = {
    "g3":    "Rec-G3_S60_1",
    "other": "Rec-022_10s_1",
}


def _data_range(vals: np.ndarray, pad_frac: float = 0.05) -> tuple[float, float]:
    """
    Calcola xmin/xmax dai dati reali con un padding proporzionale.
    pad_frac=0.05 aggiunge il 5% del range da ciascun lato.
    Arrotonda a .5 °C per avere tick puliti sull'asse.
    """
    lo, hi = float(vals.min()), float(vals.max())
    span   = max(hi - lo, 0.1)          # evita degenere se tutti i pixel = stesso valore
    pad    = span * pad_frac
    xmin   = np.floor((lo - pad) * 2) / 2   # arrotonda al .5 inferiore
    xmax   = np.ceil( (hi + pad) * 2) / 2   # arrotonda al .5 superiore
    return float(xmin), float(xmax)


# =============================================================================
# RANGE TERMICO (per la mappatura pixel -> celsius)
# =============================================================================

def _load_thermal_range() -> tuple[float, float]:
    cache_dir  = DATA_DIR / "_cache"
    candidates = list(cache_dir.glob("*meta*.json")) if cache_dir.exists() else []
    for meta_path in candidates:
        try:
            meta  = json.loads(meta_path.read_text(encoding="utf-8"))
            t_min = meta.get("temp_min") or meta.get("min_temp") or meta.get("tmin")
            t_max = meta.get("temp_max") or meta.get("max_temp") or meta.get("tmax")
            if t_min is not None and t_max is not None:
                print(f"[INFO] Range termico da {meta_path.name}: T_MIN={t_min}  T_MAX={t_max}")
                return float(t_min), float(t_max)
        except Exception as e:
            print(f"[WARN] {meta_path}: {e}")
    print("[WARN] global_meta.json non trovato. Fallback T_MIN=41.8  T_MAX=180.2")
    return 41.8, 180.2


T_MIN, T_MAX = _load_thermal_range()


def pixels_to_celsius(px: np.ndarray) -> np.ndarray:
    return T_MIN + (px.astype(np.float32) / 255.0) * (T_MAX - T_MIN)


# =============================================================================
# HELPERS
# =============================================================================

def _group(name: str) -> str:
    return "g3" if "G3" in name.upper() else "other"


def _load_frame(seq_name: str, frame_index: int) -> np.ndarray | None:
    frames_dir = DATA_DIR / f"{seq_name}_frames"
    frame_path = frames_dir / f"{frame_index:04d}.png"
    if not frame_path.exists():
        print(f"[WARN] Frame non trovato: {frame_path}")
        return None
    img = cv2.imread(str(frame_path))
    if img is None:
        print(f"[WARN] Impossibile leggere: {frame_path}")
    return img


def _polygon_to_mask(polygon: list, shape: tuple) -> np.ndarray:
    mask = np.zeros(shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(polygon, dtype=np.int32)], 1)
    return mask.astype(bool)


def _extract_bed_temp(frame_bgr: np.ndarray, polygon: list) -> tuple[float, float, np.ndarray]:
    """
    Estrae media e std spaziale [°C] della ROI bed in un frame.

    Ritorna
    -------
    (mean_C, std_C, celsius_vals)
      mean_C      : temperatura media della ROI [°C]
      std_C       : deviazione standard spaziale dei pixel della ROI [°C]
      celsius_vals: array 1-D con le temperature di tutti i pixel della ROI
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY) if frame_bgr.ndim == 3 else frame_bgr.copy()
    mask = _polygon_to_mask(polygon, gray.shape)
    vals = gray[mask]
    if vals.size == 0:
        return float("nan"), float("nan"), np.array([], dtype=np.float32)
    celsius = pixels_to_celsius(vals)
    return float(np.mean(celsius)), float(np.std(celsius)), celsius


# =============================================================================
# POLYGON SELECTOR
# =============================================================================

class PolygonSelector:
    """
    Finestra OpenCV interattiva per disegnare un poligono libero.

    Ogni punto cliccato appare subito come cerchio giallo con numero.
    Una linea tratteggiata segue il cursore fino al prossimo click.
    Quando il poligono e' chiuso, l'area viene riempita in verde semitrasparente.

    Controlli:
      Click sinistro  -> aggiungi punto
      C               -> chiudi il poligono
      U               -> undo ultimo punto (o riapre se chiuso)
      R               -> reset
      Scroll mouse    -> zoom in/out centrato sul cursore (1x - 16x)
      Enter           -> conferma (solo se chiuso con >= 3 punti)
      ESC             -> annulla
    """

    ZOOM_STEP   = 1.15
    ZOOM_MIN    = 1.0
    ZOOM_MAX    = 16.0

    LINE_COLOR   = (0,   230,   0)
    FILL_COLOR   = (0,   200,   0)
    DOT_COLOR    = (0,   220, 255)
    DOT_BORDER   = (0,     0,   0)
    CURSOR_COLOR = (180, 180, 180)
    CLOSE_COLOR  = (0,   255, 255)
    NUM_COLOR    = (0,     0,   0)

    DOT_R        = 6
    DOT_BORDER_T = 2

    def __init__(self, frame_bgr: np.ndarray, title: str):
        if frame_bgr.ndim == 2:
            frame_bgr = cv2.cvtColor(frame_bgr, cv2.COLOR_GRAY2BGR)
        elif frame_bgr.shape[2] == 1:
            frame_bgr = cv2.cvtColor(frame_bgr, cv2.COLOR_GRAY2BGR)

        if frame_bgr.dtype != np.uint8:
            f = frame_bgr.astype(np.float32)
            f = (f - f.min()) / (f.max() - f.min() + 1e-9) * 255
            frame_bgr = f.astype(np.uint8)

        h, w = frame_bgr.shape[:2]
        scale = min(4, max(1, 600 // max(h, 1)))
        if scale > 1:
            frame_bgr = cv2.resize(frame_bgr, (w * scale, h * scale),
                                   interpolation=cv2.INTER_NEAREST)
            self._scale = scale
        else:
            self._scale = 1

        self.orig      = frame_bgr.copy()
        self.title     = title
        self.points    : list[tuple[int, int]] = []
        self.closed    = False
        self.confirmed = False
        self._zoom     = 1.0
        self._pan_x    = 0.0
        self._pan_y    = 0.0
        self._mx       = 0
        self._my       = 0

    def _render(self) -> np.ndarray:
        h, w = self.orig.shape[:2]
        dw   = max(1, int(w * self._zoom))
        dh   = max(1, int(h * self._zoom))
        zoomed = cv2.resize(self.orig, (dw, dh), interpolation=cv2.INTER_LINEAR)

        win_w = min(1400, dw)
        win_h = min(900,  dh)
        px0   = int(np.clip(self._pan_x * self._zoom, 0, max(0, dw - win_w)))
        py0   = int(np.clip(self._pan_y * self._zoom, 0, max(0, dh - win_h)))
        self._pan_x = px0 / self._zoom
        self._pan_y = py0 / self._zoom

        canvas = zoomed[py0:py0+win_h, px0:px0+win_w].copy()

        def to_disp(ox, oy):
            return int(ox * self._zoom) - px0, int(oy * self._zoom) - py0

        pts_d = [to_disp(*p) for p in self.points]

        if self.closed and len(pts_d) >= 3:
            overlay = canvas.copy()
            cv2.fillPoly(overlay, [np.array(pts_d, dtype=np.int32)], self.FILL_COLOR)
            cv2.addWeighted(overlay, 0.35, canvas, 0.65, 0, canvas)

        for i in range(len(pts_d) - 1):
            cv2.line(canvas, pts_d[i], pts_d[i+1], self.LINE_COLOR, 2, cv2.LINE_AA)
        if self.closed and len(pts_d) >= 2:
            cv2.line(canvas, pts_d[-1], pts_d[0], self.CLOSE_COLOR, 2, cv2.LINE_AA)
        elif pts_d:
            cur = (self._mx - px0, self._my - py0)
            _draw_dashed_line(canvas, pts_d[-1], cur, self.CURSOR_COLOR)

        for i, (dx, dy) in enumerate(pts_d):
            cv2.circle(canvas, (dx, dy), self.DOT_R + self.DOT_BORDER_T,
                       self.DOT_BORDER, -1, cv2.LINE_AA)
            cv2.circle(canvas, (dx, dy), self.DOT_R, self.DOT_COLOR, -1, cv2.LINE_AA)
            label = str(i + 1)
            fs    = 0.35
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
            cv2.putText(canvas, label,
                        (dx - tw // 2, dy + th // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, fs, self.NUM_COLOR, 1, cv2.LINE_AA)

        status = "CHIUSO - premi Enter per confermare" if self.closed else f"{len(self.points)} punti"
        hint   = "Click->pt | C->chiudi | U->undo | R->reset | scroll->zoom | Enter->ok | ESC->annulla"
        bar_h  = 30
        cv2.rectangle(canvas,
                      (0, canvas.shape[0] - bar_h),
                      (canvas.shape[1], canvas.shape[0]),
                      (20, 20, 20), -1)
        cv2.putText(canvas,
                    f"Zoom {self._zoom:.1f}x | {status}",
                    (5, canvas.shape[0] - bar_h + 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (100, 220, 100), 1, cv2.LINE_AA)
        cv2.putText(canvas, hint,
                    (5, canvas.shape[0] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.33, (160, 160, 160), 1, cv2.LINE_AA)
        return canvas

    def _on_mouse(self, event, x, y, flags, _):
        self._mx = x + int(self._pan_x * self._zoom)
        self._my = y + int(self._pan_y * self._zoom)

        if event == cv2.EVENT_LBUTTONDOWN and not self.closed:
            ox = int(x / self._zoom + self._pan_x)
            oy = int(y / self._zoom + self._pan_y)
            h, w = self.orig.shape[:2]
            ox = int(np.clip(ox, 0, w - 1))
            oy = int(np.clip(oy, 0, h - 1))
            self.points.append((ox, oy))
            print(f"  Punto {len(self.points)}: ({ox}, {oy})")

        elif event == cv2.EVENT_MOUSEWHEEL:
            direction = 1 if flags > 0 else -1
            old_z     = self._zoom
            self._zoom = float(np.clip(old_z * (self.ZOOM_STEP ** direction),
                                       self.ZOOM_MIN, self.ZOOM_MAX))
            ox_c = x / old_z + self._pan_x
            oy_c = y / old_z + self._pan_y
            self._pan_x = ox_c - x / self._zoom
            self._pan_y = oy_c - y / self._zoom

    def run(self) -> list[tuple[int, int]] | None:
        cv2.namedWindow(self.title, cv2.WINDOW_NORMAL)
        h, w = self.orig.shape[:2]
        cv2.resizeWindow(self.title, min(1400, w), min(900, h))
        cv2.setMouseCallback(self.title, self._on_mouse)

        print(f"\n{'='*65}")
        print(f"  Disegna il contorno della base: {self.title}")
        print(f"  Click->punto | C->chiudi | U->undo | R->reset"
              f" | scroll->zoom | Enter->conferma | ESC->annulla")
        print(f"{'='*65}")

        while True:
            cv2.imshow(self.title, self._render())
            key = cv2.waitKeyEx(20)

            if key in (13, 10):
                if self.closed and len(self.points) >= 3:
                    self.confirmed = True
                    break
                else:
                    print("  [!] Prima chiudi il poligono con C (min 3 punti).")
            elif key == 27:
                print("  Selezione annullata.")
                break
            elif key in (ord('c'), ord('C')):
                if len(self.points) >= 3 and not self.closed:
                    self.closed = True
                    print(f"  Poligono chiuso ({len(self.points)} punti).")
                elif len(self.points) < 3:
                    print("  [!] Servono almeno 3 punti.")
            elif key in (ord('u'), ord('U')):
                if self.closed:
                    self.closed = False
                    print("  Poligono riaperto.")
                elif self.points:
                    print(f"  Undo: rimosso {self.points.pop()}.")
            elif key in (ord('r'), ord('R')):
                self.points.clear()
                self.closed = False
                print("  Reset.")

        cv2.destroyAllWindows()
        if self.confirmed:
            pts_orig = [(int(x / self._scale), int(y / self._scale))
                        for x, y in self.points]
            print(f"  Confermato: {len(pts_orig)} punti.")
            return pts_orig
        return None


def _draw_dashed_line(img, pt1, pt2, color, dash=8, gap=5, thickness=1):
    x1, y1 = pt1
    x2, y2 = pt2
    dist = np.hypot(x2 - x1, y2 - y1)
    if dist < 1:
        return
    dx = (x2 - x1) / dist
    dy = (y2 - y1) / dist
    step = dash + gap
    for i in range(int(dist / step) + 1):
        s = i * step
        e = min(s + dash, dist)
        cv2.line(img,
                 (int(x1 + dx * s), int(y1 + dy * s)),
                 (int(x1 + dx * e), int(y1 + dy * e)),
                 color, thickness, cv2.LINE_AA)


# =============================================================================
# MASK PREVIEW
# =============================================================================

def _show_mask_preview(name: str, frame_bgr: np.ndarray,
                       polygon: list, frame_index: int, bed_temp: float) -> bool:
    if frame_bgr.ndim == 2:
        canvas = cv2.cvtColor(frame_bgr, cv2.COLOR_GRAY2BGR)
    else:
        canvas = frame_bgr.copy()

    if canvas.dtype != np.uint8:
        f = canvas.astype(np.float32)
        f = (f - f.min()) / (f.max() - f.min() + 1e-9) * 255
        canvas = f.astype(np.uint8)

    h, w = canvas.shape[:2]
    scale = min(4, max(1, 600 // max(h, 1)))
    if scale > 1:
        canvas = cv2.resize(canvas, (w * scale, h * scale),
                            interpolation=cv2.INTER_NEAREST)

    pts_scaled = [(int(x * scale), int(y * scale)) for x, y in polygon]
    pts_arr    = np.array(pts_scaled, dtype=np.int32)

    overlay = canvas.copy()
    cv2.fillPoly(overlay, [pts_arr], (0, 200, 0))
    cv2.addWeighted(overlay, 0.40, canvas, 0.60, 0, canvas)
    cv2.polylines(canvas, [pts_arr], isClosed=True,
                  color=(0, 255, 255), thickness=2, lineType=cv2.LINE_AA)

    for i, (px, py) in enumerate(pts_scaled):
        cv2.circle(canvas, (px, py), 7, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(canvas, (px, py), 5, (0, 220, 255), -1, cv2.LINE_AA)
        label = str(i + 1)
        fs    = 0.32
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
        cv2.putText(canvas, label, (px - tw // 2, py + th // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), 1, cv2.LINE_AA)

    bar_h = 26
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], bar_h), (20, 20, 20), -1)
    temp_str = f"{bed_temp:.1f} C" if not np.isnan(bed_temp) else "NaN"
    cv2.putText(canvas,
                f"{name}  |  frame {frame_index}  |  bed_temp = {temp_str}",
                (6, bar_h - 7), cv2.FONT_HERSHEY_SIMPLEX,
                0.48, (100, 255, 100), 1, cv2.LINE_AA)

    bot_h = 22
    cv2.rectangle(canvas,
                  (0, canvas.shape[0] - bot_h),
                  (canvas.shape[1], canvas.shape[0]),
                  (20, 20, 20), -1)
    cv2.putText(canvas, "[qualsiasi tasto] prossimo provino   [ESC] salta tutti",
                (6, canvas.shape[0] - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 160), 1, cv2.LINE_AA)

    win = f"Preview: {name}"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, min(1200, canvas.shape[1]), min(800, canvas.shape[0]))
    cv2.imshow(win, canvas)
    key = cv2.waitKeyEx(0)
    cv2.destroyWindow(win)
    return key != 27


# =============================================================================
# ROI DEFINITION
# =============================================================================

def _get_or_define_polygon(group: str, roi_store: dict, redefine: bool) -> list | None:
    if group in roi_store and not redefine:
        return [tuple(p) for p in roi_store[group]]

    ref_name = ROI_REF[group]
    spec     = SPECIMENS[ref_name]
    restart  = spec["restart_frame"]
    if restart is None:
        print(f"[ERRORE] Il provino di riferimento '{ref_name}' non ha restart_frame.")
        return None

    target_frame = max(0, restart - 1)
    frame_bgr    = _load_frame(spec["seq_name"], target_frame)
    if frame_bgr is None:
        return None

    print(f"\n[ROI] Definizione poligono per gruppo '{group}' usando '{ref_name}' (frame {target_frame})")
    sel     = PolygonSelector(frame_bgr, f"ROI '{group}' — {ref_name} (frame {target_frame})")
    polygon = sel.run()
    if polygon is None:
        return None

    roi_store[group] = [list(p) for p in polygon]
    return polygon


# =============================================================================
# ISTOGRAMMI  —  range calcolato dai dati reali con padding 5%
# =============================================================================

def _save_histogram(name: str, celsius_vals: np.ndarray, n_bins: int = 40) -> None:
    """
    Istogramma con range determinato dai dati del singolo provino.
    Il padding del 5% garantisce che le barre non tocchino i bordi.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
    except ImportError:
        print("[WARN] matplotlib non disponibile, istogrammi saltati.")
        return

    if celsius_vals.size == 0:
        return

    HIST_DIR.mkdir(parents=True, exist_ok=True)

    xmin, xmax = _data_range(celsius_vals, pad_frac=0.05)
    mean_v     = float(np.mean(celsius_vals))
    std_v      = float(np.std(celsius_vals))

    fig, ax = plt.subplots(figsize=(8, 4))

    counts, bin_edges = np.histogram(celsius_vals, bins=n_bins, range=(xmin, xmax))
    bin_w = bin_edges[1] - bin_edges[0]
    ax.bar(bin_edges[:-1], counts, width=bin_w * 0.92,
           color="#2196F3", edgecolor="white", linewidth=0.3, alpha=0.85)

    ax.axvline(mean_v, color="#E53935", linewidth=1.8, linestyle="--",
               label=f"Media = {mean_v:.2f} °C\nStd  = {std_v:.2f} °C")

    ax.set_xlabel("Temperatura (°C)", fontsize=11)
    ax.set_ylabel("Conteggio pixel", fontsize=11)
    ax.set_title(f"Distribuzione temperatura bed  —  {name}",
                 fontsize=12, fontweight="bold")
    ax.set_xlim(xmin, xmax)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()

    out_path = HIST_DIR / f"hist_{name}.png"
    fig.savefig(str(out_path), dpi=120)
    plt.close(fig)
    print(f"  [hist] {out_path.name}  (range {xmin:.1f}-{xmax:.1f} °C, "
          f"mean={mean_v:.2f}, std={std_v:.2f})")


def _save_summary_plot(all_data: dict) -> None:
    """
    Grafico riepilogativo con range globale calcolato su TUTTI i provini.
    Pannello sinistro: KDE sovrapposte.
    Pannello destro:   boxplot affiancati con media (rombo rosso) sovrapposta.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from scipy.stats import gaussian_kde
    except ImportError as e:
        print(f"[WARN] Impossibile creare summary plot: {e}")
        return

    if not all_data:
        return

    # Range globale su tutti i valori
    all_concat = np.concatenate(list(all_data.values()))
    gxmin, gxmax = _data_range(all_concat, pad_frac=0.05)

    cmap  = plt.get_cmap("tab20")
    names = sorted(all_data.keys())
    n     = len(names)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # --- Pannello sinistro: KDE ---
    ax_kde = axes[0]
    x_grid = np.linspace(gxmin, gxmax, 600)

    for i, nm in enumerate(names):
        vals = all_data[nm]
        if vals.size < 5:
            continue
        color = cmap(i / max(n - 1, 1))
        try:
            kde = gaussian_kde(vals, bw_method="scott")
            ax_kde.plot(x_grid, kde(x_grid), color=color, linewidth=1.8, label=nm)
        except Exception:
            pass

    ax_kde.set_xlabel("Temperatura (°C)", fontsize=11)
    ax_kde.set_ylabel("Densità stimata (KDE)", fontsize=11)
    ax_kde.set_title("Distribuzione temperatura bed  —  tutti i provini (KDE)",
                     fontsize=12, fontweight="bold")
    ax_kde.set_xlim(gxmin, gxmax)
    ax_kde.grid(axis="both", linestyle="--", alpha=0.35)
    ax_kde.legend(fontsize=8, ncol=max(1, n // 10), loc="best", framealpha=0.7)

    # --- Pannello destro: boxplot ---
    ax_box = axes[1]
    data_list  = [all_data[nm] for nm in names if all_data[nm].size > 0]
    names_plot = [nm for nm in names if all_data[nm].size > 0]

    bp = ax_box.boxplot(data_list, vert=True, patch_artist=True,
                        medianprops=dict(color="black", linewidth=2))
    for patch, nm in zip(bp["boxes"], names_plot):
        i = names.index(nm)
        patch.set_facecolor(cmap(i / max(n - 1, 1)))
        patch.set_alpha(0.75)

    # Aggiunge la MEDIA per ogni provino come rombo rosso
    means = [float(np.mean(all_data[nm])) for nm in names_plot]
    ax_box.scatter(
        range(1, len(names_plot) + 1),
        means,
        marker="D",
        color="red",
        s=50,
        zorder=5,
        label="Media"
    )
    ax_box.legend(fontsize=9, loc="best")

    ax_box.set_xticks(range(1, len(names_plot) + 1))
    ax_box.set_xticklabels(names_plot, rotation=45, ha="right", fontsize=8)
    ax_box.set_ylabel("Temperatura (°C)", fontsize=11)
    ax_box.set_ylim(gxmin, gxmax)
    ax_box.set_title("Boxplot temperatura bed  —  tutti i provini",
                     fontsize=12, fontweight="bold")
    ax_box.grid(axis="y", linestyle="--", alpha=0.35)

    fig.tight_layout()
    out_path = HIST_DIR / "summary_all_provini.png"
    fig.savefig(str(out_path), dpi=130)
    plt.close(fig)
    print(f"\n[hist] Grafico riepilogativo -> {out_path}  "
          f"(range globale {gxmin:.1f}-{gxmax:.1f} °C)")


# =============================================================================
# MAIN
# =============================================================================

def main(redefine_roi: bool = False, preview_only: bool = False):
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)

    roi_store: dict = {}
    if ROI_JSON.exists() and not redefine_roi:
        roi_store = json.loads(ROI_JSON.read_text(encoding="utf-8"))
        print(f"[ROI] Caricati da {ROI_JSON}: {list(roi_store.keys())}")

    pause_specimens = {
        name: spec
        for name, spec in SPECIMENS.items()
        if spec["valid"] and spec.get("restart_frame") is not None
    }

    print(f"[INFO] Provini con pausa: {sorted(pause_specimens.keys())}")
    print(f"[INFO] T_MIN={T_MIN}  T_MAX={T_MAX}")
    print(f"[INFO] Istogrammi: range dinamico calcolato sui dati reali (+5% padding)")

    if not preview_only:
        needed_groups = {_group(name) for name in pause_specimens}
        for grp in sorted(needed_groups):
            if grp not in roi_store or redefine_roi:
                polygon = _get_or_define_polygon(grp, roi_store, redefine_roi)
                if polygon is None:
                    print(f"[ERRORE] Impossibile definire poligono per '{grp}'. Uscita.")
                    sys.exit(1)
        ROI_JSON.write_text(json.dumps(roi_store, indent=2), encoding="utf-8")
        print(f"[ROI] Salvati in {ROI_JSON}")

    rows: list[dict] = []
    all_px_celsius: dict = {}
    do_preview = True

    for name, spec in sorted(pause_specimens.items()):
        restart_frame = spec["restart_frame"]
        target_frame  = max(0, restart_frame - 1)
        seq_name      = spec["seq_name"]
        grp           = _group(name)

        if grp not in roi_store:
            print(f"[SKIP] {name}: poligono per '{grp}' non definito")
            continue

        polygon   = [tuple(p) for p in roi_store[grp]]
        frame_bgr = _load_frame(seq_name, target_frame)

        if frame_bgr is None:
            bed_temp     = float("nan")
            bed_temp_std = float("nan")
            celsius_vals = np.array([], dtype=np.float32)
        else:
            bed_temp, bed_temp_std, celsius_vals = _extract_bed_temp(frame_bgr, polygon)
            print(f"  ok  {name:35s}  group={grp}  "
                  f"frame={target_frame}  bed_temp={bed_temp:.2f} C  std={bed_temp_std:.2f} C")

        rows.append({
            "provino":            name,
            "group":              grp,
            "restart_frame":      restart_frame,
            "frame_used":         target_frame,
            "bed_temp_C":         round(bed_temp,     3) if not np.isnan(bed_temp)     else float("nan"),
            "bed_temp_std_C":     round(bed_temp_std, 3) if not np.isnan(bed_temp_std) else float("nan"),
            "polygon_pts":        json.dumps([list(p) for p in polygon]),
        })

        if celsius_vals.size > 0 and not preview_only:
            all_px_celsius[name] = celsius_vals
            _save_histogram(name, celsius_vals)

        if do_preview and frame_bgr is not None:
            do_preview = _show_mask_preview(name, frame_bgr, polygon, target_frame, bed_temp)

    if not preview_only:
        if all_px_celsius:
            _save_summary_plot(all_px_celsius)

        df = pd.DataFrame(rows)
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"\n[DONE] {len(df)} righe -> {OUTPUT_CSV}")
        print(df[["provino", "group", "frame_used", "bed_temp_C", "bed_temp_std_C"]].to_string(index=False))
    else:
        print("\n[preview-only] Nessun CSV ne' istogrammi scritti.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--redefine-roi", action="store_true",
                    help="Forza ri-selezione poligoni anche se gia' salvati")
    ap.add_argument("--preview-only", action="store_true",
                    help="Mostra solo le preview senza ricalcolare il CSV")
    args = ap.parse_args()
    main(redefine_roi=args.redefine_roi, preview_only=args.preview_only)
