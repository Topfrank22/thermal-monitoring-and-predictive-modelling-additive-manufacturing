"""
roi_dataset_builder.py
-----------------------
Legge nozzle_db.csv, calcola la ROI per ogni frame e costruisce un dataset
analitco configurabile.

Flusso:
    1. [Opzionale] Visualizer interattivo: mostra la ROI frame per frame,
       navigabile con frecce <- ->. Premi Q per uscire e procedere.
    2. Pipeline: per ogni (specimen, layer_index, obs_index) estrae i pixel
       della ROI dal frame e calcola le metriche configurate.
    3. Salva il dataset CSV finale.

===========================================================================
CONFIG  -  tutto quello che vuoi cambiare e' qui
===========================================================================

ROI_PARAMS_STD / ROI_PARAMS_G3:
    Definiscono la ROI come offset in pixel rispetto alla PUNTA dell'ugello.
    ROI_PARAMS_STD : provini senza 'G3' nel nome (kernel_2)
    ROI_PARAMS_G3  : provini con    'G3' nel nome (kernel_3)

    Gli assi seguono la convenzione delle coordinate immagine:
        x positivo = destra
        y positivo = basso

    Il rettangolo visualizzato include i bordi: i pixel sul bordo
    del rettangolo fanno parte della ROI ed entrano nel calcolo delle metriche.

METRICS:
    Dizionario { nome_colonna: funzione(pixel_array_2D_celsius) -> valore }.
    pixel_array e' un np.ndarray 2D shape (n_righe, n_colonne) in gradi Celsius.
    Riga 0 = y_start, colonna 0 = x_start.

OSS_FILTER:
    'core' -> solo frame_type == 'core' (default)
    'all'  -> pre + core + post

DATASET_NAME:
    Nome del file CSV di output (senza estensione).

ROI VALIDITY (bordi pezzo):
    Legge piece_boundaries.json generato da piece_boundaries.py.
    Per ogni specimen sono definite due colonne verticali:
        x_left  : colonna piu' a sinistra del pezzo (inclusa)
        x_right : colonna piu' a destra  del pezzo (inclusa)

    Colonne aggiunte al dataset:
        roi_valid_frac  : float [0-1], frazione pixel ROI dentro i bordi pezzo
        roi_complete    : bool, True se roi_valid_frac == 1.0

    Se piece_boundaries.json non esiste o lo specimen non e' presente,
    roi_valid_frac = NaN e roi_complete = None.

    Genera piece_boundaries.json con:
        python piece_boundaries.py
"""

import sys
import json
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT  = SCRIPT_DIR.parent

sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(REPO_ROOT))

from config import DATA_DIR
from Dataset_giusti.creazione_dataset.frame_selector import get_layer_frames


# ===========================================================================
# Carica T_MIN / T_MAX dal global_meta.json
# ===========================================================================

def _load_thermal_range(data_dir: str | Path) -> tuple[float, float]:
    cache_dir = Path(data_dir) / "_cache"
    candidates = list(cache_dir.glob("*meta*.json")) if cache_dir.exists() else []
    for meta_path in candidates:
        try:
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
            t_min = meta.get("temp_min") or meta.get("min_temp") or meta.get("tmin")
            t_max = meta.get("temp_max") or meta.get("max_temp") or meta.get("tmax")
            if t_min is not None and t_max is not None:
                print(f"[INFO] Range termico da: {meta_path.name}  T_MIN={t_min}  T_MAX={t_max}")
                return float(t_min), float(t_max)
        except Exception as e:
            print(f"[WARN] {meta_path}: {e}")
    print("[WARN] global_meta.json non trovato. Fallback T_MIN=41.8  T_MAX=180.2")
    return 41.8, 180.2


T_MIN, T_MAX = _load_thermal_range(DATA_DIR)


# ===========================================================================
# Carica piece_boundaries.json
# ===========================================================================

BOUNDARIES_PATH = SCRIPT_DIR / "datasets" / "piece_boundaries.json"

def _load_boundaries(path: Path) -> dict:
    """
    Legge piece_boundaries.json.
    Restituisce dict {specimen: {"x_left": int, "x_right": int}}.
    Se il file non esiste restituisce {} con un warning.
    """
    if not path.exists():
        print(f"[WARN] {path} non trovato. "
              "roi_valid_frac sara' NaN. Esegui: python piece_boundaries.py")
        return {}
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    print(f"[INFO] piece_boundaries: {len(data)} specimen -> {list(data.keys())}")
    return data


PIECE_BOUNDARIES: dict = _load_boundaries(BOUNDARIES_PATH)


# ===========================================================================
# CONFIG
# ===========================================================================

NOZZLE_DB_PATH = SCRIPT_DIR / "datasets" / "nozzle_db.csv"
DATASET_NAME   = "ROI_wide_3_10_depth_1_3"
OUTPUT_DIR     = SCRIPT_DIR / "datasets"

# ---------------------------------------------------------------------------
# Offset ugello -> ROI  (positivo = basso / destra)
# Cambia questi valori per calibrare la ROI sui due gruppi di video
# ---------------------------------------------------------------------------

offset_ugello_std = 1   # provini standard (senza G3 nel nome, kernel_2)
offset_ugello_g3  = 0   # provini G3       (con G3 nel nome,    kernel_3)
X_START = 3
X_END   = 10
Y_START = 1
Y_END = 3


ROI_PARAMS_STD = {
    "x_start":  X_START,
    "x_end":    X_END,
    "y_start":  Y_START + offset_ugello_std,
    "y_end":    Y_END + offset_ugello_std,
}

ROI_PARAMS_G3 = {
    "x_start":  X_START,
    "x_end":   X_END,
    "y_start":  Y_START + offset_ugello_g3,
    "y_end":    Y_END + offset_ugello_g3,
}

OSS_FILTER = "core"

METRICS = {
    "roi_mean_C":     lambda px: float(np.mean(px)),
    "roi_max_C":      lambda px: float(np.max(px)),
    "roi_min_C":      lambda px: float(np.min(px)),
    "roi_std_C":      lambda px: float(np.std(px)),
    "roi_median_C":   lambda px: float(np.median(px)),
    "roi_pixels_raw": lambda px: ",".join(f"{v:.2f}" for v in px.flatten()),
}

# --- Visualizer ---
ENABLE_VISUALIZER     = True
VISUALIZER_MAX_FRAMES = None
VISUALIZER_SCALE      = 1.0
ZOOM_STEP             = 0.15
ZOOM_MIN              = 1.0
ZOOM_MAX              = 20.0

# ===========================================================================


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def select_roi_params(specimen_name: str,
                      params_std: dict = ROI_PARAMS_STD,
                      params_g3:  dict = ROI_PARAMS_G3) -> dict:
    """Restituisce ROI_PARAMS_G3 se il nome contiene 'G3', altrimenti ROI_PARAMS_STD."""
    return params_g3 if "G3" in specimen_name else params_std


def pixels_to_celsius(px_array: np.ndarray) -> np.ndarray:
    return T_MIN + (px_array.astype(np.float32) / 255.0) * (T_MAX - T_MIN)


def extract_roi(frame, tip_x, tip_y, x_start, x_end, y_start, y_end):
    h, w = frame.shape
    r0 = int(np.clip(tip_y + y_start, 0, h - 1))
    r1 = int(np.clip(tip_y + y_end,   0, h - 1))
    c0 = int(np.clip(tip_x + x_start, 0, w - 1))
    c1 = int(np.clip(tip_x + x_end,   0, w - 1))
    if r1 < r0 or c1 < c0:
        return None
    return frame[r0:r1+1, c0:c1+1]


def roi_rect(tip_x, tip_y, x_start, x_end, y_start, y_end, frame_shape):
    h, w = frame_shape
    r0 = int(np.clip(tip_y + y_start, 0, h - 1))
    r1 = int(np.clip(tip_y + y_end,   0, h - 1))
    c0 = int(np.clip(tip_x + x_start, 0, w - 1))
    c1 = int(np.clip(tip_x + x_end,   0, w - 1))
    return c0, r0, c1, r1


def load_frame_gray(frames_dir: Path, frame_idx: int):
    path = frames_dir / f"{frame_idx:04d}.png"
    if not path.exists():
        return None
    return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)


def build_frame_list(df_nozzle: pd.DataFrame, oss_filter: str) -> list:
    df = df_nozzle[df_nozzle["frame_type"] == "core"].copy() \
         if oss_filter == "core" else df_nozzle.copy()
    items = []
    for (specimen, layer_idx), group in df.groupby(["specimen", "layer_index"], sort=False):
        layer_info = get_layer_frames(specimen, int(layer_idx), margin=0)
        frames_dir = layer_info["frames_dir"]
        for obs_idx, (_, row) in enumerate(group.iterrows(), start=1):
            items.append((row.to_dict(), frames_dir, obs_idx))
    return items


# ---------------------------------------------------------------------------
# ROI validity: bordi pezzo
# ---------------------------------------------------------------------------

def compute_roi_validity(
    c0: int, c1: int,
    specimen: str,
    boundaries: dict,
) -> tuple[float, bool | None]:
    """
    Calcola roi_valid_frac e roi_complete in base ai bordi del pezzo.

    Returns
    -------
    (roi_valid_frac, roi_complete)
        roi_valid_frac : float [0.0, 1.0] - frazione colonne ROI dentro i bordi
        roi_complete   : bool  - True se tutte le colonne sono dentro i bordi
                         None  se lo specimen non ha boundaries definiti
    """
    if specimen not in boundaries:
        return float("nan"), None

    x_left  = boundaries[specimen]["x_left"]
    x_right = boundaries[specimen]["x_right"]

    cols  = np.arange(c0, c1 + 1)
    valid = (cols >= x_left) & (cols <= x_right)
    frac  = float(valid.sum() / len(cols))
    return round(frac, 4), bool(frac == 1.0)


# ---------------------------------------------------------------------------
# VISUALIZER
# ---------------------------------------------------------------------------

def run_visualizer(
    frame_items: list,
    boundaries: dict,
    params_std: dict = ROI_PARAMS_STD,
    params_g3:  dict = ROI_PARAMS_G3,
    max_frames: int | None = VISUALIZER_MAX_FRAMES,
    scale: float = VISUALIZER_SCALE,
) -> None:
    """
    Visualizer interattivo OpenCV.

    Navigazione:
        Freccia destra / D  : frame successivo
        Freccia sinistra / A: frame precedente
        Freccia su / W      : +10 frame
        Freccia giu / S     : -10 frame
        Rotella mouse       : zoom in/out centrato sul cursore (min 1x)
        + / =               : zoom in  (tastiera)
        -                   : zoom out (tastiera)
        Q / ESC             : chiudi e procedi

    Colore rettangolo ROI:
        Verde  = roi_complete (tutta la ROI dentro i bordi del pezzo)
        Giallo = parzialmente dentro
        Rosso  = completamente fuori  (o boundaries non definiti)

    Il pan NON si resetta al cambio frame.
    """
    items = frame_items if max_frames is None else frame_items[:max_frames]
    if not items:
        print("[VISUALIZER] Nessun frame.")
        return

    state = {
        "scale":  max(scale, ZOOM_MIN),
        "pan_x":  0.0,
        "pan_y":  0.0,
        "redraw": True,
    }

    win = "ROI Visualizer  |  A/D: naviga   W/S: +-10   rotella: zoom   Q: esci"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEWHEEL:
            old_s = state["scale"]
            new_s = round(min(old_s + ZOOM_STEP, ZOOM_MAX), 3) if flags > 0 \
                    else round(max(old_s - ZOOM_STEP, ZOOM_MIN), 3)
            if new_s == old_s:
                return
            orig_x = (x + state["pan_x"]) / old_s
            orig_y = (y + state["pan_y"]) / old_s
            state["pan_x"] = orig_x * new_s - x
            state["pan_y"] = orig_y * new_s - y
            state["scale"] = new_s
            state["redraw"] = True

    cv2.setMouseCallback(win, on_mouse)

    idx         = 0
    need_render = True
    base_frame  = None
    h_f = w_f   = 1

    while True:
        if need_render:
            row_dict, frames_dir, obs_idx = items[idx]
            frame = load_frame_gray(frames_dir, int(row_dict["frame_idx"]))

            if frame is None:
                base_frame = np.zeros((240, 320, 3), dtype=np.uint8)
                cv2.putText(base_frame, "FILE MANCANTE", (60, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                specimen = row_dict["specimen"]
                tip_x    = int(row_dict["tip_x"])
                tip_y    = int(row_dict["tip_y"])

                # Selezione ROI params in base al gruppo
                roi_params = select_roi_params(specimen, params_std, params_g3)

                c0, r0, c1, r1 = roi_rect(
                    tip_x, tip_y,
                    roi_params["x_start"], roi_params["x_end"],
                    roi_params["y_start"], roi_params["y_end"],
                    frame.shape,
                )
                base_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

                # Disegna bordi pezzo se disponibili
                if specimen in boundaries:
                    xl = boundaries[specimen]["x_left"]
                    xr = boundaries[specimen]["x_right"]
                    h_vis = base_frame.shape[0]
                    cv2.line(base_frame, (xl, 0), (xl, h_vis - 1), (0, 255, 0), 1)
                    cv2.line(base_frame, (xr, 0), (xr, h_vis - 1), (0, 100, 255), 1)

                # Validita' ROI
                frac, complete = compute_roi_validity(c0, c1, specimen, boundaries)
                if complete is True:
                    rect_color = (0, 220, 0)    # verde
                elif complete is False and frac > 0:
                    rect_color = (0, 200, 255)  # giallo
                else:
                    rect_color = (0, 0, 255)    # rosso

                cv2.rectangle(base_frame, (c0, r0), (c1, r1), rect_color, 1)
                cv2.circle(base_frame, (tip_x, tip_y), 2, (0, 255, 255), -1)

                layer     = int(row_dict["layer_index"])
                ftype     = row_dict["frame_type"]
                fidx      = int(row_dict["frame_idx"])
                conf      = float(row_dict["confidence"])
                direction = row_dict.get("direction_deg", float("nan"))
                dir_str   = f"{direction:.1f}" if not pd.isna(direction) else "NaN"
                frac_str  = f"{frac:.2f}" if not np.isnan(frac) else "N/A"
                grp_label = "G3" if "G3" in specimen else "STD"
                hud = [
                    f"{idx+1}/{len(items)}",
                    f"{specimen}  L{layer}  obs{obs_idx}  [{ftype}]  [{grp_label}]",
                    f"frame:{fidx}  conf:{conf:.3f}  dir:{dir_str}deg",
                    f"tip:({tip_x},{tip_y})",
                    f"ROI x:[{c0}-{c1}] y:[{r0}-{r1}]  ({c1-c0+1}x{r1-r0+1}px)",
                    f"roi_valid:{frac_str}  complete:{complete}",
                    f"offset_y:{roi_params['y_start']}..{roi_params['y_end']}",
                ]
                for i, line in enumerate(hud):
                    yp = 16 + i * 17
                    cv2.putText(base_frame, line, (8, yp), cv2.FONT_HERSHEY_SIMPLEX,
                                0.40, (20, 20, 20), 3, cv2.LINE_AA)
                    cv2.putText(base_frame, line, (8, yp), cv2.FONT_HERSHEY_SIMPLEX,
                                0.40, (230, 230, 230), 1, cv2.LINE_AA)

            h_f, w_f = base_frame.shape[:2]
            sc = state["scale"]
            state["pan_x"] = float(np.clip(state["pan_x"], 0, max(0.0, w_f * sc - w_f)))
            state["pan_y"] = float(np.clip(state["pan_y"], 0, max(0.0, h_f * sc - h_f)))
            need_render     = False
            state["redraw"] = True

        if state["redraw"]:
            sc = state["scale"]
            scaled = cv2.resize(base_frame, None, fx=sc, fy=sc,
                                interpolation=cv2.INTER_NEAREST)
            hs, ws = scaled.shape[:2]
            px = int(np.clip(state["pan_x"], 0, max(0, ws - w_f)))
            py = int(np.clip(state["pan_y"], 0, max(0, hs - h_f)))
            state["pan_x"] = float(px)
            state["pan_y"] = float(py)
            view_w = min(w_f, ws - px)
            view_h = min(h_f, hs - py)
            view   = scaled[py:py+view_h, px:px+view_w].copy()
            zoom_label = f"zoom {sc:.2f}x"
            cv2.putText(view, zoom_label, (6, view_h - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 3, cv2.LINE_AA)
            cv2.putText(view, zoom_label, (6, view_h - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 255, 100), 1, cv2.LINE_AA)
            cv2.imshow(win, view)
            state["redraw"] = False

        key = cv2.waitKeyEx(30)
        if key == -1:
            continue
        if key in (ord('q'), ord('Q'), 27):
            break
        elif key in (ord('d'), ord('D'), 2555904, 65363):
            idx = min(idx + 1, len(items) - 1)
            need_render = True
        elif key in (ord('a'), ord('A'), 2424832, 65361):
            idx = max(idx - 1, 0)
            need_render = True
        elif key in (ord('w'), ord('W'), 2490368, 65362):
            idx = min(idx + 10, len(items) - 1)
            need_render = True
        elif key in (ord('s'), ord('S'), 2621440, 65364):
            idx = max(idx - 10, 0)
            need_render = True
        elif key in (ord('+'), ord('=')):
            old_s = state["scale"]
            new_s = round(min(old_s + ZOOM_STEP, ZOOM_MAX), 3)
            cx = state["pan_x"] + w_f / 2
            cy = state["pan_y"] + h_f / 2
            state["pan_x"] = cx / old_s * new_s - w_f / 2
            state["pan_y"] = cy / old_s * new_s - h_f / 2
            state["scale"] = new_s
            state["redraw"] = True
        elif key == ord('-'):
            old_s = state["scale"]
            new_s = round(max(old_s - ZOOM_STEP, ZOOM_MIN), 3)
            cx = state["pan_x"] + w_f / 2
            cy = state["pan_y"] + h_f / 2
            state["pan_x"] = cx / old_s * new_s - w_f / 2
            state["pan_y"] = cy / old_s * new_s - h_f / 2
            state["scale"] = new_s
            state["redraw"] = True

    cv2.destroyAllWindows()
    print("[VISUALIZER] Chiuso.")


# ---------------------------------------------------------------------------
# PIPELINE PRINCIPALE
# ---------------------------------------------------------------------------

def build_roi_dataset(
    nozzle_db_path: Path    = NOZZLE_DB_PATH,
    dataset_name: str       = DATASET_NAME,
    output_dir: Path        = OUTPUT_DIR,
    params_std: dict        = ROI_PARAMS_STD,
    params_g3:  dict        = ROI_PARAMS_G3,
    metrics: dict           = METRICS,
    oss_filter: str         = OSS_FILTER,
    enable_visualizer: bool = ENABLE_VISUALIZER,
    boundaries: dict        = PIECE_BOUNDARIES,
) -> pd.DataFrame:
    if not nozzle_db_path.exists():
        raise FileNotFoundError(
            f"nozzle_db non trovato: {nozzle_db_path}\n"
            "Esegui prima: python build_nozzle_db.py"
        )

    df_nozzle = pd.read_csv(nozzle_db_path)
    print(f"[INFO] nozzle_db: {len(df_nozzle)} righe")

    frame_items = build_frame_list(df_nozzle, oss_filter)
    print(f"[INFO] Frame ({oss_filter}): {len(frame_items)}")
    print(f"[INFO] ROI_PARAMS_STD : {params_std}")
    print(f"[INFO] ROI_PARAMS_G3  : {params_g3}")
    print(f"[INFO] T=[{T_MIN}, {T_MAX}] C")
    print(f"[INFO] Piece boundaries: {len(boundaries)} specimen definiti")

    if enable_visualizer:
        print("\n[INFO] Visualizer \u2014 Q per continuare...")
        run_visualizer(frame_items, boundaries, params_std, params_g3)
        print("[INFO] Inizio estrazione...\n")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / f"{dataset_name}.csv"
    records = []

    for row_dict, frames_dir, obs_idx in tqdm(frame_items, desc="ROI extraction"):
        frame = load_frame_gray(frames_dir, int(row_dict["frame_idx"]))
        if frame is None:
            tqdm.write(f"  [WARN] Mancante: {int(row_dict['frame_idx']):04d}.png")
            continue

        specimen   = row_dict["specimen"]
        tip_x      = int(row_dict["tip_x"])
        tip_y      = int(row_dict["tip_y"])

        # Selezione ROI params in base al gruppo
        roi_params = select_roi_params(specimen, params_std, params_g3)

        roi = extract_roi(frame, tip_x, tip_y, **roi_params)
        if roi is None:
            tqdm.write(f"  [WARN] ROI vuota frame {row_dict['frame_idx']}")
            continue

        roi_celsius = pixels_to_celsius(roi)
        c0, r0, c1, r1 = roi_rect(tip_x, tip_y,
            roi_params["x_start"], roi_params["x_end"],
            roi_params["y_start"], roi_params["y_end"],
            frame.shape)

        frac, complete = compute_roi_validity(c0, c1, specimen, boundaries)

        record = {
            "specimen":      specimen,
            "layer_index":   int(row_dict["layer_index"]),
            "obs_index":     obs_idx,
            "frame_idx":     int(row_dict["frame_idx"]),
            "frame_type":    row_dict["frame_type"],
            "tip_x":         tip_x,
            "tip_y":         tip_y,
            "direction_deg": row_dict.get("direction_deg", float("nan")),
            "movement_px":   row_dict.get("movement_px",   float("nan")),
            "confidence":    float(row_dict["confidence"]),
            "roi_c0": c0, "roi_r0": r0, "roi_c1": c1, "roi_r1": r1,
            "roi_valid_frac": frac,
            "roi_complete":   complete,
        }
        for col_name, func in metrics.items():
            try:
                record[col_name] = round(func(roi_celsius), 4) \
                    if col_name != "roi_pixels_raw" else func(roi_celsius)
            except Exception as e:
                record[col_name] = float("nan")
                tqdm.write(f"  [WARN] '{col_name}': {e}")

        records.append(record)

    df_out = pd.DataFrame(records)
    if df_out.empty:
        print("[WARN] Dataset vuoto.")
        return df_out

    n_complete = int(df_out["roi_complete"].sum()) if df_out["roi_complete"].notna().any() else 0
    print(f"\n[OK] {output_csv}")
    print(f"     {len(df_out)} righe totali | "
          f"{n_complete} roi_complete ({100*n_complete/len(df_out):.1f}%) | "
          f"{df_out['specimen'].nunique()} provini")
    df_out.to_csv(output_csv, index=False)
    print(df_out.head(6).to_string(index=False))
    return df_out


if __name__ == "__main__":
    build_roi_dataset()
