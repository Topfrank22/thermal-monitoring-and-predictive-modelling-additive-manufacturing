"""
roi_dataset_runner.py
---------------------
Entry point unificato per la costruzione dei due dataset ROI:
    1. Dataset STD   -> usa nozzle_db_std.csv  -> <DATASET_NAME>_STD.csv
    2. Dataset REG   -> usa nozzle_db.csv       -> <DATASET_NAME>.csv

Entrambi usano gli stessi parametri ROI configurati qui sotto.
Se ENABLE_VISUALIZER = True, il visualizer OpenCV viene lanciato due volte
in sequenza (prima STD, poi REG) con un banner HUD che indica quale
dataset si sta ispezionando. Premi Q per passare al successivo.

===========================================================================
CONFIG  -  tutto quello che vuoi cambiare e' qui
===========================================================================
"""

import sys
from pathlib import Path

SCRIPT_DIR     = Path(__file__).resolve().parent
REPO_ROOT      = SCRIPT_DIR.parent
WORKSPACE_ROOT = REPO_ROOT.parent

sys.path[:0] = [str(WORKSPACE_ROOT), str(REPO_ROOT), str(SCRIPT_DIR)]

import cv2
import numpy as np
import pandas as pd

from Dataset_giusti.creazione_dataset.roi_dataset_builder import (
    build_roi_dataset,
    build_frame_list,
    load_frame_gray,
    roi_rect,
    select_roi_params,
    compute_roi_validity,
    PIECE_BOUNDARIES,
    VISUALIZER_MAX_FRAMES,
    VISUALIZER_SCALE,
    ZOOM_STEP,
    ZOOM_MIN,
    ZOOM_MAX,
)

# ===========================================================================
# CONFIG
# ===========================================================================

# --- ROI offsets rispetto alla punta dell'ugello ---
# x positivo = destra  |  y positivo = basso
X_START = 2
X_END   = 12
Y_START = 2
Y_END   = 5

OFFSET_NON_G3 = 1   # provini senza 'G3' nel nome
OFFSET_G3     = 0   # provini con    'G3' nel nome

# --- Filtro osservazioni ---
OSS_FILTER = "core"   # "core" oppure "all"

# --- Nome base per i CSV di output ---
# Genera: <DATASET_NAME>_STD.csv  e  <DATASET_NAME>.csv
DATASET_NAME = "ROI_wide_2_12_depth_2_5"

# --- Visualizer interattivo OpenCV ---
# Se True: prima STD, poi REG.  Premi Q per passare al successivo.
ENABLE_VISUALIZER = False

# --- Metriche ---
METRICS = {
    "roi_mean_C":     lambda px: float(np.mean(px)),
    "roi_max_C":      lambda px: float(np.max(px)),
    "roi_min_C":      lambda px: float(np.min(px)),
    "roi_std_C":      lambda px: float(np.std(px)),
    "roi_median_C":   lambda px: float(np.median(px)),
    "roi_pixels_raw": lambda px: ",".join(f"{v:.2f}" for v in px.flatten()),
}

# ===========================================================================
# NON TOCCARE OLTRE QUESTA LINEA
# ===========================================================================

OUTPUT_DIR    = SCRIPT_DIR / "datasets"
NOZZLE_DB_STD = SCRIPT_DIR / "datasets" / "nozzle_db_std.csv"
NOZZLE_DB_REG = SCRIPT_DIR / "datasets" / "nozzle_db.csv"

# Costruisce i due dizionari ROI dagli offset
ROI_PARAMS_NON_G3 = {
    "x_start": X_START,
    "x_end":   X_END,
    "y_start": Y_START + OFFSET_NON_G3,
    "y_end":   Y_END   + OFFSET_NON_G3,
}

ROI_PARAMS_G3 = {
    "x_start": X_START,
    "x_end":   X_END,
    "y_start": Y_START + OFFSET_G3,
    "y_end":   Y_END   + OFFSET_G3,
}


# ---------------------------------------------------------------------------
# Visualizer con banner — replica run_visualizer() aggiungendo il label
# ---------------------------------------------------------------------------

def run_visualizer_labeled(
    frame_items: list,
    label: str,
    boundaries: dict = PIECE_BOUNDARIES,
    params_std: dict = ROI_PARAMS_NON_G3,
    params_g3:  dict = ROI_PARAMS_G3,
    max_frames: int | None = VISUALIZER_MAX_FRAMES,
    scale: float = VISUALIZER_SCALE,
) -> None:
    """
    Visualizer interattivo identico a run_visualizer() di roi_dataset_builder,
    ma con un banner [DATASET: <label>] fisso nell'angolo in alto a destra
    sovrapposto al frame.

    Navigazione:
        A / D  o  frecce sx/dx : frame precedente / successivo
        W / S  o  frecce su/giu: -10 / +10 frame
        Rotella mouse           : zoom centrato sul cursore
        + / =  e  -            : zoom da tastiera
        Q / ESC                : chiudi e procedi
    """
    items = frame_items if max_frames is None else frame_items[:max_frames]
    if not items:
        print(f"[VISUALIZER:{label}] Nessun frame.")
        return

    state = {
        "scale":  max(scale, ZOOM_MIN),
        "pan_x":  0.0,
        "pan_y":  0.0,
        "redraw": True,
    }

    WIN = "ROI Visualizer  |  A/D: naviga   W/S: +-10   rotella: zoom   Q: esci"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)

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

    cv2.setMouseCallback(WIN, on_mouse)

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

                roi_params = select_roi_params(specimen, params_std, params_g3)

                c0, r0, c1, r1 = roi_rect(
                    tip_x, tip_y,
                    roi_params["x_start"], roi_params["x_end"],
                    roi_params["y_start"], roi_params["y_end"],
                    frame.shape,
                )
                base_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

                # Bordi pezzo
                if specimen in boundaries:
                    xl = boundaries[specimen]["x_left"]
                    xr = boundaries[specimen]["x_right"]
                    h_vis = base_frame.shape[0]
                    cv2.line(base_frame, (xl, 0), (xl, h_vis - 1), (0, 255, 0), 1)
                    cv2.line(base_frame, (xr, 0), (xr, h_vis - 1), (0, 100, 255), 1)

                frac, complete = compute_roi_validity(c0, c1, specimen, boundaries)
                if complete is True:
                    rect_color = (0, 220, 0)
                elif complete is False and frac > 0:
                    rect_color = (0, 200, 255)
                else:
                    rect_color = (0, 0, 255)

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

                # --- Banner DATASET label (angolo in alto a destra) ---
                banner    = f"[ DATASET: {label} ]"
                font      = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.55
                thickness  = 1
                (bw, bh), _ = cv2.getTextSize(banner, font, font_scale, thickness)
                bx = base_frame.shape[1] - bw - 8
                by = bh + 6
                # sfondo banner
                cv2.rectangle(base_frame,
                               (bx - 4, by - bh - 4),
                               (bx + bw + 4, by + 4),
                               (30, 30, 30), -1)
                # testo banner
                cv2.putText(base_frame, banner, (bx, by), font,
                            font_scale, (0, 220, 255), thickness, cv2.LINE_AA)

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
            cv2.imshow(WIN, view)
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
    print(f"[VISUALIZER:{label}] Chiuso.")


# ---------------------------------------------------------------------------
# Helper: carica nozzle_db e lancia il visualizer labeled
# ---------------------------------------------------------------------------

def _visualize(nozzle_db_path: Path, label: str) -> None:
    if not nozzle_db_path.exists():
        print(f"[WARN] {nozzle_db_path} non trovato, skip visualizer [{label}].")
        return
    df_nozzle   = pd.read_csv(nozzle_db_path)
    frame_items = build_frame_list(df_nozzle, OSS_FILTER)
    if not frame_items:
        print(f"[WARN] Nessun frame trovato per [{label}], skip visualizer.")
        return
    print(f"\n[VISUALIZER] Dataset: {label}  ({len(frame_items)} frame)")
    print("[VISUALIZER] Premi Q per chiudere e passare al passo successivo...\n")
    run_visualizer_labeled(
        frame_items=frame_items,
        label=label,
        boundaries=PIECE_BOUNDARIES,
        params_std=ROI_PARAMS_NON_G3,
        params_g3=ROI_PARAMS_G3,
    )


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  ROI DATASET RUNNER")
    print(f"  ROI_PARAMS_NON_G3 : {ROI_PARAMS_NON_G3}")
    print(f"  ROI_PARAMS_G3     : {ROI_PARAMS_G3}")
    print(f"  OSS_FILTER        : {OSS_FILTER}")
    print(f"  VISUALIZER        : {ENABLE_VISUALIZER}")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Dataset STD
    # ------------------------------------------------------------------
    print("\n>>> [1/2] Dataset STD")

    if ENABLE_VISUALIZER:
        _visualize(NOZZLE_DB_STD, label="STD")

    df_std = build_roi_dataset(
        nozzle_db_path=NOZZLE_DB_STD,
        dataset_name=DATASET_NAME + "_STD",
        output_dir=OUTPUT_DIR,
        params_std=ROI_PARAMS_NON_G3,
        params_g3=ROI_PARAMS_G3,
        metrics=METRICS,
        oss_filter=OSS_FILTER,
        enable_visualizer=False,
        boundaries=PIECE_BOUNDARIES,
    )

    # ------------------------------------------------------------------
    # 2. Dataset REG (normale)
    # ------------------------------------------------------------------
    print("\n>>> [2/2] Dataset REG (normale)")

    if ENABLE_VISUALIZER:
        _visualize(NOZZLE_DB_REG, label="REG")

    df_reg = build_roi_dataset(
        nozzle_db_path=NOZZLE_DB_REG,
        dataset_name=DATASET_NAME,
        output_dir=OUTPUT_DIR,
        params_std=ROI_PARAMS_NON_G3,
        params_g3=ROI_PARAMS_G3,
        metrics=METRICS,
        oss_filter=OSS_FILTER,
        enable_visualizer=False,
        boundaries=PIECE_BOUNDARIES,
    )

    # ------------------------------------------------------------------
    # Riepilogo finale
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  RIEPILOGO FINALE")
    print("=" * 60)
    std_rows = len(df_std) if df_std is not None and not df_std.empty else 0
    reg_rows = len(df_reg) if df_reg is not None and not df_reg.empty else 0
    print(f"  STD  -> {OUTPUT_DIR / (DATASET_NAME + '_STD.csv')}  ({std_rows} righe)")
    print(f"  REG  -> {OUTPUT_DIR / (DATASET_NAME + '.csv')}  ({reg_rows} righe)")
    print("=" * 60)


if __name__ == "__main__":
    main()
