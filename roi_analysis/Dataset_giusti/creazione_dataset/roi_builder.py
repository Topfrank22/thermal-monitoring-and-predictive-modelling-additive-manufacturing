"""
roi_builder.py
--------------
Definisce la ROI (Region of Interest) dinamica per ogni frame,
based on the nozzle position and movement direction computed by nozzle_tracker.

Logica ROI:
    Il punto di partenza e' la posizione dell'ugello (nozzle_x, nozzle_y).
    La ROI e' un rettangolo posizionato DAVANTI all'ugello nella direzione
    di movimento, con un margine laterale simmetrico.

    +-------------------------------+
    |          offset_ahead         |  <- distanza ugello -> bordo anteriore ROI
    |    [UGELLO]---direzione-->    |
    |          offset_behind        |  <- distanza ugello -> bordo posteriore
    +-------------------------------+
         <--- roi_width --->

Output:
    CSV con colonne:
        frame_idx, frame_file, nozzle_x, nozzle_y, direction_deg,
        roi_x, roi_y, roi_w, roi_h
    dove (roi_x, roi_y) e' il vertice top-left della ROI.

Uso standalone:
    python roi_builder.py --frames data/Rec-G3_S60_frames \\
                           --kernel data/feature_extraction/kernel.png \\
                           --frames_idx 100 200 300 \\
                           --output output/roi.csv \\
                           --visualize

Oppure importato come modulo:
    from roi_builder import build_roi_dataframe
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import argparse

# Import del tracker locale
from Dataset_giusti.creazione_dataset.nozzle_tracker import track_nozzle


# ---------------------------------------------------------------------------
# Parametri ROI di default (modificabili via CLI o come kwargs)
# ---------------------------------------------------------------------------

DEFAULT_ROI_PARAMS = {
    "offset_ahead":  80,   # pixel davanti all'ugello (nella direzione di movimento)
    "offset_behind": 20,   # pixel dietro l'ugello
    "half_width":    60,   # semi-larghezza laterale della ROI
}


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------

def compute_roi(
    nozzle_x: int,
    nozzle_y: int,
    direction_deg: float,
    frame_shape: tuple[int, int],
    offset_ahead: int  = DEFAULT_ROI_PARAMS["offset_ahead"],
    offset_behind: int = DEFAULT_ROI_PARAMS["offset_behind"],
    half_width: int    = DEFAULT_ROI_PARAMS["half_width"],
) -> tuple[int, int, int, int]:
    """
    Calcola la ROI rettangolare attorno all'ugello orientata nella direzione di movimento.

    La ROI e' allineata agli assi (AABB) per semplicita' d'uso con OpenCV;
    l'orientamento della direzione viene usato per decidere quale lato
    e' "davanti" e quale e' "dietro".

    Args:
        nozzle_x, nozzle_y : centro ugello in pixel
        direction_deg      : angolo di movimento (0=destra, 90=su, -90=giu)
        frame_shape        : (height, width) del frame per clipping
        offset_ahead       : pixel davanti all'ugello
        offset_behind      : pixel dietro l'ugello
        half_width         : semi-larghezza perpendicolare alla direzione

    Returns:
        (x, y, w, h) – vertice top-left e dimensioni della ROI (già clippata)
    """
    rad = np.radians(direction_deg)
    # Versore direzione (asse X positivo = destra, Y positivo = giu in coord pixel)
    dir_x =  np.cos(rad)
    dir_y = -np.sin(rad)  # inversione Y per coordinate immagine

    # Versore perpendicolare
    perp_x = -dir_y
    perp_y =  dir_x

    # I 4 vertici del rettangolo orientato
    corners = np.array([
        [nozzle_x + dir_x * offset_ahead  + perp_x * half_width,
         nozzle_y + dir_y * offset_ahead  + perp_y * half_width],
        [nozzle_x + dir_x * offset_ahead  - perp_x * half_width,
         nozzle_y + dir_y * offset_ahead  - perp_y * half_width],
        [nozzle_x - dir_x * offset_behind + perp_x * half_width,
         nozzle_y - dir_y * offset_behind + perp_y * half_width],
        [nozzle_x - dir_x * offset_behind - perp_x * half_width,
         nozzle_y - dir_y * offset_behind - perp_y * half_width],
    ])

    # AABB (Axis-Aligned Bounding Box) dei vertici
    x_min = int(np.floor(corners[:, 0].min()))
    y_min = int(np.floor(corners[:, 1].min()))
    x_max = int(np.ceil(corners[:, 0].max()))
    y_max = int(np.ceil(corners[:, 1].max()))

    # Clipping ai bordi del frame
    h_frame, w_frame = frame_shape
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(w_frame, x_max)
    y_max = min(h_frame, y_max)

    roi_x = x_min
    roi_y = y_min
    roi_w = x_max - x_min
    roi_h = y_max - y_min

    return roi_x, roi_y, roi_w, roi_h


def build_roi_dataframe(
    frames_dir: str | Path,
    kernel_path: str | Path,
    frame_indices: list[int] | None = None,
    offset_ahead: int  = DEFAULT_ROI_PARAMS["offset_ahead"],
    offset_behind: int = DEFAULT_ROI_PARAMS["offset_behind"],
    half_width: int    = DEFAULT_ROI_PARAMS["half_width"],
) -> pd.DataFrame:
    """
    Pipeline completa: tracking ugello -> calcolo ROI per ogni frame.

    Returns:
        DataFrame con colonne di tracking + roi_x, roi_y, roi_w, roi_h
    """
    frames_dir  = Path(frames_dir)
    kernel_path = Path(kernel_path)

    # Step 1: tracking
    print("[INFO] Avvio tracking ugello...")
    df_track = track_nozzle(frames_dir, kernel_path, frame_indices)

    if df_track.empty:
        print("[WARN] Nessun frame tracciato. Output vuoto.")
        return df_track

    # Carica un frame campione per ottenere le dimensioni
    sample_path = frames_dir / df_track.iloc[0]["frame_file"]
    sample = cv2.imread(str(sample_path), cv2.IMREAD_GRAYSCALE)
    frame_shape = sample.shape  # (height, width)

    # Step 2: calcolo ROI
    print("[INFO] Calcolo ROI...")
    roi_records = []
    for _, row in df_track.iterrows():
        roi_x, roi_y, roi_w, roi_h = compute_roi(
            nozzle_x=int(row["nozzle_x"]),
            nozzle_y=int(row["nozzle_y"]),
            direction_deg=float(row["direction_deg"]),
            frame_shape=frame_shape,
            offset_ahead=offset_ahead,
            offset_behind=offset_behind,
            half_width=half_width,
        )
        roi_records.append({
            "roi_x": roi_x,
            "roi_y": roi_y,
            "roi_w": roi_w,
            "roi_h": roi_h,
        })

    df_roi = pd.concat([df_track, pd.DataFrame(roi_records)], axis=1)
    return df_roi


# ---------------------------------------------------------------------------
# Visualizzazione opzionale
# ---------------------------------------------------------------------------

def visualize_roi(
    frames_dir: str | Path,
    df_roi: pd.DataFrame,
    output_dir: str | Path,
    max_frames: int = 20,
) -> None:
    """
    Salva immagini con ugello (cerchio verde) e ROI (rettangolo rosso) disegnati.
    Utile per verifica visiva.
    """
    frames_dir = Path(frames_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, row in df_roi.head(max_frames).iterrows():
        frame_path = frames_dir / row["frame_file"]
        frame = cv2.imread(str(frame_path), cv2.IMREAD_GRAYSCALE)
        if frame is None:
            continue
        vis = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        nx, ny = int(row["nozzle_x"]), int(row["nozzle_y"])
        rx, ry, rw, rh = int(row["roi_x"]), int(row["roi_y"]), int(row["roi_w"]), int(row["roi_h"])

        # Ugello: punto verde
        cv2.circle(vis, (nx, ny), 4, (0, 255, 0), -1)

        # ROI: rettangolo rosso
        cv2.rectangle(vis, (rx, ry), (rx + rw, ry + rh), (0, 0, 255), 1)

        # Freccia direzione
        rad = np.radians(row["direction_deg"])
        arrow_len = 30
        ax = int(nx + arrow_len * np.cos(rad))
        ay = int(ny - arrow_len * np.sin(rad))
        cv2.arrowedLine(vis, (nx, ny), (ax, ay), (255, 255, 0), 1, tipLength=0.3)

        # Label confidenza
        label = f"conf:{row['confidence']:.2f}  dir:{row['direction_deg']:.1f}deg"
        cv2.putText(vis, label, (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (200, 200, 200), 1, cv2.LINE_AA)

        out_file = output_dir / f"roi_{row['frame_file']}"
        cv2.imwrite(str(out_file), vis)

    print(f"[OK] Visualizzazioni salvate in: {output_dir}")


# ---------------------------------------------------------------------------
# CLI standalone
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description="Calcola ROI dinamica basata sul tracking dell'ugello.")
    p.add_argument("--frames",        required=True,  help="Cartella frame .png")
    p.add_argument("--kernel",        required=True,  help="Path al kernel .png")
    p.add_argument("--frames_idx",    nargs="*", type=int, default=None,
                   help="Indici frame (0-based). Default: tutti.")
    p.add_argument("--output",        default="output/roi.csv",
                   help="CSV di output (default: output/roi.csv)")
    p.add_argument("--offset_ahead",  type=int, default=DEFAULT_ROI_PARAMS["offset_ahead"],
                   help=f"Pixel davanti ugello (default: {DEFAULT_ROI_PARAMS['offset_ahead']})")
    p.add_argument("--offset_behind", type=int, default=DEFAULT_ROI_PARAMS["offset_behind"],
                   help=f"Pixel dietro ugello (default: {DEFAULT_ROI_PARAMS['offset_behind']})")
    p.add_argument("--half_width",    type=int, default=DEFAULT_ROI_PARAMS["half_width"],
                   help=f"Semi-larghezza ROI (default: {DEFAULT_ROI_PARAMS['half_width']})")
    p.add_argument("--visualize",     action="store_true",
                   help="Salva immagini di debug con ugello e ROI disegnati")
    p.add_argument("--vis_output",    default="output/roi_debug",
                   help="Cartella immagini debug (default: output/roi_debug)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    print(f"[INFO] Frame dir    : {args.frames}")
    print(f"[INFO] Kernel       : {args.kernel}")
    print(f"[INFO] Indici       : {args.frames_idx if args.frames_idx else 'tutti'}")
    print(f"[INFO] offset_ahead : {args.offset_ahead}")
    print(f"[INFO] offset_behind: {args.offset_behind}")
    print(f"[INFO] half_width   : {args.half_width}")

    df = build_roi_dataframe(
        frames_dir=args.frames,
        kernel_path=args.kernel,
        frame_indices=args.frames_idx,
        offset_ahead=args.offset_ahead,
        offset_behind=args.offset_behind,
        half_width=args.half_width,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[OK] ROI salvata in: {out_path}")
    print(df[["frame_idx", "frame_file", "nozzle_x", "nozzle_y",
              "direction_deg", "movement_px", "roi_x", "roi_y", "roi_w", "roi_h"]].to_string(index=False))

    if args.visualize:
        visualize_roi(
            frames_dir=args.frames,
            df_roi=df,
            output_dir=args.vis_output,
        )
