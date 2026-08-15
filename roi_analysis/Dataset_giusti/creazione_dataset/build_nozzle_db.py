"""
build_nozzle_db.py
------------------
Pipeline automatica: itera su TUTTI i provini validi con layers annotati
(da frame_selector.SPECIMENS), per ogni layer e per ogni frame (pre+core+post)
esegue il tracking dell'ugello e costruisce un database CSV unico.

Struttura del CSV output (nozzle_db.csv):

    specimen        : nome del provino (es. Rec-022_10s_1)
    layer_index     : indice del layer (1, 2, 3, ...)
    frame_type      : 'pre' | 'core' | 'post'
    frame_idx       : indice assoluto del frame nella cartella _frames (0001.png -> 1)
    tip_x           : coordinata X punta ugello (pixel)
    tip_y           : coordinata Y punta ugello (pixel, bordo inferiore kernel)
    confidence      : score template matching [0, 1]
    delta_x         : spostamento X rispetto al frame precedente NELLO STESSO LAYER
    delta_y         : spostamento Y rispetto al frame precedente NELLO STESSO LAYER
    direction_deg   : angolo di movimento [deg]  (0=destra, 90=su, -90=giu)
                      calcolato rispetto al frame precedente nello stesso layer.
                      Per il primo frame di ogni layer: NaN.
    movement_px     : modulo spostamento [pixel]. NaN per il primo frame di ogni layer.

Uso:
    python Dataset_giusti/creazione_dataset/build_nozzle_db.py
    oppure direttamente con percorso assoluto — il progetto root viene
    rilevato automaticamente da __file__.

Parametri configurabili nella sezione CONFIG.

Kernel:
    - kernel_2.png  : usato per tutti i provini che NON hanno 'G3' nel nome
    - kernel_3.png  : usato per i provini con 'G3' nel nome
    Entrambi devono stare in: data/feature_extraction/
"""

import sys
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

SCRIPT_DIR   = Path(__file__).resolve().parent          # .../Dataset_giusti/creazione_dataset
DATASET_DIR  = SCRIPT_DIR.parent                        # .../Dataset_giusti
PROJECT_ROOT = SCRIPT_DIR.parent.parent                 # .../Lab Data science  (contiene config.py)

# Aggiunge al sys.path tutte le directory necessarie (in ordine di priorità)
for _p in (str(SCRIPT_DIR), str(DATASET_DIR), str(PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from config import DATA_DIR
from Dataset_giusti.creazione_dataset.frame_selector import SPECIMENS, get_layer_frames, LayerNotDefinedError
from Dataset_giusti.creazione_dataset.nozzle_tracker import get_nozzle_tip, load_kernel


# ===========================================================================
# CONFIG
# ===========================================================================

# Kernel per provini standard (senza G3 nel nome)
KERNEL_PATH_STD = PROJECT_ROOT / "data" / "feature_extraction" / "kernel_2.png"

# Kernel per provini G3 (riprese in momento diverso, inquadratura diversa)
KERNEL_PATH_G3  = PROJECT_ROOT / "data" / "feature_extraction" / "kernel_3.png"

MARGIN = 5

# Output nella sottocartella datasets/ dentro la cartella di lavoro
OUTPUT_CSV = SCRIPT_DIR / "datasets" / "nozzle_db.csv"

SAVE_DEBUG_IMAGES    = False
DEBUG_MAX_PER_LAYER  = 5

# ===========================================================================


def select_kernel(specimen_name: str,
                  kernel_std: np.ndarray,
                  kernel_g3: np.ndarray) -> np.ndarray:
    """Restituisce kernel_g3 se il nome contiene 'G3', altrimenti kernel_std."""
    if "G3" in specimen_name:
        return kernel_g3
    return kernel_std


def process_layer(
    specimen_name: str,
    layer_index: int,
    kernel: np.ndarray,
    margin: int = MARGIN,
    debug_dir: Path | None = None,
    debug_max: int = DEBUG_MAX_PER_LAYER,
) -> list[dict]:
    layer_info = get_layer_frames(specimen_name, layer_index, margin=margin)
    frames_dir = layer_info["frames_dir"]

    if not frames_dir.exists():
        print(f"  [SKIP] Cartella non trovata: {frames_dir}")
        return []

    sequence = (
        [(idx, "pre")  for idx in layer_info["frames_pre"]]  +
        [(idx, "core") for idx in layer_info["frames_core"]] +
        [(idx, "post") for idx in layer_info["frames_post"]]
    )

    records  = []
    prev_tip = None
    debug_count = 0

    for frame_idx, frame_type in sequence:
        frame_path = frames_dir / f"{frame_idx:04d}.png"
        if not frame_path.exists():
            print(f"  [WARN] Frame mancante: {frame_path.name}")
            continue

        frame = cv2.imread(str(frame_path), cv2.IMREAD_GRAYSCALE)
        if frame is None:
            print(f"  [WARN] Frame non leggibile: {frame_path.name}")
            continue

        tip_x, tip_y, conf = get_nozzle_tip(frame, kernel)

        if prev_tip is not None:
            dx = tip_x - prev_tip[0]
            dy = tip_y - prev_tip[1]
            direction_deg = float(np.degrees(np.arctan2(-dy, dx)))
            movement_px   = float(np.hypot(dx, dy))
        else:
            dx, dy        = np.nan, np.nan
            direction_deg = np.nan
            movement_px   = np.nan

        records.append({
            "specimen":      specimen_name,
            "layer_index":   layer_index,
            "frame_type":    frame_type,
            "frame_idx":     frame_idx,
            "tip_x":         tip_x,
            "tip_y":         tip_y,
            "confidence":    round(conf, 4),
            "delta_x":       dx,
            "delta_y":       dy,
            "direction_deg": round(direction_deg, 2) if not np.isnan(direction_deg) else np.nan,
            "movement_px":   round(movement_px, 2)   if not np.isnan(movement_px)   else np.nan,
        })
        prev_tip = (tip_x, tip_y)

        if debug_dir is not None and debug_count < debug_max:
            vis = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            kh, _ = kernel.shape
            cv2.circle(vis, (tip_x, tip_y), 3, (0, 0, 255), -1)
            cv2.circle(vis, (tip_x, tip_y - kh // 2), 3, (0, 255, 0), 1)
            if not np.isnan(direction_deg):
                rad = np.radians(direction_deg)
                cv2.arrowedLine(vis, (tip_x, tip_y),
                                (int(tip_x + 25*np.cos(rad)), int(tip_y - 25*np.sin(rad))),
                                (255, 255, 0), 1, tipLength=0.35)
            cv2.putText(vis, f"{frame_type} f{frame_idx} conf:{conf:.2f}",
                        (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            cv2.imwrite(str(debug_dir / f"{specimen_name}_L{layer_index}_{frame_type}_{frame_idx:04d}.png"), vis)
            debug_count += 1

    return records


def build_nozzle_db(
    kernel_path_std: Path = KERNEL_PATH_STD,
    kernel_path_g3:  Path = KERNEL_PATH_G3,
    margin: int           = MARGIN,
    output_csv: Path      = OUTPUT_CSV,
    save_debug: bool      = SAVE_DEBUG_IMAGES,
) -> pd.DataFrame:
    kernel_std = load_kernel(kernel_path_std)
    kernel_g3  = load_kernel(kernel_path_g3)
    print(f"[INFO] Kernel STD caricato: {kernel_path_std}  shape={kernel_std.shape}")
    print(f"[INFO] Kernel G3  caricato: {kernel_path_g3}   shape={kernel_g3.shape}")
    print(f"[INFO] Margine frame      : {margin}")

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    debug_dir = None
    if save_debug:
        debug_dir = output_csv.parent / "debug_nozzle"
        debug_dir.mkdir(exist_ok=True)

    all_records = []
    valid_specimens = {
        name: spec for name, spec in SPECIMENS.items()
        if spec["valid"] and spec["layers_front"] is not None
    }

    print(f"[INFO] Provini da processare: {len(valid_specimens)}")
    for name, spec in valid_specimens.items():
        k_label   = "G3" if "G3" in name else "STD"
        n_layers  = len(spec["layers_front"])
        print(f"  - {name}  [kernel_{k_label}]  ({n_layers} layer)")
    print()

    for specimen_name, spec in tqdm(valid_specimens.items(), desc="Provini", unit="spec"):
        kernel   = select_kernel(specimen_name, kernel_std, kernel_g3)
        n_layers = len(spec["layers_front"])   # numero reale di layer del provino
        for layer_idx in range(1, n_layers + 1):
            tqdm.write(f"  [{specimen_name}] Layer {layer_idx}/{n_layers}...")
            try:
                records = process_layer(specimen_name, layer_idx, kernel, margin, debug_dir)
                all_records.extend(records)
                tqdm.write(f"    -> {len(records)} frame")
            except LayerNotDefinedError as e:
                tqdm.write(f"    [SKIP] {e}")
            except Exception as e:
                tqdm.write(f"    [ERROR] {e}")

    df = pd.DataFrame(all_records)
    if df.empty:
        print("[WARN] Nessun record. Controlla i path dei frame.")
        return df

    for col in ["layer_index", "frame_idx", "tip_x", "tip_y"]:
        df[col] = df[col].astype(int)

    df.to_csv(output_csv, index=False)
    print(f"\n[OK] Salvato: {output_csv}")
    print(f"[OK] Righe: {len(df)}  |  Provini: {df['specimen'].nunique()}")
    print(df.head(6).to_string(index=False))
    return df


if __name__ == "__main__":
    build_nozzle_db()
