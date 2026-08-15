"""
nozzle_tracker.py
-----------------
Core del tracking dell'ugello su un singolo frame.

Funzioni esportate:
    get_nozzle_tip(frame_gray, kernel_gray) -> (tip_x, tip_y, confidence)

Nota sulla "punta dell'ugello":
    Non e' il centro del kernel, ma il punto piu' in basso al centro:
        tip_x = match_loc_x + kernel_w // 2
        tip_y = match_loc_y + kernel_h          <- bordo inferiore del kernel
"""

import cv2
import numpy as np
from pathlib import Path


def get_nozzle_tip(
    frame_gray: np.ndarray,
    kernel_gray: np.ndarray,
) -> tuple[int, int, float]:
    """
    Trova la punta dell'ugello (punto piu' in basso al centro del kernel)
    tramite template matching normalizzato.

    Args:
        frame_gray  : frame in scala di grigi (numpy array)
        kernel_gray : kernel in scala di grigi (immagine di riferimento ugello)

    Returns:
        (tip_x, tip_y, confidence)
            tip_x      : coordinata X della punta ugello (centro orizzontale)
            tip_y      : coordinata Y della punta ugello (bordo inferiore kernel)
            confidence : score template matching [0, 1]
    """
    res = cv2.matchTemplate(frame_gray, kernel_gray, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)

    kh, kw = kernel_gray.shape
    tip_x = max_loc[0] + kw // 2   # centro orizzontale del kernel
    tip_y = max_loc[1] + kh        # bordo inferiore del kernel = punta ugello

    return int(tip_x), int(tip_y), float(max_val)


def load_kernel(kernel_path: str | Path) -> np.ndarray:
    """
    Carica il kernel da disco. Lancia ValueError se non trovato.
    """
    kernel_path = Path(kernel_path)
    if not kernel_path.exists():
        raise FileNotFoundError(f"Kernel non trovato: {kernel_path}")
    kernel = cv2.imread(str(kernel_path), cv2.IMREAD_GRAYSCALE)
    if kernel is None:
        raise ValueError(f"Impossibile leggere il kernel: {kernel_path}")
    return kernel
