"""
piece_boundaries.py
--------------------
Tool interattivo per definire i bordi del pezzo per ogni specimen.

Funzionamento:
    Per ogni specimen carica l'ultimo frame disponibile (il piu' avanzato,
    quindi quello con il pezzo piu' esteso) e ti chiede di tracciare:
        1. Bordo SINISTRO del pezzo  (linea verticale, premi S)
        2. Bordo DESTRO  del pezzo  (linea verticale, premi D)

    Le coordinate x definiscono le colonne limite INCLUSE nella ROI valida:
        pixel validi = colonne con x_assoluta in [x_left, x_right]

    Il risultato viene salvato in:
        creazione_dataset_layer_frontale/datasets/piece_boundaries.json

    Formato JSON:
        {
            "S01": {"x_left": 120, "x_right": 430},
            "S02": {"x_left": 115, "x_right": 425},
            ...
        }

Uso:
    python piece_boundaries.py

Controlli finestra:
    Muovi il mouse     : la linea guida segue il cursore
    S                  : segna il bordo SINISTRO nella posizione corrente
    D                  : segna il bordo DESTRO nella posizione corrente
    Rotella mouse      : zoom in/out centrato sul cursore
    Invio / Spazio     : conferma e passa allo specimen successivo
    R                  : reset bordi per lo specimen corrente
    Q / ESC            : salva ed esci (anche se non tutti gli specimen sono pronti)

Nota: se piece_boundaries.json esiste gia', gli specimen gia' definiti
vengono saltati (mostrati in verde). Puoi rieseguire per aggiungere/correggere.
"""

import sys
import json
import cv2
import numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT  = SCRIPT_DIR.parent

sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(REPO_ROOT))

from config import DATA_DIR
from Dataset_giusti.creazione_dataset.frame_selector import get_layer_frames

OUTPUT_PATH = SCRIPT_DIR / "datasets" / "piece_boundaries.json"

ZOOM_STEP = 0.15
ZOOM_MIN  = 1.0
ZOOM_MAX  = 20.0


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def find_specimens(data_dir: Path) -> list[str]:
    """Trova tutti gli specimen disponibili nella cartella dati."""
    specimens = sorted([
        d.name for d in data_dir.iterdir()
        if d.is_dir() and not d.name.startswith("_")
    ])
    return specimens


def get_last_frame(specimen: str) -> tuple[np.ndarray | None, Path | None, int | None]:
    """
    Trova l'ultimo frame disponibile per uno specimen.
    Restituisce (frame_gray, frames_dir, frame_idx) oppure (None, None, None).
    """
    spec_dir = Path(DATA_DIR) / specimen
    if not spec_dir.exists():
        return None, None, None

    # Cerca i PNG direttamente nella cartella dello specimen
    pngs = sorted(spec_dir.glob("*.png"))
    if not pngs:
        return None, None, None

    # Prendi l'ultimo frame
    last_png  = pngs[-1]
    frame_idx = int(last_png.stem)
    frame     = cv2.imread(str(last_png), cv2.IMREAD_GRAYSCALE)
    if frame is not None:
        return frame, spec_dir, frame_idx

    return None, None, None


def load_boundaries() -> dict:
    if OUTPUT_PATH.exists():
        with open(OUTPUT_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_boundaries(boundaries: dict) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(boundaries, f, indent=2)
    print(f"[OK] Salvato: {OUTPUT_PATH}")


# ---------------------------------------------------------------------------
# Annotator interattivo per singolo specimen
# ---------------------------------------------------------------------------

def annotate_specimen(
    specimen: str,
    frame_gray: np.ndarray,
    existing: dict | None = None,
) -> dict | None:
    """
    Mostra il frame e permette di tracciare i bordi sinistro e destro.
    Restituisce {"x_left": int, "x_right": int} oppure None se ESC/Q.
    """
    h_f, w_f = frame_gray.shape
    base = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)

    state = {
        "scale":   1.0,
        "pan_x":   0.0,
        "pan_y":   0.0,
        "mouse_x": 0,
        "mouse_y": 0,
        "x_left":  existing["x_left"]  if existing else None,
        "x_right": existing["x_right"] if existing else None,
        "redraw":  True,
    }

    win = f"[{specimen}]  S=bordo sinistro  D=bordo destro  Invio=conferma  R=reset  Q=esci"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    def on_mouse(event, x, y, flags, param):
        state["mouse_x"] = x
        state["mouse_y"] = y
        state["redraw"]  = True
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

    cv2.setMouseCallback(win, on_mouse)

    result = None

    while True:
        if state["redraw"]:
            sc  = state["scale"]
            vis = base.copy()

            # Disegna bordi gia' fissati (in coordinate originali)
            if state["x_left"] is not None:
                cv2.line(vis, (state["x_left"], 0), (state["x_left"], h_f - 1),
                         (0, 255, 0), 1)
                cv2.putText(vis, f"L={state['x_left']}",
                            (state["x_left"] + 2, 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
            if state["x_right"] is not None:
                cv2.line(vis, (state["x_right"], 0), (state["x_right"], h_f - 1),
                         (0, 100, 255), 1)
                cv2.putText(vis, f"R={state['x_right']}",
                            (state["x_right"] + 2, 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 100, 255), 1)

            # Linea guida cursore (in coordinate originali)
            # Converti coordinate finestra -> coordinate originali
            orig_cursor_x = int((state["mouse_x"] + state["pan_x"]) / sc)
            orig_cursor_x = int(np.clip(orig_cursor_x, 0, w_f - 1))
            cv2.line(vis, (orig_cursor_x, 0), (orig_cursor_x, h_f - 1),
                     (200, 200, 0), 1, cv2.LINE_AA)
            cv2.putText(vis, f"x={orig_cursor_x}",
                        (orig_cursor_x + 2, h_f - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 0), 1)

            # HUD istruzioni
            hud_lines = [
                f"{specimen}  |  S: sinistro  D: destro  Invio: conferma  R: reset",
                f"L={state['x_left']}  R={state['x_right']}",
            ]
            for i, line in enumerate(hud_lines):
                yp = 14 + i * 15
                cv2.putText(vis, line, (4, yp), cv2.FONT_HERSHEY_SIMPLEX,
                            0.38, (10, 10, 10), 3, cv2.LINE_AA)
                cv2.putText(vis, line, (4, yp), cv2.FONT_HERSHEY_SIMPLEX,
                            0.38, (230, 230, 230), 1, cv2.LINE_AA)

            # Zoom + pan
            scaled = cv2.resize(vis, None, fx=sc, fy=sc,
                                interpolation=cv2.INTER_NEAREST)
            hs, ws = scaled.shape[:2]
            px = int(np.clip(state["pan_x"], 0, max(0, ws - w_f)))
            py = int(np.clip(state["pan_y"], 0, max(0, hs - h_f)))
            state["pan_x"] = float(px)
            state["pan_y"] = float(py)
            view_w = min(w_f, ws - px)
            view_h = min(h_f, hs - py)
            view   = scaled[py:py+view_h, px:px+view_w]
            cv2.imshow(win, view)
            state["redraw"] = False

        key = cv2.waitKeyEx(30)
        if key == -1:
            continue

        sc = state["scale"]
        orig_cursor_x = int(np.clip(
            (state["mouse_x"] + state["pan_x"]) / sc, 0, w_f - 1
        ))

        if key in (ord('q'), ord('Q'), 27):
            result = None
            break
        elif key in (ord('s'), ord('S')):
            state["x_left"]  = orig_cursor_x
            state["redraw"]  = True
            print(f"  [{specimen}] bordo sinistro = x={orig_cursor_x}")
        elif key in (ord('d'), ord('D')):
            state["x_right"] = orig_cursor_x
            state["redraw"]  = True
            print(f"  [{specimen}] bordo destro   = x={orig_cursor_x}")
        elif key in (ord('r'), ord('R')):
            state["x_left"]  = None
            state["x_right"] = None
            state["redraw"]  = True
            print(f"  [{specimen}] reset")
        elif key in (13, 32):  # Invio o Spazio
            if state["x_left"] is None or state["x_right"] is None:
                print(f"  [{specimen}] Definisci entrambi i bordi prima di confermare.")
                continue
            if state["x_left"] >= state["x_right"]:
                print(f"  [{specimen}] Il bordo sinistro deve essere < destro.")
                continue
            result = {"x_left": state["x_left"], "x_right": state["x_right"]}
            break

    cv2.destroyWindow(win)
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    data_dir   = Path(DATA_DIR)
    specimens  = find_specimens(data_dir)
    boundaries = load_boundaries()

    if not specimens:
        print(f"[ERR] Nessuno specimen trovato in {data_dir}")
        return

    print(f"[INFO] Specimen trovati: {specimens}")
    print(f"[INFO] Gia' definiti:    {list(boundaries.keys())}")
    print()

    for specimen in specimens:
        existing = boundaries.get(specimen)
        if existing:
            print(f"[SKIP] {specimen} gia' definito: {existing}  (ri-lancia con R per modificare)")

        frame, _, frame_idx = get_last_frame(specimen)
        if frame is None:
            print(f"[WARN] {specimen}: nessun frame trovato, skip.")
            continue

        print(f"[INFO] {specimen}: ultimo frame disponibile idx={frame_idx}")
        result = annotate_specimen(specimen, frame, existing=existing)

        if result is None:
            print(f"[INFO] {specimen}: skip (Q premuto). Salvo quello che ho.")
            save_boundaries(boundaries)
            return  # esci completamente se Q

        boundaries[specimen] = result
        print(f"[OK]   {specimen}: x_left={result['x_left']}  x_right={result['x_right']}")
        save_boundaries(boundaries)  # salva dopo ogni specimen

    print("\n[DONE] Tutti gli specimen processati.")
    print(json.dumps(boundaries, indent=2))


if __name__ == "__main__":
    main()
