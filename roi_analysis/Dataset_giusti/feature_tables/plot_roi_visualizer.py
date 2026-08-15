"""
plot_roi_visualizer.py
----------------------
Generates four sets of composite PNGs per specimen family, each in its own
output subfolder under Analisi_dataset/<DATASET_NAME>/ROI_visualizer/:

    All ROIs/
        All core frames, full colour scale [T_MIN, T_MAX].

    Complete ROIs/
        Only frames where roi_complete == True, same absolute colour scale.

    Compl delta ROIs/
        Only complete frames. Each pixel is shown as:
            delta_px = T_px(layer k, frame f) - mean_T_px(layer 1)
        where mean_T_px(layer 1) is the per-pixel mean over ALL complete
        frames of layer 1 for that specimen.
        Colourmap: 'RdBu_r' (blue = cooler, red = warmer than layer-1 baseline).
        Scale is symmetric: [-DELTA_MAX, +DELTA_MAX] (configurable).

    Compl ROIs heating rate/
        Only complete frames. Each pixel is shown as:
            dT_px = T_px(layer k, frame f) - T_px(layer k-1, same f_col)
        For layer 1 (no previous layer) the absolute temperature is shown
        (same as Complete ROIs, scale [T_MIN, T_MAX]).
        Colourmap: 'RdBu_r' for delta layers, 'inferno' for layer 1.

ROI pixel arrays are reconstructed from the 8-bit PNG frames using the
global thermal range [T_MIN, T_MAX].  ROI coordinates (roi_r0/c0/r1/c1)
and completeness flags (roi_complete) are read directly from the dataset CSV.

CONFIG block at the top — set DATASET_NAME and visual parameters.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent

CREAZIONE_DS = None
for parent in Path(__file__).resolve().parents:
    candidate = parent / "creazione_dataset"
    if candidate.exists() and candidate.is_dir():
        CREAZIONE_DS = candidate
        break
if CREAZIONE_DS is None:
    raise FileNotFoundError("Folder 'creazione_dataset' not found.")

CONFIG_DIR = None
for parent in Path(__file__).resolve().parents:
    if (parent / "config.py").exists():
        CONFIG_DIR = parent
        break
if CONFIG_DIR is None:
    raise FileNotFoundError("config.py not found.")

sys.path.insert(0, str(CONFIG_DIR))
sys.path.insert(0, str(CREAZIONE_DS))

from config import DATA_DIR           # noqa: E402
from frame_selector import SPECIMENS  # noqa: E402

# ===========================================================================
#  CONFIG
# ===========================================================================

DATASET_NAME = "ROI_wide_3_10_depth_1_3"  # CSV filename (without .csv)

FAMILIES = {
    "10s":    ["Rec-022_10s_1", "Rec-029_10s_2"],
    "30s":    ["Rec-025_30s_1", "Rec-028_30s_2", "Rec-031_30s_3", "Rec-032_30s_4"],
    "60s":    ["Rec-026_60s_2"],
    "90s":    ["Rec-024_90s_1", "Rec-030_90s_2"],
    "G3_S60": ["Rec-G3_S60_1"],
    "std":    ["Rec-027_std_2"],
    "G3_std": ["Rec-G3_std_1"],
}

# Output thumbnail size (nearest-neighbour resize of the ROI crop).
THUMB_W = 80   # px width per thumbnail
THUMB_H = 32   # px height per thumbnail — keep aspect close to ROI shape

# Colourmaps
COLORMAP_ABS   = "inferno"   # absolute temperature views
COLORMAP_DELTA = "RdBu_r"   # delta / heating-rate views (diverging)

# Symmetric range for delta / heating-rate colourmaps [deg C].
# Pixels outside this range are clipped to the colourmap extremes.
DELTA_MAX = 30.0   # [°C]  e.g. ±30 °C shown as full blue/red

# Gaps / separators (output pixels)
GAP_FRAMES_H    = 2
GAP_SPECIMENS_V = 4
GAP_LAYERS_V    = 18

# Colours (RGB tuples)
SEP_LAYER_COLOR = (40,  40,  40)
SEP_SPEC_COLOR  = (190, 190, 190)
BG_COLOR        = (240, 240, 240)
MISSING_COLOR   = (200, 200, 200)

LABEL_WIDTH = 160
DPI         = 150

# ===========================================================================
#  END CONFIG
# ===========================================================================

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as mcm

ANALISI_DIR  = SCRIPT_DIR.parent / "Analisi_dataset"
VIZ_ROOT     = ANALISI_DIR / DATASET_NAME / "ROI_visualizer"
FRAMES_ROOT  = Path(DATA_DIR)

# ---------------------------------------------------------------------------
# Load the dataset CSV
# ---------------------------------------------------------------------------
DATASETS_DIR = CREAZIONE_DS / "datasets"
dataset_csv  = DATASETS_DIR / f"{DATASET_NAME}.csv"

if not dataset_csv.exists():
    raise FileNotFoundError(
        f"Dataset CSV not found:\n  {dataset_csv}\n"
        "Run roi_dataset_builder.py first."
    )

print(f"[INFO] Loading dataset: {dataset_csv}")
_load_cols = [
    "specimen", "frame_idx", "layer_index", "frame_type",
    "roi_r0", "roi_c0", "roi_r1", "roi_c1",
]
# roi_complete is optional (may not be present if piece_boundaries.json is missing)
try:
    df_all = pd.read_csv(dataset_csv, usecols=_load_cols + ["roi_complete"])
    HAS_COMPLETE_COL = True
except ValueError:
    df_all = pd.read_csv(dataset_csv, usecols=_load_cols)
    df_all["roi_complete"] = True   # treat all as complete if column absent
    HAS_COMPLETE_COL = False
    print("[WARN] 'roi_complete' column not found — treating all frames as complete.")

df_core = df_all[df_all["frame_type"] == "core"].copy()
df_core["roi_complete"] = df_core["roi_complete"].fillna(False).astype(bool)

# ROI lookup: (specimen, frame_idx) -> roi coords  [used by all modes]
ROI_LOOKUP = df_core.set_index(["specimen", "frame_idx"])[
    ["roi_r0", "roi_c0", "roi_r1", "roi_c1", "roi_complete", "layer_index"]
]
print(f"[INFO] ROI lookup: {len(ROI_LOOKUP)} core frames  "
      f"({df_core['roi_complete'].sum()} complete)")

# Global temperature range from CSV stats
try:
    df_tmp = pd.read_csv(dataset_csv, usecols=["roi_min_C", "roi_max_C", "frame_type"])
    df_tmp = df_tmp[df_tmp["frame_type"] == "core"]
    GLOBAL_TMIN = float(df_tmp["roi_min_C"].min())
    GLOBAL_TMAX = float(df_tmp["roi_max_C"].max())
    print(f"[INFO] Global temp range: [{GLOBAL_TMIN:.1f}, {GLOBAL_TMAX:.1f}] °C")
except Exception:
    GLOBAL_TMIN, GLOBAL_TMAX = 41.8, 180.2
    print(f"[WARN] Falling back to T range [{GLOBAL_TMIN}, {GLOBAL_TMAX}] °C")

# Colourmap LUTs
_CMAP_ABS   = mcm.get_cmap(COLORMAP_ABS,   256)
_CMAP_DELTA = mcm.get_cmap(COLORMAP_DELTA, 256)


# ===========================================================================
# Core pixel helpers
# ===========================================================================

def _gray_to_celsius(gray: np.ndarray) -> np.ndarray:
    """8-bit greyscale -> °C (float32)."""
    return GLOBAL_TMIN + (gray.astype(np.float32) / 255.0) * (GLOBAL_TMAX - GLOBAL_TMIN)


def _load_roi_celsius(frames_dir: Path, frame_idx: int,
                      specimen: str) -> np.ndarray | None:
    """
    Load the ROI crop for one frame and return it in °C (float32 2D array).
    Returns None if the frame is missing or not in the CSV.
    """
    try:
        row = ROI_LOOKUP.loc[(specimen, frame_idx)]
        r0, c0 = int(row["roi_r0"]), int(row["roi_c0"])
        r1, c1 = int(row["roi_r1"]), int(row["roi_c1"])
    except KeyError:
        return None
    fpath = frames_dir / f"{frame_idx:04d}.png"
    if not fpath.exists():
        return None
    try:
        gray = np.array(Image.open(fpath).convert("L"))
        crop = gray[r0: r1 + 1, c0: c1 + 1]
        return _gray_to_celsius(crop) if crop.size > 0 else None
    except Exception:
        return None


def _celsius_to_thumb_abs(roi_c: np.ndarray) -> np.ndarray:
    """Absolute temperature array -> (THUMB_H, THUMB_W, 3) uint8 via inferno."""
    norm = np.clip((roi_c - GLOBAL_TMIN) / (GLOBAL_TMAX - GLOBAL_TMIN), 0.0, 1.0)
    rgb  = (_CMAP_ABS(norm)[:, :, :3] * 255).astype(np.uint8)
    return np.array(Image.fromarray(rgb).resize((THUMB_W, THUMB_H), Image.NEAREST))


def _celsius_to_thumb_delta(delta_c: np.ndarray) -> np.ndarray:
    """Delta temperature array -> (THUMB_H, THUMB_W, 3) uint8 via RdBu_r.
    Range is symmetric: [-DELTA_MAX, +DELTA_MAX]."""
    norm = np.clip((delta_c + DELTA_MAX) / (2 * DELTA_MAX), 0.0, 1.0)
    rgb  = (_CMAP_DELTA(norm)[:, :, :3] * 255).astype(np.uint8)
    return np.array(Image.fromarray(rgb).resize((THUMB_W, THUMB_H), Image.NEAREST))


def _blank() -> np.ndarray:
    return np.full((THUMB_H, THUMB_W, 3), MISSING_COLOR, dtype=np.uint8)


# ===========================================================================
# Pre-compute per-specimen baselines needed for delta / heating-rate modes
# ===========================================================================

def compute_layer1_mean(
    spec_key: str,
    frames_per_layer: list[list[int]],
    only_complete: bool = True,
) -> np.ndarray | None:
    """
    Compute the per-pixel mean temperature (float32) over ALL frames
    of layer 1 (index 0) for `spec_key`.
    If only_complete=True, only frames with roi_complete==True are used.
    Returns None if no valid frames are found.
    """
    if not frames_per_layer:
        return None
    seq_name   = SPECIMENS[spec_key]["seq_name"]
    frames_dir = FRAMES_ROOT / f"{seq_name}_frames"
    layer1_frames = frames_per_layer[0]
    arrays = []
    for fidx in layer1_frames:
        if only_complete:
            try:
                if not ROI_LOOKUP.loc[(spec_key, fidx), "roi_complete"]:
                    continue
            except KeyError:
                continue
        arr = _load_roi_celsius(frames_dir, fidx, spec_key)
        if arr is not None:
            arrays.append(arr)
    if not arrays:
        return None
    return np.mean(np.stack(arrays, axis=0), axis=0).astype(np.float32)


def load_layer_roi_celsius(
    spec_key: str,
    frames_per_layer: list[list[int]],
    layer_idx: int,
    only_complete: bool = True,
) -> dict[int, np.ndarray]:
    """
    Load all (optionally complete) ROI frames for `spec_key` at `layer_idx`.
    Returns {frame_idx: celsius_array}.
    """
    if layer_idx >= len(frames_per_layer):
        return {}
    seq_name   = SPECIMENS[spec_key]["seq_name"]
    frames_dir = FRAMES_ROOT / f"{seq_name}_frames"
    result = {}
    for fidx in frames_per_layer[layer_idx]:
        if only_complete:
            try:
                if not ROI_LOOKUP.loc[(spec_key, fidx), "roi_complete"]:
                    continue
            except KeyError:
                continue
        arr = _load_roi_celsius(frames_dir, fidx, spec_key)
        if arr is not None:
            result[fidx] = arr
    return result


# ===========================================================================
# Canvas assembly (generic — works for all modes)
# ===========================================================================

def _assemble_canvas(rows: list, max_frames_per_layer: list, max_layers: int) -> np.ndarray:
    max_f_global = max(max_frames_per_layer) if max_frames_per_layer else 1
    canvas_w = LABEL_WIDTH + max_f_global * (THUMB_W + GAP_FRAMES_H)
    n_spec_seps = len(rows) - max_layers
    canvas_h = (
        len(rows) * THUMB_H
        + n_spec_seps * GAP_SPECIMENS_V
        + max_layers  * GAP_LAYERS_V
    )
    canvas = np.full((canvas_h, canvas_w, 3), BG_COLOR, dtype=np.uint8)
    y, prev_layer = 0, -1
    for row_arr, spec_key, layer_idx, _ in rows:
        if layer_idx != prev_layer:
            canvas[y: y + GAP_LAYERS_V] = SEP_LAYER_COLOR
            y += GAP_LAYERS_V
            prev_layer = layer_idx
        else:
            canvas[y: y + GAP_SPECIMENS_V] = SEP_SPEC_COLOR
            y += GAP_SPECIMENS_V
        canvas[y: y + THUMB_H, :canvas_w] = row_arr
        y += THUMB_H
    return canvas


def _render_and_save(
    canvas: np.ndarray,
    rows: list,
    max_frames_per_layer: list,
    family_name: str,
    out_path: Path,
    cmap_name: str,
    vmin: float,
    vmax: float,
    title_suffix: str,
) -> None:
    canvas_h, canvas_w, _ = canvas.shape
    fig, ax = plt.subplots(figsize=(canvas_w / DPI, canvas_h / DPI), dpi=DPI)
    ax.imshow(canvas, aspect="auto", interpolation="nearest")
    ax.axis("off")

    y, prev_layer = 0, -1
    for row_arr, spec_key, layer_idx, _ in rows:
        if layer_idx != prev_layer:
            bar_cy = y + GAP_LAYERS_V / 2
            ax.text(LABEL_WIDTH / 2, bar_cy, f"Layer {layer_idx + 1}",
                    color="white", fontsize=7, fontweight="bold",
                    ha="center", va="center")
            ax.text(LABEL_WIDTH + (canvas_w - LABEL_WIDTH) / 2, bar_cy,
                    f"— Layer {layer_idx + 1} —",
                    color="white", fontsize=6.5, fontweight="bold",
                    ha="center", va="center", alpha=0.55)
            y += GAP_LAYERS_V
            prev_layer = layer_idx
        else:
            y += GAP_SPECIMENS_V
        short = spec_key.replace("Rec-", "").replace("_", " ")
        ax.text(LABEL_WIDTH / 2, y + THUMB_H / 2, short,
                color=(0.1, 0.1, 0.1), fontsize=6.5, ha="center", va="center")
        y += THUMB_H

    max_f = max(max_frames_per_layer) if max_frames_per_layer else 1
    for f_col in range(max_f):
        xc = LABEL_WIDTH + f_col * (THUMB_W + GAP_FRAMES_H) + THUMB_W / 2
        ax.text(xc, GAP_LAYERS_V / 2, str(f_col),
                color="white", fontsize=5, ha="center", va="center")

    sm = plt.cm.ScalarMappable(cmap=cmap_name, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation="vertical",
                        fraction=0.008, pad=0.005, shrink=0.6)
    cbar.set_label("°C", fontsize=7, color="white")
    cbar.ax.yaxis.set_tick_params(color="white", labelsize=6, labelcolor="white")
    cbar.outline.set_edgecolor("white")

    fig.suptitle(
        f"Family: {family_name}  |  {DATASET_NAME}  |  {title_suffix}",
        fontsize=8, color="white",
        bbox=dict(boxstyle="square,pad=0.3", fc=(0.1, 0.1, 0.1), ec="none"),
    )
    plt.subplots_adjust(left=0, right=0.98, top=0.99, bottom=0)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight", facecolor="#111111")
    plt.close(fig)
    print(f"    [OK] {out_path.name}  ({canvas_w}x{canvas_h} px)")


# ===========================================================================
# Mode builders
# ===========================================================================

def _build_rows_abs(family_name: str, spec_keys: list[str],
                    only_complete: bool) -> tuple | None:
    """
    Build (rows, max_frames_per_layer, max_layers) for absolute-temp modes.
    If only_complete=True, frames that are not roi_complete are shown as blanks.
    """
    spec_data, max_layers = {}, 0
    for sk in spec_keys:
        if sk not in SPECIMENS: continue
        info = SPECIMENS[sk]
        if not info.get("valid", True) or info.get("layers_front") is None: continue
        fpl = [list(range(fs, fe + 1)) for (fs, fe) in info["layers_front"]]
        spec_data[sk] = fpl
        max_layers = max(max_layers, len(fpl))
    if not spec_data: return None

    max_fpl = []
    for li in range(max_layers):
        mx = max((len(fpl[li]) for fpl in spec_data.values() if li < len(fpl)), default=0)
        max_fpl.append(mx)
    canvas_w = LABEL_WIDTH + max(max_fpl, default=1) * (THUMB_W + GAP_FRAMES_H)
    rows = []

    for layer_idx in range(max_layers):
        for sk, fpl in spec_data.items():
            seq_name   = SPECIMENS[sk]["seq_name"]
            frames_dir = FRAMES_ROOT / f"{seq_name}_frames"
            row = np.full((THUMB_H, canvas_w, 3), BG_COLOR, dtype=np.uint8)
            if layer_idx < len(fpl):
                for f_col, fidx in enumerate(fpl[layer_idx]):
                    x0 = LABEL_WIDTH + f_col * (THUMB_W + GAP_FRAMES_H)
                    is_ok = True
                    if only_complete:
                        try:
                            is_ok = bool(ROI_LOOKUP.loc[(sk, fidx), "roi_complete"])
                        except KeyError:
                            is_ok = False
                    if is_ok:
                        arr = _load_roi_celsius(frames_dir, fidx, sk)
                        thumb = _celsius_to_thumb_abs(arr) if arr is not None else _blank()
                    else:
                        thumb = _blank()
                    row[:, x0: x0 + THUMB_W] = thumb
                    sx = x0 + THUMB_W
                    if sx + GAP_FRAMES_H <= canvas_w:
                        row[:, sx: sx + GAP_FRAMES_H] = SEP_SPEC_COLOR
            rows.append((row, sk, layer_idx, True))

    return rows, max_fpl, max_layers


def _build_rows_delta(family_name: str, spec_keys: list[str]) -> tuple | None:
    """
    Build rows for Compl delta ROIs:
        delta = T_px(layer k, frame f) - mean_T_px(layer 1, complete frames)
    """
    spec_data, max_layers = {}, 0
    for sk in spec_keys:
        if sk not in SPECIMENS: continue
        info = SPECIMENS[sk]
        if not info.get("valid", True) or info.get("layers_front") is None: continue
        fpl = [list(range(fs, fe + 1)) for (fs, fe) in info["layers_front"]]
        spec_data[sk] = fpl
        max_layers = max(max_layers, len(fpl))
    if not spec_data: return None

    # Precompute layer-1 baseline for each specimen
    baselines = {}
    for sk, fpl in spec_data.items():
        baselines[sk] = compute_layer1_mean(sk, fpl, only_complete=True)
        if baselines[sk] is None:
            print(f"  [WARN] No complete layer-1 frames for '{sk}' (delta mode).")

    max_fpl = []
    for li in range(max_layers):
        mx = max((len(fpl[li]) for fpl in spec_data.values() if li < len(fpl)), default=0)
        max_fpl.append(mx)
    canvas_w = LABEL_WIDTH + max(max_fpl, default=1) * (THUMB_W + GAP_FRAMES_H)
    rows = []

    for layer_idx in range(max_layers):
        for sk, fpl in spec_data.items():
            seq_name   = SPECIMENS[sk]["seq_name"]
            frames_dir = FRAMES_ROOT / f"{seq_name}_frames"
            row = np.full((THUMB_H, canvas_w, 3), BG_COLOR, dtype=np.uint8)
            base = baselines.get(sk)
            if layer_idx < len(fpl):
                for f_col, fidx in enumerate(fpl[layer_idx]):
                    x0 = LABEL_WIDTH + f_col * (THUMB_W + GAP_FRAMES_H)
                    try:
                        is_ok = bool(ROI_LOOKUP.loc[(sk, fidx), "roi_complete"])
                    except KeyError:
                        is_ok = False
                    if is_ok and base is not None:
                        arr = _load_roi_celsius(frames_dir, fidx, sk)
                        if arr is not None and arr.shape == base.shape:
                            thumb = _celsius_to_thumb_delta(arr - base)
                        else:
                            thumb = _blank()
                    else:
                        thumb = _blank()
                    row[:, x0: x0 + THUMB_W] = thumb
                    sx = x0 + THUMB_W
                    if sx + GAP_FRAMES_H <= canvas_w:
                        row[:, sx: sx + GAP_FRAMES_H] = SEP_SPEC_COLOR
            rows.append((row, sk, layer_idx, True))

    return rows, max_fpl, max_layers


def _build_rows_heating_rate(family_name: str, spec_keys: list[str]) -> tuple | None:
    """
    Build rows for Compl ROIs heating rate:
        layer 1  -> absolute temperature (inferno, [T_MIN, T_MAX])
        layer k  -> T_px(layer k, f_col) - T_px(layer k-1, same f_col)  [RdBu_r]
    Only complete frames are used.  If the corresponding frame in layer k-1
    is missing/incomplete, the cell is shown as a blank.
    """
    spec_data, max_layers = {}, 0
    for sk in spec_keys:
        if sk not in SPECIMENS: continue
        info = SPECIMENS[sk]
        if not info.get("valid", True) or info.get("layers_front") is None: continue
        fpl = [list(range(fs, fe + 1)) for (fs, fe) in info["layers_front"]]
        spec_data[sk] = fpl
        max_layers = max(max_layers, len(fpl))
    if not spec_data: return None

    # Pre-load ALL complete celsius arrays per (specimen, layer_idx, f_col)
    # Structure:  cache[sk][layer_idx][f_col] = celsius_array or None
    cache: dict[str, list[dict[int, np.ndarray | None]]] = {}
    for sk, fpl in spec_data.items():
        seq_name   = SPECIMENS[sk]["seq_name"]
        frames_dir = FRAMES_ROOT / f"{seq_name}_frames"
        cache[sk] = []
        for li, layer_frames in enumerate(fpl):
            layer_dict = {}
            for f_col, fidx in enumerate(layer_frames):
                try:
                    is_ok = bool(ROI_LOOKUP.loc[(sk, fidx), "roi_complete"])
                except KeyError:
                    is_ok = False
                if is_ok:
                    layer_dict[f_col] = _load_roi_celsius(frames_dir, fidx, sk)
                else:
                    layer_dict[f_col] = None
            cache[sk].append(layer_dict)

    max_fpl = []
    for li in range(max_layers):
        mx = max((len(fpl[li]) for fpl in spec_data.values() if li < len(fpl)), default=0)
        max_fpl.append(mx)
    canvas_w = LABEL_WIDTH + max(max_fpl, default=1) * (THUMB_W + GAP_FRAMES_H)
    rows = []

    for layer_idx in range(max_layers):
        for sk, fpl in spec_data.items():
            row = np.full((THUMB_H, canvas_w, 3), BG_COLOR, dtype=np.uint8)
            if layer_idx < len(fpl):
                n_frames = len(fpl[layer_idx])
                for f_col in range(n_frames):
                    x0 = LABEL_WIDTH + f_col * (THUMB_W + GAP_FRAMES_H)
                    arr_curr = cache[sk][layer_idx].get(f_col)
                    if arr_curr is None:
                        thumb = _blank()
                    elif layer_idx == 0:
                        # Layer 1: show absolute temperature
                        thumb = _celsius_to_thumb_abs(arr_curr)
                    else:
                        arr_prev = cache[sk][layer_idx - 1].get(f_col)
                        if arr_prev is None or arr_prev.shape != arr_curr.shape:
                            thumb = _blank()
                        else:
                            thumb = _celsius_to_thumb_delta(arr_curr - arr_prev)
                    row[:, x0: x0 + THUMB_W] = thumb
                    sx = x0 + THUMB_W
                    if sx + GAP_FRAMES_H <= canvas_w:
                        row[:, sx: sx + GAP_FRAMES_H] = SEP_SPEC_COLOR
            rows.append((row, sk, layer_idx, True))

    return rows, max_fpl, max_layers


# ===========================================================================
# Main loop: 4 modes
# ===========================================================================

MODES = [
    {
        "folder":  "All ROIs",
        "builder": lambda fn, sk: _build_rows_abs(fn, sk, only_complete=False),
        "cmap":    COLORMAP_ABS,
        "vmin":    GLOBAL_TMIN if "GLOBAL_TMIN" in dir() else 41.8,
        "vmax":    GLOBAL_TMAX if "GLOBAL_TMAX" in dir() else 180.2,
        "suffix":  f"All frames  |  cmap: {COLORMAP_ABS}",
    },
    {
        "folder":  "Complete ROIs",
        "builder": lambda fn, sk: _build_rows_abs(fn, sk, only_complete=True),
        "cmap":    COLORMAP_ABS,
        "vmin":    GLOBAL_TMIN if "GLOBAL_TMIN" in dir() else 41.8,
        "vmax":    GLOBAL_TMAX if "GLOBAL_TMAX" in dir() else 180.2,
        "suffix":  f"roi_complete only  |  cmap: {COLORMAP_ABS}",
    },
    {
        "folder":  "Compl delta ROIs",
        "builder": lambda fn, sk: _build_rows_delta(fn, sk),
        "cmap":    COLORMAP_DELTA,
        "vmin":    -DELTA_MAX,
        "vmax":    +DELTA_MAX,
        "suffix":  f"\u0394T vs layer-1 mean  |  cmap: {COLORMAP_DELTA}  [±{DELTA_MAX}°C]",
    },
    {
        "folder":  "Compl ROIs heating rate",
        "builder": lambda fn, sk: _build_rows_heating_rate(fn, sk),
        "cmap":    COLORMAP_DELTA,  # layer 1 uses ABS internally
        "vmin":    -DELTA_MAX,
        "vmax":    +DELTA_MAX,
        "suffix":  f"\u0394T vs prev layer (same f_col)  |  layer 1 = abs temp",
    },
]

print(f"[INFO] ROI Visualizer  —  dataset: {DATASET_NAME}")
print(f"[INFO] Output root: {VIZ_ROOT}\n")

for mode in MODES:
    folder_name = mode["folder"]
    out_dir     = VIZ_ROOT / folder_name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[MODE] {folder_name}")

    for family_name, spec_keys in FAMILIES.items():
        print(f"  [FAM] {family_name}")
        result = mode["builder"](family_name, spec_keys)
        if result is None:
            print(f"    [SKIP] No valid data.")
            continue
        rows, max_fpl, max_layers = result
        canvas = _assemble_canvas(rows, max_fpl, max_layers)
        out_path = out_dir / f"ROI_vis_{family_name}.png"
        _render_and_save(
            canvas, rows, max_fpl,
            family_name, out_path,
            cmap_name   = mode["cmap"],
            vmin        = mode["vmin"],
            vmax        = mode["vmax"],
            title_suffix= mode["suffix"],
        )

print(f"\n[DONE]  {VIZ_ROOT}")
