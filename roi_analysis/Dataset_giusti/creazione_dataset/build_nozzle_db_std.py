"""
build_nozzle_db_std.py
---------------------
Pipeline standard-only: itera sui provini standard con molti layer e costruisce
un nozzle_db dedicato, usando le stesse regole di build_nozzle_db.py.

Output: datasets/nozzle_db_std.csv
"""

import sys
import pandas as pd
from pathlib import Path
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT  = SCRIPT_DIR.parent
WORKSPACE_ROOT = REPO_ROOT.parent

sys.path[:0] = [str(WORKSPACE_ROOT), str(REPO_ROOT), str(SCRIPT_DIR)]

from Dataset_giusti.creazione_dataset.frame_selector import SPECIMENS, LayerNotDefinedError
from Dataset_giusti.creazione_dataset.nozzle_tracker import load_kernel
from Dataset_giusti.creazione_dataset.build_nozzle_db import process_layer, select_kernel


# ===========================================================================
# CONFIG
# ===========================================================================

STANDARD_SPECIMENS = [
    "Rec-027_std_2",
    "Rec-G3_std_1",
]

# Kernel per provini standard (senza G3 nel nome)
KERNEL_PATH_STD = WORKSPACE_ROOT / "data" / "feature_extraction" / "kernel_2.png"

# Kernel per provini G3 (riprese in momento diverso, inquadratura diversa)
KERNEL_PATH_G3  = WORKSPACE_ROOT / "data" / "feature_extraction" / "kernel_3.png"

MARGIN = 5

# Output nella sottocartella datasets/ dentro la cartella di lavoro
OUTPUT_CSV = SCRIPT_DIR / "datasets" / "nozzle_db_std.csv"

SAVE_DEBUG_IMAGES   = False
DEBUG_MAX_PER_LAYER = 5

# ===========================================================================


def _resolve_specimens(specimen_subset: list[str]) -> dict:
    missing = [name for name in specimen_subset if name not in SPECIMENS]
    if missing:
        print(f"[WARN] Specimen non trovati in SPECIMENS: {missing}")
    return {
        name: spec for name, spec in SPECIMENS.items()
        if name in specimen_subset and spec["valid"] and spec["layers_front"] is not None
    }


def build_nozzle_db_std(
    specimen_subset: list[str] = STANDARD_SPECIMENS,
    kernel_path_std: Path = KERNEL_PATH_STD,
    kernel_path_g3:  Path = KERNEL_PATH_G3,
    margin: int = MARGIN,
    output_csv: Path = OUTPUT_CSV,
    save_debug: bool = SAVE_DEBUG_IMAGES,
    debug_max: int = DEBUG_MAX_PER_LAYER,
) -> pd.DataFrame:
    kernel_std = load_kernel(kernel_path_std)
    kernel_g3  = load_kernel(kernel_path_g3)
    print(f"[INFO] Kernel STD caricato: {kernel_path_std}  shape={kernel_std.shape}")
    print(f"[INFO] Kernel G3  caricato: {kernel_path_g3}   shape={kernel_g3.shape}")
    print(f"[INFO] Margine frame      : {margin}")

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    debug_dir = None
    if save_debug:
        debug_dir = output_csv.parent / "debug_nozzle_std"
        debug_dir.mkdir(exist_ok=True)

    valid_specimens = _resolve_specimens(specimen_subset)
    if not valid_specimens:
        print("[WARN] Nessun provino valido da processare.")
        return pd.DataFrame()

    print(f"[INFO] Provini STD da processare: {len(valid_specimens)}")
    for name in valid_specimens:
        k_label = "G3" if "G3" in name else "STD"
        print(f"  - {name}  [kernel_{k_label}]")
    print()

    all_records = []
    for specimen_name in tqdm(valid_specimens, desc="Provini STD", unit="spec"):
        kernel = select_kernel(specimen_name, kernel_std, kernel_g3)
        num_layers = len(valid_specimens[specimen_name]["layers_front"])
        for layer_idx in range(1, num_layers + 1):
            tqdm.write(f"  [{specimen_name}] Layer {layer_idx}...")
            try:
                records = process_layer(
                    specimen_name,
                    layer_idx,
                    kernel,
                    margin,
                    debug_dir,
                    debug_max,
                )
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
    build_nozzle_db_std()
