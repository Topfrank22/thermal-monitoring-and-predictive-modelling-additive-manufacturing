"""
build_roi_dataset_std.py
-----------------------
Costruisce il dataset ROI per i provini standard usando nozzle_db_std.csv.
Usa le stesse regole e metriche di roi_dataset_builder.py.

Output: datasets/ROI_STD_wide_3_10_depth_1_3.csv
"""

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT  = SCRIPT_DIR.parent
WORKSPACE_ROOT = REPO_ROOT.parent

sys.path[:0] = [str(WORKSPACE_ROOT), str(REPO_ROOT), str(SCRIPT_DIR)]

from Dataset_giusti.creazione_dataset.roi_dataset_builder import (
    build_roi_dataset,
    ROI_PARAMS_STD,
    ROI_PARAMS_G3,
    METRICS,
    PIECE_BOUNDARIES,
)


# ===========================================================================
# CONFIG
# ===========================================================================

NOZZLE_DB_STD = SCRIPT_DIR / "datasets" / "nozzle_db_std.csv"
DATASET_NAME  = "ROI_STD_wide_3_10_depth_1_3"
OUTPUT_DIR    = SCRIPT_DIR / "datasets"

OSS_FILTER        = "core"
ENABLE_VISUALIZER = False

# ===========================================================================


def build_roi_dataset_std():
    return build_roi_dataset(
        nozzle_db_path=NOZZLE_DB_STD,
        dataset_name=DATASET_NAME,
        output_dir=OUTPUT_DIR,
        params_std=ROI_PARAMS_STD,
        params_g3=ROI_PARAMS_G3,
        metrics=METRICS,
        oss_filter=OSS_FILTER,
        enable_visualizer=ENABLE_VISUALIZER,
        boundaries=PIECE_BOUNDARIES,
    )


if __name__ == "__main__":
    build_roi_dataset_std()
