#!/usr/bin/env python3
"""Grid search orchestrator for the existing ROI pipeline.

This version only edits config values in script files and then runs them with
subprocess. It does not import pipeline modules or call their functions.
"""

from __future__ import annotations

import itertools
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

PYTHON_EXE = sys.executable

ROI_DATASET_RUNNER = Path(r"C:\Users\tomga\Desktop\Lab Data science\Dataset_giusti\creazione_dataset\FIXED_VARIABLES_roi_dataset_runner.py")
PAUSED_FEATURE_BUILDER = Path(r"C:\Users\tomga\Desktop\Lab Data science\Dataset_giusti\feature_tables\build_feature_table_roi.py")
STD_FEATURE_BUILDER = Path(r"C:\Users\tomga\Desktop\Lab Data science\Dataset_giusti\feature_tables\build_feature_table_roi_STD.py")
PCA_PREDICTOR = Path(r"C:\Users\tomga\Desktop\Lab Data science\Dataset_giusti\Modelli_feature_tables\eda\pca_predict_breaking.py")
MECHANICAL_PROPERTIES_CSV = Path(r"C:\Users\tomga\Desktop\Lab Data science\Dataset_giusti\mechanical_properties\mechanical_properties_summary.csv")

DATASET_DIR = ROI_DATASET_RUNNER.parent / "datasets"
GRID_OUTPUT_ROOT = Path(r"C:\Users\tomga\Desktop\Lab Data science\Dataset_giusti\roi_grid_search\grid_search_results")

ENABLE_VISUALIZER = False
OSSFILTER = "core"

XSTART_VALUES = [2, 3]
XEND_VALUES = [4, 5, 6, 8, 10, 11, 12]
YSTART_VALUES = [1, 2]
YEND_VALUES = [2,3,4,5]
OFFSET_NONG3_VALUES = [1]
OFFSET_G3_VALUES = [0]

MAX_COMBINATIONS: int | None = None
KEEP_TEMP_SCRIPTS_ON_FAILURE = True


def validate_paths() -> None:
    required_paths = [ROI_DATASET_RUNNER, PAUSED_FEATURE_BUILDER, STD_FEATURE_BUILDER, PCA_PREDICTOR, MECHANICAL_PROPERTIES_CSV]
    missing = [str(path) for path in required_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("The following configured files do not exist:\n- " + "\n- ".join(missing))
    if not DATASET_DIR.is_dir():
        raise FileNotFoundError(f"DATASET_DIR does not exist: {DATASET_DIR}")


def roi_name(params: dict[str, int]) -> str:
    return f"ROI_x{params['xstart']}-{params['xend']}_y{params['ystart']}-{params['yend']}_offNonG3{params['offset_nong3']}_offG3{params['offset_g3']}"


def replace_assignment(source: str, variable: str, replacement: str) -> str:
    pattern = rf"(?m)^({re.escape(variable)}\s*=\s*).*$"
    updated, count = re.subn(pattern, lambda match: f"{match.group(1)}{replacement}", source, count=1)
    if count != 1:
        raise RuntimeError(f"Could not uniquely replace assignment '{variable}'.")
    return updated


def rewrite_config_file(path: Path, replacements: dict[str, str]) -> Path:
    source = path.read_text(encoding="utf-8")
    for variable, replacement in replacements.items():
        source = replace_assignment(source, variable, replacement)
    path.write_text(source, encoding="utf-8")
    return path


def run_script(script_path: Path, log_path: Path) -> None:
    with log_path.open("w", encoding="utf-8") as log_file:
        result = subprocess.run([PYTHON_EXE, str(script_path)], cwd=str(script_path.parent), stdout=log_file, stderr=subprocess.STDOUT, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"{script_path.name} failed with exit code {result.returncode}. See log: {log_path}")


def copy_required(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(f"Expected pipeline output was not created: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def parse_pca_summary(summary_path: Path, params: dict[str, int]) -> list[dict[str, Any]]:
    if not summary_path.is_file():
        raise FileNotFoundError(f"PCA summary file not found: {summary_path}")
    text = summary_path.read_text(encoding="utf-8", errors="replace")
    rows: list[dict[str, Any]] = []
    for match in re.finditer(r"Model:\s*(?P<model>.+?)\n\s*LOOCV RMSE\s*:\s*(?P<rmse>[-+]?\d*\.?\d+)\s*kg\s*\n\s*LOOCV R²\s*:\s*(?P<r2>[-+]?\d*\.?\d+)", text):
        rows.append({**params, "model": match.group("model").strip(), "loocv_rmse_kg": float(match.group("rmse")), "loocv_r2": float(match.group("r2"))})
    best_match = re.search(r"Best by LOOCV RMSE:\s*(?P<model>.+?)\s*\((?P<rmse>[-+]?\d*\.?\d+)\s*kg\)", text)
    for row in rows:
        row["best_model_by_rmse"] = best_match.group("model").strip() if best_match else None
        row["best_rmse_kg"] = float(best_match.group("rmse")) if best_match else None
    return rows


def roi_dataset_exists(dataset_name: str) -> bool:
    return (DATASET_DIR / f"{dataset_name}.csv").is_file() and (DATASET_DIR / f"{dataset_name}_STD.csv").is_file()


def build_grid() -> list[dict[str, int]]:
    keys = ["xstart", "xend", "ystart", "yend", "offset_nong3", "offset_g3"]
    values = [XSTART_VALUES, XEND_VALUES, YSTART_VALUES, YEND_VALUES, OFFSET_NONG3_VALUES, OFFSET_G3_VALUES]
    combos = [dict(zip(keys, comb, strict=True)) for comb in itertools.product(*values)]
    valid = [p for p in combos if p["xend"] > p["xstart"] and p["yend"] > p["ystart"]]
    return valid[:MAX_COMBINATIONS] if MAX_COMBINATIONS else valid


def run_one_roi(params: dict[str, int]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    dataset_name = roi_name(params)
    roi_folder = GRID_OUTPUT_ROOT / dataset_name
    logs_folder = roi_folder / "logs"
    datasets_folder = roi_folder / "datasets"
    feature_tables_folder = roi_folder / "feature_tables"
    analysis_folder = roi_folder / "analysis"
    for folder in [logs_folder, datasets_folder, feature_tables_folder, analysis_folder]:
        folder.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "dataset_name": dataset_name,
        "roi_parameters": params,
        "status": "running",
        "started_at_unix": time.time(),
        "roi_dataset_reused": False,
    }
    summary_rows: list[dict[str, Any]] = []

    try:
        print(f"\n{'=' * 80}")
        print(f"[START] {dataset_name}")
        print(f"[ROI] {params}")

        paused_dataset = DATASET_DIR / f"{dataset_name}.csv"
        std_dataset = DATASET_DIR / f"{dataset_name}_STD.csv"

        if roi_dataset_exists(dataset_name):
            print("[1/4] ROI datasets already exist, reusing them...")
            manifest["roi_dataset_reused"] = True
        else:
            print("[1/4] Generating ROI datasets...")
            rewrite_config_file(ROI_DATASET_RUNNER, {
                "X_START": str(params["xstart"]),
                "X_END": str(params["xend"]),
                "Y_START": str(params["ystart"]),
                "Y_END": str(params["yend"]),
                "OFFSET_NON_G3": str(params["offset_nong3"]),
                "OFFSET_G3": str(params["offset_g3"]),
                "DATASET_NAME": repr(dataset_name),
                "ENABLE_VISUALIZER": str(ENABLE_VISUALIZER),
                "OSSFILTER": repr(OSSFILTER),
            })
            run_script(ROI_DATASET_RUNNER, logs_folder / "01_roi_dataset_runner.log")

        if not paused_dataset.is_file():
            raise FileNotFoundError(f"Missing paused ROI dataset: {paused_dataset}")
        if not std_dataset.is_file():
            raise FileNotFoundError(f"Missing STD ROI dataset: {std_dataset}")
        copy_required(paused_dataset, datasets_folder / paused_dataset.name)
        copy_required(std_dataset, datasets_folder / std_dataset.name)

        print("[2/4] Building paused ROI feature table...")
        rewrite_config_file(PAUSED_FEATURE_BUILDER, {
            "DATASET_NAME": repr(dataset_name),
        })
        run_script(PAUSED_FEATURE_BUILDER, logs_folder / "02_paused_feature_table.log")
        paused_feature_source = PAUSED_FEATURE_BUILDER.parent / "output" / f"feature_table_roi_{dataset_name}.csv"
        paused_feature_output = feature_tables_folder / paused_feature_source.name
        copy_required(paused_feature_source, paused_feature_output)

        print("[3/4] Building STD ROI feature table...")
        rewrite_config_file(STD_FEATURE_BUILDER, {
            "DATASET_NAME": repr(dataset_name),
        })
        run_script(STD_FEATURE_BUILDER, logs_folder / "03_std_feature_table.log")
        std_feature_source = STD_FEATURE_BUILDER.parent / "output" / f"feature_table_roi_STD_{dataset_name}.csv"
        std_feature_output = feature_tables_folder / std_feature_source.name
        copy_required(std_feature_source, std_feature_output)

        print("[4/4] Running PCA / LOOCV / model comparison...")
        rewrite_config_file(PCA_PREDICTOR, {
            "PATH_ROI_PAUSA": repr(str(paused_feature_source)),
            "PATH_ROI_STD": repr(str(std_feature_source)),
            "PATH_MECH": repr(str(MECHANICAL_PROPERTIES_CSV)),
            "ROOT_OUT": repr(str(analysis_folder)),
        })
        run_script(PCA_PREDICTOR, logs_folder / "04_pca_predict_breaking.log")

        pca_summary = analysis_folder / "08_Summary" / "pca_prediction_summary.txt"
        summary_rows = parse_pca_summary(pca_summary, params)
        for row in summary_rows:
            row["dataset_name"] = dataset_name

        manifest.update({
            "status": "success",
            "finished_at_unix": time.time(),
            "elapsed_seconds": round(time.time() - manifest["started_at_unix"], 2),
            "pca_summary": str(pca_summary),
            "n_model_rows": len(summary_rows),
        })
        print(f"[DONE] {dataset_name} completed successfully.")

    except Exception as exc:
        manifest.update({
            "status": "failed",
            "finished_at_unix": time.time(),
            "elapsed_seconds": round(time.time() - manifest["started_at_unix"], 2),
            "error": f"{type(exc).__name__}: {exc}",
        })
        print(f"[FAIL] {dataset_name}: {manifest['error']}")

    finally:
        (roi_folder / "run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    return summary_rows, manifest


def main() -> None:
    validate_paths()
    GRID_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    grid = build_grid()
    if not grid:
        raise ValueError("The ROI grid is empty after validation.")

    print("=" * 80)
    print("ROI GRID SEARCH")
    print(f"Combinations: {len(grid)}")
    print(f"Output root: {GRID_OUTPUT_ROOT}")
    print("=" * 80)

    all_summary_rows: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []

    for index, params in enumerate(grid, start=1):
        print(f"\n[GRID {index}/{len(grid)}]")
        summary_rows, manifest = run_one_roi(params)
        all_summary_rows.extend(summary_rows)
        run_rows.append({
            **params,
            "dataset_name": manifest["dataset_name"],
            "status": manifest["status"],
            "elapsed_seconds": manifest.get("elapsed_seconds"),
            "error": manifest.get("error"),
            "roi_dataset_reused": manifest.get("roi_dataset_reused", False),
            "output_folder": str(GRID_OUTPUT_ROOT / manifest["dataset_name"]),
        })

    runs_df = pd.DataFrame(run_rows)
    runs_df.to_csv(GRID_OUTPUT_ROOT / "grid_run_status.csv", index=False)

    if all_summary_rows:
        ranking_df = pd.DataFrame(all_summary_rows).sort_values(["loocv_rmse_kg", "loocv_r2"], ascending=[True, False], na_position="last")
        ranking_columns = [
            "dataset_name",
            "xstart",
            "xend",
            "ystart",
            "yend",
            "offset_nong3",
            "offset_g3",
            "model",
            "loocv_rmse_kg",
            "loocv_r2",
            "best_model_by_rmse",
            "best_rmse_kg",
        ]
        ranking_df = ranking_df[[column for column in ranking_columns if column in ranking_df.columns]]
        ranking_df.to_csv(GRID_OUTPUT_ROOT / "roi_model_ranking.csv", index=False)
        best_per_roi = (
            ranking_df.sort_values(["dataset_name", "loocv_rmse_kg", "loocv_r2"], ascending=[True, True, False], na_position="last")
            .groupby("dataset_name", as_index=False)
            .first()
            .sort_values(["loocv_rmse_kg", "loocv_r2"], ascending=[True, False], na_position="last")
        )
        best_per_roi.to_csv(GRID_OUTPUT_ROOT / "roi_best_model_ranking.csv", index=False)

    successful = int((runs_df["status"] == "success").sum())
    failed = int((runs_df["status"] == "failed").sum())

    print("\n" + "=" * 80)
    print("GRID SEARCH FINISHED")
    print(f"Successful ROI runs: {successful}")
    print(f"Failed ROI runs: {failed}")
    print(f"Results: {GRID_OUTPUT_ROOT}")
    print("=" * 80)


if __name__ == "__main__":
    main()
