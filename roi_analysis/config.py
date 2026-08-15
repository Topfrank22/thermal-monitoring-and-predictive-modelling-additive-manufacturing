"""Configurazione minima del progetto.

`DATA_DIR` punta alla cartella che contiene i file `.seq` e le cartelle
`<seq_name>_frames` generate dal notebook.

Priorita':
1. variabile ambiente `ROI_STUDY_DATA_DIR`
2. cartella locale `Thermal_videos/` accanto a questo file
3. fallback al path storico del progetto originale
"""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent

_default_data_dir = PROJECT_ROOT / "Thermal_videos"
_legacy_data_dir = Path(r"C:\Users\tomga\Desktop\Lab Data science\Thermal_videos")

DATA_DIR = Path(os.environ.get("ROI_STUDY_DATA_DIR", _default_data_dir))
if not DATA_DIR.is_absolute():
	DATA_DIR = (PROJECT_ROOT / DATA_DIR).resolve()

if not DATA_DIR.exists() and _legacy_data_dir.exists():
	DATA_DIR = _legacy_data_dir

DATA_DIR = str(DATA_DIR)
