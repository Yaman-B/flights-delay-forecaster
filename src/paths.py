"""Canonical project paths. Anchored to this file's location so they resolve
no matter where a script or notebook is launched from."""
from pathlib import Path

# paths.py lives at <repo>/src/paths.py -> the repo root is two levels up.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH  = PROJECT_ROOT / "data" / "processed" / "flights_weather.parquet"
FIG_DIR    = PROJECT_ROOT / "reports" / "figures"
TBL_DIR    = PROJECT_ROOT / "reports" / "tables"
TARGET_COL = "disrupted"

FIG_DIR.mkdir(parents=True, exist_ok=True)
TBL_DIR.mkdir(parents=True, exist_ok=True)