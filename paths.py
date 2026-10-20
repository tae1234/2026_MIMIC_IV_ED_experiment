"""
paths.py -- machine-specific locations, read from environment variables.

    MIMIC_DIR : folder containing the MIMIC-IV / MIMIC-IV-ED CSV(.gz) files
                (searched recursively, e.g. <MIMIC_DIR>/ed, /hosp, /icu)
    WORK_DIR  : intermediate ROW-LEVEL files (parquet, caches, predictions,
                fitted models). Never share or commit (PhysioNet DUA).
    OUT_DIR   : aggregate outputs only (tables, figures). Safe to share.

Importing this module creates WORK_DIR/OUT_DIR and makes WORK_DIR the current
working directory, so the analysis scripts can keep their original relative
file names (e.g. "triage_abbr.parquet").
"""
import glob
import gzip
import os
import shutil
import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
MIMIC_DIR = Path(os.environ.get("MIMIC_DIR", "./mimic")).resolve()
WORK_DIR = Path(os.environ.get("WORK_DIR", "./work")).resolve()
OUT_DIR = Path(os.environ.get("OUT_DIR", "./outputs")).resolve()

# Search order used by ensure_parquet() in 05_build_ed_master_dataset.py
MIMIC_DIRS = [str(MIMIC_DIR / "ed"), str(MIMIC_DIR / "hosp"), str(MIMIC_DIR / "icu"), str(MIMIC_DIR)]

WORK_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)
os.chdir(WORK_DIR)
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))


def ensure_parquet(table_name: str, search_dirs: list = MIMIC_DIRS) -> str:
    """Return '<table>.parquet' in WORK_DIR, converting the MIMIC CSV(.gz) once."""
    import polars as pl
    parquet_file = f"{table_name}.parquet"
    if os.path.exists(parquet_file):
        return parquet_file
    found_file = next(
        (matches[0] for d in search_dirs if
         (matches := sorted(glob.glob(os.path.join(d, "**", f"{table_name}.csv*"), recursive=True)))),
        None,
    )
    if not found_file:
        raise FileNotFoundError(f"Missing source file for '{table_name}' (.csv or .csv.gz) under {MIMIC_DIR}")
    print(f"Converting {found_file} -> {parquet_file}")
    opts = dict(infer_schema_length=10000, null_values=["", "NA", "NaN"])
    try:
        pl.scan_csv(found_file, **opts).sink_parquet(parquet_file)
    except Exception as e:
        if not found_file.endswith(".gz"):
            raise
        # Older polars versions cannot stream .gz files: decompress to a temporary CSV first.
        print(f"  streaming conversion failed ({type(e).__name__}); decompressing first")
        tmp = f"_tmp_{table_name}.csv"
        with gzip.open(found_file, "rb") as fi, open(tmp, "wb") as fo:
            shutil.copyfileobj(fi, fo, length=16 * 1024 * 1024)
        pl.scan_csv(tmp, **opts).sink_parquet(parquet_file)
        os.remove(tmp)
    return parquet_file
