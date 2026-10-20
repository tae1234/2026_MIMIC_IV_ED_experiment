"""
00_convert_mimic_to_parquet.py
Converts the MIMIC-IV / MIMIC-IV-ED source tables used by the pipeline from
CSV(.gz) to parquet in WORK_DIR (one-off; existing parquet files are reused).
"""
import paths

TABLES = [
    # MIMIC-IV-ED
    "edstays", "triage", "vitalsign",
    # MIMIC-IV hosp
    "patients", "admissions", "labevents", "d_labitems",
    # MIMIC-IV icu
    "icustays", "procedureevents", "inputevents",
]

if __name__ == "__main__":
    for t in TABLES:
        paths.ensure_parquet(t)
    print(f"Done. Parquet files are in {paths.WORK_DIR}")
