"""
================================================================================
step01_create_ed_base_cohort.py [Option B: Version Including All Patients]
================================================================================
"""
import polars as pl
import os
import glob


def ensure_parquet(table_name: str, search_dirs: list) -> str:
    parquet_file = f"{table_name}.parquet"
    if os.path.exists(parquet_file):
        return parquet_file
    print(f"⚠️ '{parquet_file}' not found. Searching for the original file...")
    found_file = None
    for directory in search_dirs:
        pattern = os.path.join(directory, "**", f"{table_name}.csv*")
        matches = glob.glob(pattern, recursive=True)
        if matches:
            found_file = matches[0]
            break
    if not found_file:
        raise FileNotFoundError(f"❌ Could not find original file for {table_name}!")
    print(f"⏳ Converting '{table_name}' data to high-speed Parquet format...")
    pl.read_csv(found_file, infer_schema_length=10000, null_values=["", "NA", "NaN"]).write_parquet(parquet_file)
    return parquet_file


def create_ed_base_cohort():
    MIMIC_DIRS = ["mimic/ed", "mimic/hosp", "mimic/icu", "mimic"]
    print("⏳ Checking and scanning data...")

    file_edstays = ensure_parquet("edstays", MIMIC_DIRS)
    file_patients = ensure_parquet("patients", MIMIC_DIRS)
    file_triage = "ed_triage_with_topics_v5_fixed.parquet"

    lf_ed = pl.scan_parquet(file_edstays).with_columns([
        pl.col("subject_id").cast(pl.Int64, strict=False),
        pl.col("hadm_id").cast(pl.Int64, strict=False),
        pl.col("stay_id").cast(pl.Int64, strict=False)
    ])

    lf_patients = pl.scan_parquet(file_patients).with_columns([
        pl.col("subject_id").cast(pl.Int64, strict=False),
        pl.col("anchor_age").cast(pl.Int64, strict=False)
    ])

    target_systems = [
        'GI/Abdominal', 'Cardiac', 'Neurological', 'Trauma/Injury',
        'Respiratory', 'Musculoskeletal', 'Psychiatric/MH', 'Infectious',
        'Genitourinary', 'Substance/Tox', 'Dermatological',
        'Endocrine/Metabolic', 'Hematology/Oncology'
    ]
    lf_triage = pl.scan_parquet(file_triage).with_columns([
        pl.col("subject_id").cast(pl.Int64, strict=False),
        pl.col("stay_id").cast(pl.Int64, strict=False)
    ]).filter(pl.col("cc_system").is_in(target_systems))

    ctx = pl.SQLContext()
    ctx.register("edstays", lf_ed)
    ctx.register("patients", lf_patients)
    ctx.register("triage", lf_triage)

    # =========================================================
    # [Modified] Removed the WHERE clause to include all patients, 
    # even those discharged home!
    # =========================================================
    sql_base_cohort = """
        SELECT 
            e.subject_id, e.hadm_id, e.stay_id,
            e.intime AS ed_intime, e.outtime AS ed_outtime,
            e.disposition, e.gender, e.race, e.arrival_transport,
            p.anchor_age,
            t.cc_system, t.chiefcomplaint_proc
        FROM edstays AS e
        INNER JOIN patients AS p ON e.subject_id = p.subject_id
        INNER JOIN triage AS t ON e.subject_id = t.subject_id AND e.stay_id = t.stay_id
    """
    lf_base_cohort = ctx.execute(sql_base_cohort)

    output_filename = "ed_base_cohort_with_cc.parquet"
    lf_base_cohort.sink_parquet(output_filename)
    print(f"✅ Cohort creation complete! (Including discharged patients): {output_filename}\n")


if __name__ == "__main__":
    create_ed_base_cohort()