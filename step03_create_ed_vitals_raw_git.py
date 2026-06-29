"""
================================================================================
step03_create_ed_vitals_raw.py [Precise Timepoints Version - Base Cohort Modified]
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
        raise FileNotFoundError(f"❌ Could not find original file (.csv/gz) for {table_name}!")

    print(f"⏳ Converting '{table_name}' data to high-speed Parquet format...")
    pl.read_csv(found_file, infer_schema_length=10000, null_values=["", "NA", "NaN"]).write_parquet(parquet_file)
    print(f"✅ Conversion complete: {parquet_file} generated\n")
    return parquet_file

def create_ed_vitals_raw_precise_timepoints():
    MIMIC_DIRS = ["mimic/ed", "mimic/hosp", "mimic/icu", "mimic"]

    print("⏳ Checking and preparing necessary data...")
    file_vitals = ensure_parquet("vitalsign", MIMIC_DIRS)

    print("\n⏳ Scanning data and casting types (Lazy Loading)...")

    # =========================================================
    # [Modified] Using 'ed_cohort_with_labels.parquet' from Step 02
    # =========================================================
    lf_base = pl.scan_parquet("ed_cohort_with_labels.parquet").with_columns([
        pl.col("stay_id").cast(pl.Int64, strict=False),
        pl.col("ed_intime").cast(pl.Datetime, strict=False),
        pl.col("ed_outtime").cast(pl.Datetime, strict=False)
    ])

    lf_vitals = pl.scan_parquet(file_vitals).with_columns([
        pl.col("stay_id").cast(pl.Int64, strict=False),
        pl.col("charttime").cast(pl.Datetime, strict=False),
        pl.col("sbp").cast(pl.Float64, strict=False),
        pl.col("dbp").cast(pl.Float64, strict=False),
        pl.col("heartrate").cast(pl.Float64, strict=False),
        pl.col("resprate").cast(pl.Float64, strict=False),
        pl.col("o2sat").cast(pl.Float64, strict=False),
        pl.col("temperature").cast(pl.Float64, strict=False)
    ])

    ctx = pl.SQLContext()
    ctx.register("ed_base_cohort", lf_base)
    ctx.register("vitalsign", lf_vitals)

    # =========================================================
    # Extracting exact data entry timepoints up to ED outtime 
    # to capture precise longitudinal trends.
    # =========================================================
    sql_vitals_raw = """
        SELECT 
            v.stay_id, 
            v.charttime, 
            v.sbp, 
            v.dbp, 
            v.heartrate, 
            v.resprate, 
            v.o2sat, 
            v.temperature
        FROM vitalsign AS v
        INNER JOIN ed_base_cohort AS b
            ON v.stay_id = b.stay_id
        WHERE v.charttime >= b.ed_intime - INTERVAL '2 hour' 
          AND v.charttime <= b.ed_outtime
    """

    print("\n▶ Executing SQL query optimization and filtering for precise timepoints...")
    lf_vitals_result = ctx.execute(sql_vitals_raw)

    output_filename = "ed_vitals_raw_precise.parquet"
    print(f"💾 Saving to '{output_filename}'...")
    lf_vitals_result.sink_parquet(output_filename)

    total_rows = pl.scan_parquet(output_filename).select(pl.len()).collect().item()
    print(f"✅ Complete! Total records: {total_rows:,}")

    print("\n🎉 Vitals Raw data generation based on precise entry timepoints is complete!")

if __name__ == "__main__":
    create_ed_vitals_raw_precise_timepoints()