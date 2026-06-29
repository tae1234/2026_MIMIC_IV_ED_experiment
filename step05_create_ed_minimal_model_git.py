"""
================================================================================
step05_create_ed_minimal_model.py [Research Design Integration - Merging T0 Variables]
================================================================================
Purpose:
    Aligning with the research design (Not All Patients Need More Than Triage),
    this script safely merges the core T0 information ('Triage Acuity' and 'Pain')
    and clustering data (cc_topic, cc_keywords) into the foundational dataset 
    completed in Step 04 (Label + Vitals).
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
        raise FileNotFoundError(f"❌ Could not find original file (.csv or .csv.gz) for {table_name}!")

    print(f"🔄 Original file found: {found_file}")
    print(f"⏳ Converting '{table_name}' data to high-speed Parquet format...")
    pl.read_csv(found_file, infer_schema_length=10000, null_values=["", "NA", "NaN"]).write_parquet(parquet_file)
    print(f"✅ Conversion complete: {parquet_file} generated\n")
    return parquet_file

def create_ed_minimal_model_precise():
    MIMIC_DIRS = ["mimic/ed", "mimic/hosp", "mimic/icu", "mimic"]

    print("⏳ Scanning and preparing common tables (Lazy Loading)...")

    # 1. Original triage.parquet (Extracting core T0 variables: acuity, pain)
    file_triage_orig = ensure_parquet("triage", MIMIC_DIRS)
    lf_triage_orig = pl.scan_parquet(file_triage_orig).with_columns([
        pl.col("stay_id").cast(pl.Int64, strict=False),
        pl.col("acuity").cast(pl.Float64, strict=False),
        pl.col("pain").cast(pl.Utf8, strict=False)
    ]).select(["stay_id", "acuity", "pain"])

    # 2. Triage Topics source (Extracting AI clustering results)
    lf_triage_topics = pl.scan_parquet("ed_triage_with_topics_v5_fixed.parquet").with_columns(
        pl.col("stay_id").cast(pl.Int64, strict=False)
    ).select(["stay_id", "cc_topic", "cc_keywords"])

    print(f"\n▶ Executing Master Table Join...")

    # 3. [Core Modification] Loading the complete skeleton (Base + Label + Vitals) finished in Step 04
    # Note: Assuming the merged file is named ed_final_dataset_precise.parquet as per the precise timepoints methodology.
    lf_final_dataset = pl.scan_parquet("ed_final_dataset_precise.parquet")

    ctx = pl.SQLContext()
    ctx.register("final_dataset", lf_final_dataset)
    ctx.register("triage_orig", lf_triage_orig)
    ctx.register("triage_topics", lf_triage_topics)

    # 4. [Optimized SQL] Retain all columns from final_dataset (f) and LEFT JOIN only the T0 variables
    sql_minimal_model = """
        SELECT 
            f.*,
            t_orig.acuity AS triage_acuity,
            t_orig.pain AS triage_pain,
            t_topic.cc_topic,
            t_topic.cc_keywords
        FROM final_dataset AS f
        LEFT JOIN triage_orig AS t_orig 
            ON f.stay_id = t_orig.stay_id
        LEFT JOIN triage_topics AS t_topic 
            ON f.stay_id = t_topic.stay_id
    """

    output_filename = "ed_minimal_model_precise.parquet"
    lf_minimal_model = ctx.execute(sql_minimal_model)
    lf_minimal_model.sink_parquet(output_filename)

    total_rows = pl.scan_parquet(output_filename).select(pl.len()).collect().item()
    print(f"💾 Saved '{output_filename}'! (Total: {total_rows:,} records)")

    print("\n🎉 The Minimal Model master table based on precise timepoints has been successfully generated!")

    # Verification: Check if the columns were properly appended
    print("\n📊 [Result Verification: Data Preview]")
    df_check = pl.scan_parquet(output_filename).head(3).select([
        "stay_id", "anchor_age", "triage_acuity", "cc_system", "target_admit_or_expire"
    ]).collect()
    print(df_check)

if __name__ == "__main__":
    create_ed_minimal_model_precise()