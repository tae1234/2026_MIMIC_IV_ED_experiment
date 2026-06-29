"""
================================================================================
extract_lab_events.py
Description: Extracts core lab events for ED cohort using exact charttimes
================================================================================
"""

import os
import glob
import polars as pl

def ensure_parquet(table_name: str, search_dirs: list) -> str:
    parquet_file = f"{table_name}.parquet"
    if os.path.exists(parquet_file):
        return parquet_file

    found_file = next(
        (matches[0] for d in search_dirs if (matches := glob.glob(os.path.join(d, "**", f"{table_name}.csv*"), recursive=True))), 
        None
    )

    if not found_file:
        raise FileNotFoundError(f"Source file for '{table_name}' (.csv or .csv.gz) not found.")

    print(f"Converting '{found_file}' to Parquet format...")
    pl.read_csv(
        found_file, 
        infer_schema_length=10000, 
        null_values=["", "NA", "NaN"]
    ).write_parquet(parquet_file)
    
    return parquet_file

def extract_lab_events():
    mimic_dirs = ["mimic/ed", "mimic/hosp", "mimic/icu", "mimic"]

    file_labevents = ensure_parquet("labevents", mimic_dirs)
    file_d_labitems = ensure_parquet("d_labitems", mimic_dirs)

    lf_base = pl.scan_parquet("ed_cohort_with_labels.parquet").with_columns([
        pl.col("subject_id").cast(pl.Int64, strict=False),
        pl.col("stay_id").cast(pl.Int64, strict=False),
        pl.col("ed_intime").cast(pl.Datetime, strict=False)
    ])

    lf_labevents = pl.scan_parquet(file_labevents).with_columns([
        pl.col("subject_id").cast(pl.Int64, strict=False),
        pl.col("itemid").cast(pl.Int64, strict=False),
        pl.col("charttime").cast(pl.Datetime, strict=False),
        pl.col("valuenum").cast(pl.Float64, strict=False)
    ])

    lf_d_labitems = pl.scan_parquet(file_d_labitems).with_columns([
        pl.col("itemid").cast(pl.Int64, strict=False)
    ])

    ctx = pl.SQLContext()
    ctx.register("base_cohort", lf_base)
    ctx.register("labevents", lf_labevents)
    ctx.register("d_labitems", lf_d_labitems)

    # Extract continuous lab events based on precise data entry timepoints
    sql_query = """
        SELECT 
            b.stay_id, 
            b.subject_id, 
            l.itemid, 
            d.label AS lab_name, 
            l.charttime, 
            l.valuenum, 
            l.valueuom, 
            CASE 
                WHEN LOWER(d.label) LIKE '%white blood cells%' THEN 'WBC' 
                WHEN LOWER(d.label) LIKE '%hemoglobin%' THEN 'Hemoglobin' 
                WHEN LOWER(d.label) LIKE '%platelet%' THEN 'Platelet' 
                WHEN LOWER(d.label) LIKE '%creatinine%' THEN 'Creatinine' 
                WHEN LOWER(d.label) LIKE '%sodium%' THEN 'Sodium' 
                WHEN LOWER(d.label) LIKE '%potassium%' THEN 'Potassium' 
                WHEN LOWER(d.label) LIKE '%bicarbonate%' THEN 'Bicarbonate' 
                WHEN LOWER(d.label) LIKE '%lactate%' THEN 'Lactate' 
                WHEN LOWER(d.label) LIKE '%glucose%' THEN 'Glucose' 
                ELSE 'Other' 
            END AS lab_category
        FROM labevents AS l
        INNER JOIN base_cohort AS b ON l.subject_id = b.subject_id
        INNER JOIN d_labitems AS d  ON l.itemid = d.itemid
        WHERE 
            l.charttime >= b.ed_intime - INTERVAL '2 hour'
            AND l.valuenum IS NOT NULL 
    """

    print("Executing join and extracting continuous timepoint data...")
    lf_lab_raw = ctx.execute(sql_query)

    # Keep only the 9 core lab categories
    lf_lab_raw_filtered = lf_lab_raw.filter(pl.col("lab_category") != "Other")

    output_filename = "lab_raw_continuous.parquet"
    lf_lab_raw_filtered.sink_parquet(output_filename)

    total_rows = pl.scan_parquet(output_filename).select(pl.len()).collect().item()
    print(f"Extraction complete. Total records: {total_rows:,}")

if __name__ == "__main__":
    extract_lab_events()