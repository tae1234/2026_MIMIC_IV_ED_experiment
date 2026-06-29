"""
================================================================================
create_master_dataset.py
Description: Generates the final master dataset for ML models by pivoting lab 
             records, generating measurement flags, and joining with the base cohort.
================================================================================
"""

import os
import polars as pl

def generate_master_dataset():
    print("Processing lab pivoting and merging master dataset...")

    input_lab_file = "lab_raw_continuous.parquet"
    if not os.path.exists(input_lab_file):
        print(f"Error: {input_lab_file} does not exist. Please run step06 first.")
        return

    lf_lab_raw = pl.scan_parquet(input_lab_file).with_columns([
        pl.col("stay_id").cast(pl.Int64, strict=False)
    ])

    ctx = pl.SQLContext()
    ctx.register("lab_raw", lf_lab_raw)

    # Pivot lab events and generate measurement indicator flags
    sql_lab_pivot = """
        WITH RankedLabs AS (
            SELECT stay_id, 
                   lab_category, 
                   valuenum, 
                   ROW_NUMBER() OVER(PARTITION BY stay_id, lab_category ORDER BY charttime) AS rn
            FROM lab_raw
        ),
        PivotedLabs AS (
            SELECT stay_id, 
                   MAX(CASE WHEN lab_category = 'WBC' THEN valuenum END)         AS wbc, 
                   MAX(CASE WHEN lab_category = 'Hemoglobin' THEN valuenum END)  AS hb, 
                   MAX(CASE WHEN lab_category = 'Platelet' THEN valuenum END)    AS plt, 
                   MAX(CASE WHEN lab_category = 'Creatinine' THEN valuenum END)  AS cr, 
                   MAX(CASE WHEN lab_category = 'Sodium' THEN valuenum END)      AS na, 
                   MAX(CASE WHEN lab_category = 'Potassium' THEN valuenum END)   AS k, 
                   MAX(CASE WHEN lab_category = 'Bicarbonate' THEN valuenum END) AS hco3, 
                   MAX(CASE WHEN lab_category = 'Lactate' THEN valuenum END)     AS lactate, 
                   MAX(CASE WHEN lab_category = 'Glucose' THEN valuenum END)     AS glucose
            FROM RankedLabs
            WHERE rn = 1
            GROUP BY stay_id
        )
        SELECT stay_id, 
               wbc, hb, plt, cr, na, k, hco3, lactate, glucose, 
               CAST(CASE WHEN wbc IS NOT NULL THEN 1 ELSE 0 END AS TINYINT)     AS wbc_measured, 
               CAST(CASE WHEN hb IS NOT NULL THEN 1 ELSE 0 END AS TINYINT)      AS hb_measured, 
               CAST(CASE WHEN plt IS NOT NULL THEN 1 ELSE 0 END AS TINYINT)     AS plt_measured, 
               CAST(CASE WHEN cr IS NOT NULL THEN 1 ELSE 0 END AS TINYINT)      AS cr_measured, 
               CAST(CASE WHEN lactate IS NOT NULL THEN 1 ELSE 0 END AS TINYINT) AS lactate_measured
        FROM PivotedLabs 
    """

    lf_lab_pivoted = ctx.execute(sql_lab_pivot)

    input_minimal_file = "ed_minimal_model.parquet"
    lf_minimal = pl.scan_parquet(input_minimal_file)

    # Left join pivoted lab data (maintains null values for patients without lab tests)
    lf_master = lf_minimal.join(lf_lab_pivoted, on="stay_id", how="left")

    output_file = "ed_master_dataset.parquet"
    print(f"Saving master dataset to '{output_file}'...")
    lf_master.sink_parquet(output_file)

    total_rows = pl.scan_parquet(output_file).select(pl.len()).collect().item()
    print(f"Master dataset generation complete. Total records: {total_rows:,}")

    # Display preview of the master dataset
    print("Verifying data structure (Preview):")
    df_check = pl.scan_parquet(output_file).head(3).select([
        "stay_id", "triage_acuity", "cc_system", "heartrate_max", "lactate", "lactate_measured",
        "target_admit_or_expire"
    ]).collect()
    print(df_check)

if __name__ == "__main__":
    generate_master_dataset()