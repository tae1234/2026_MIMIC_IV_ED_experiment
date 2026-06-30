"""
================================================================================
finalize_analysis_data.py
Description: Integrates composite outcomes, handles physiological outliers by
             clipping extreme values (Lactate, Heart Rate), drops legacy target
             labels, and generates final analysis-ready Parquet datasets.

             *** TIME-POINT AWARE VERSION ***
             Runs once per elapsed-time window (0.5h, 1h, 2h), turning each
                 ed_master_dataset_{win}.parquet
             into
                 analysis_master_{win}.parquet
             which is exactly what 4_run_all_experiments.py loads via
             load_and_split_data(window=...).
================================================================================
"""

import os
import polars as pl

# Must match TIME_WINDOWS in 1_build_ed_master_dataset.py
TIME_WINDOWS = ["0.5h", "1h", "2h"]

OUTCOME_FILE = "outcome_critical_illness.parquet"


def finalize_one(win_label: str):
    input_master = f"ed_master_dataset_{win_label}.parquet"
    output_file = f"analysis_master_{win_label}.parquet"

    if not os.path.exists(input_master):
        raise FileNotFoundError(
            f"❌ '{input_master}' not found. Run 1_build_ed_master_dataset.py first."
        )

    print(f"\n########## Finalizing time point: {win_label} ##########")
    print("Loading composite outcome targets...")
    lf_outcome = pl.scan_parquet(OUTCOME_FILE).with_columns(
        pl.col("stay_id").cast(pl.Int64, strict=False)
    ).select(["stay_id", "early_critical_illness", "icu_24h", "death_hosp"])

    print(f"Integrating master table and cleaning data ('{input_master}')...")
    lf_ext = pl.scan_parquet(input_master).with_columns(
        pl.col("stay_id").cast(pl.Int64, strict=False)
    )

    # Drop legacy label if it exists (use collect_schema to avoid PerformanceWarning)
    if "target_admit_or_expire" in lf_ext.collect_schema().names():
        lf_ext = lf_ext.drop("target_admit_or_expire")

    lf_master = lf_ext.join(lf_outcome, on="stay_id", how="inner")

    # Clip physiological outliers to clinically sensible maximums
    lf_master = lf_master.with_columns([
        pl.when(pl.col("lactate") > 30.0).then(30.0).otherwise(pl.col("lactate")).alias("lactate"),
        pl.when(pl.col("heartrate_max") > 300.0).then(300.0).otherwise(pl.col("heartrate_max")).alias("heartrate_max"),
    ])

    # Remove duplicate joined columns and sort
    lf_master = lf_master.select(pl.all().exclude("^.*_right$")).sort("stay_id")

    print(f"Saving cleaned dataset to '{output_file}'...")
    lf_master.sink_parquet(output_file)

    total_rows = pl.scan_parquet(output_file).select(pl.len()).collect().item()
    print(f"  -> Complete. Total analysis records: {total_rows:,}")
    return output_file


def finalize_datasets():
    outputs = [finalize_one(w) for w in TIME_WINDOWS]
    print("\nAnalysis master datasets successfully generated:")
    for o in outputs:
        print(f"  - {o}")


if __name__ == "__main__":
    finalize_datasets()