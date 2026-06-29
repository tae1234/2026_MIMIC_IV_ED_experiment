"""
================================================================================
step04_create_ed_vitals_summary.py [Precise Timepoints Version]
================================================================================
Purpose:
    1. Loads the vitals_raw data based on precise data entry timepoints (up to ED discharge).
    2. Extracts one row of physiological summary metrics (min, max, first) per patient based on stay_id.
    3. Dynamically generates clean column names for downstream modeling.
================================================================================
"""

import polars as pl


def create_ed_vitals_summary_precise():
    input_file = "ed_vitals_raw_precise.parquet"
    output_file = "ed_vitals_summary_precise.parquet"

    print(f"\n▶ Executing data sorting and aggregation for precise timepoints...")

    # 1. Ultra-fast aggregation using Polars Native API (superior to SQL for time-series)
    # - Sorting by stay_id and charttime in ascending order ensures first() always yields the 'initial measurement'
    lf_summary = (
        pl.scan_parquet(input_file)
        .sort(["stay_id", "charttime"])
        .group_by("stay_id")
        .agg([
            # Minimums (Shock/Crisis detection)
            pl.col("sbp").min().alias("sbp_min"),
            pl.col("dbp").min().alias("dbp_min"),
            pl.col("o2sat").min().alias("spo2_min"),

            # Maximums (Tachycardia/Tachypnea detection)
            pl.col("heartrate").max().alias("hr_max"),
            pl.col("resprate").max().alias("rr_max"),
            pl.col("temperature").max().alias("temp_max"),

            # First measurements upon arrival (drop_nulls safely ignores missing early values)
            pl.col("sbp").drop_nulls().first().alias("sbp_first"),
            pl.col("heartrate").drop_nulls().first().alias("hr_first"),
            pl.col("o2sat").drop_nulls().first().alias("spo2_first")
        ])
    )

    print(f"💾 Saving to '{output_file}'...")
    lf_summary.sink_parquet(output_file)

    # Verification (Ultra-fast metadata scan)
    total_rows = pl.scan_parquet(output_file).select(pl.len()).collect().item()
    print(f"✅ Complete! Unique patients (stay_id): {total_rows:,}")

    print("\n🎉 Vitals Summary data generation based on precise entry timepoints is complete!")

    # Print a sample of the results
    print("\n📊 [Result Preview]")
    print(pl.scan_parquet(output_file).head(5).collect())


if __name__ == "__main__":
    create_ed_vitals_summary_precise()