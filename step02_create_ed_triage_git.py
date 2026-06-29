"""
================================================================================
step02_create_labels.py [Option B: Define Target as Admit OR Expire]
================================================================================
"""
import polars as pl
import os
import glob

# (The ensure_parquet function is used identically to Step 01)
def ensure_parquet(table_name: str, search_dirs: list) -> str:
    parquet_file = f"{table_name}.parquet"
    if os.path.exists(parquet_file): 
        return parquet_file
    
    found_file = glob.glob(f"{search_dirs[0]}/**/{table_name}.csv*", recursive=True)[0]
    pl.read_csv(found_file, infer_schema_length=10000, null_values=["", "NA", "NaN"]).write_parquet(parquet_file)
    return parquet_file

def create_labels():
    MIMIC_DIRS = ["mimic/ed", "mimic/hosp", "mimic/icu", "mimic"]
    print("⏳ Scanning data...")

    file_adm = ensure_parquet("admissions", MIMIC_DIRS)
    file_base = "ed_base_cohort_with_cc.parquet"

    lf_base = pl.scan_parquet(file_base)
    lf_adm = pl.scan_parquet(file_adm).with_columns([
        pl.col("hadm_id").cast(pl.Int64, strict=False),
        pl.col("hospital_expire_flag").cast(pl.Int32, strict=False)
    ]).select(["hadm_id", "hospital_expire_flag"])

    ctx = pl.SQLContext()
    ctx.register("base", lf_base)
    ctx.register("admissions", lf_adm)

    # =========================================================
    # [Modified] New target definition logic for Option B
    # 1. Admitted patients (hadm_id IS NOT NULL)
    # 2. Expired (deceased) patients (Death in ED or in-hospital death)
    # =========================================================
    sql_labels = """
        SELECT 
            b.*,
            -- [Create Label] 1 if admitted (hadm_id exists) or expired, 0 if mild case discharged home
            CASE 
                WHEN b.hadm_id IS NOT NULL THEN 1
                WHEN COALESCE(a.hospital_expire_flag, 0) = 1 THEN 1
                WHEN b.disposition = 'EXPIRED' THEN 1
                ELSE 0 
            END AS target_admit_or_expire
            
        FROM base AS b
        LEFT JOIN admissions AS a
            ON b.hadm_id = a.hadm_id
    """
    lf_labeled = ctx.execute(sql_labels)

    output_filename = "ed_cohort_with_labels.parquet"
    lf_labeled.sink_parquet(output_filename)
    print(f"✅ Label creation complete!: {output_filename}\n")

    # Check the resulting distribution
    df_dist = pl.scan_parquet(output_filename).select(["target_admit_or_expire"]).collect()
    total = len(df_dist)
    target_count = df_dist['target_admit_or_expire'].sum()

    print("📊 [Target (Admit or Expire) Distribution Check]")
    print(f" - Total ED visits: {total:,} patients")
    print(f" - Severe (Admit OR Expire)     [Label=1]: {target_count:,} patients ({target_count/total*100:.1f}%)")
    print(f" - Mild (Discharged home, etc.) [Label=0]: {total - target_count:,} patients ({(total - target_count)/total*100:.1f}%)")

if __name__ == "__main__":
    create_labels()