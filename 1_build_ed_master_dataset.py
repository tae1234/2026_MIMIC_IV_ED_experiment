"""
================================================================================
build_ed_master_dataset.py
Description: Full pipeline for ED cohort extraction, vital/lab aggregation,
             composite outcome generation, and model master table building.

             *** TIME-POINT AWARE VERSION ***
             Vitals/labs are aggregated separately for elapsed-time windows of
             0.5h, 1h, and 2h measured from ED arrival (intime), producing one
             master table per time point:
                 ed_master_dataset_0.5h.parquet
                 ed_master_dataset_1h.parquet
                 ed_master_dataset_2h.parquet
             so that downstream scripts (3_, 4_) can compare ML algorithms
             (LR / RF / XGB / ResMLP) across subgroups and time points.
================================================================================
"""

import os
import glob
import polars as pl

MIMIC_DIRS = ["mimic/ed", "mimic/hosp", "mimic/icu", "mimic"]

# Time points (hours elapsed from ED intime) to build feature snapshots for.
TIME_WINDOWS = {"0.5h": 0.5, "1h": 1.0, "2h": 2.0}

# Lower bound grace period (hours) before intime, to capture pre-registration /
# EMS labs that clinically belong to the arrival episode. Set to 0.0 to forbid
# any pre-arrival data. Kept small so it does not contaminate the 0.5h snapshot.
PRE_ARRIVAL_GRACE_H = 0.25  # 15 minutes


def ensure_parquet(table_name: str, search_dirs: list) -> str:
    parquet_file = f"{table_name}.parquet"
    if os.path.exists(parquet_file):
        return parquet_file

    print(f"⚠️ '{parquet_file}' not found. Searching for original file...")
    found_file = next(
        (matches[0] for d in search_dirs if
         (matches := glob.glob(os.path.join(d, "**", f"{table_name}.csv*"), recursive=True))),
        None
    )
    if not found_file:
        raise FileNotFoundError(f"❌ Missing source file for '{table_name}' (.csv or .csv.gz)")

    print(f"⏳ Converting '{table_name}' to high-speed Parquet format (Low-Memory Streaming Mode)...")

    try:
        pl.scan_csv(found_file, infer_schema_length=10000, null_values=["", "NA", "NaN"]).sink_parquet(parquet_file)
    except Exception as e:
        print(f"  -> Streaming failed ({e}). Falling back to Batched processing...")
        reader = pl.read_csv_batched(found_file, infer_schema_length=10000, null_values=["", "NA", "NaN"])
        batches = reader.next_batches(100)
        pl.concat(batches).write_parquet(parquet_file)

    return parquet_file


def build_base_and_labels(ctx):
    """Step 1 & 2: cohort + labels. Time-point independent, runs once."""
    print("--- [Step 1: Build Base Cohort (All Patients Included)] ---")
    file_edstays = ensure_parquet("edstays", MIMIC_DIRS)
    file_patients = ensure_parquet("patients", MIMIC_DIRS)
    file_triage = "ed_triage_with_topics_v5_fixed.parquet"

    lf_ed = pl.scan_parquet(file_edstays).select(
        ["subject_id", "hadm_id", "stay_id", "intime", "outtime", "disposition", "gender", "race", "arrival_transport"])
    lf_patients = pl.scan_parquet(file_patients).select(["subject_id", "anchor_age"])
    lf_triage = pl.scan_parquet(file_triage).select(["subject_id", "stay_id", "cc_system", "chiefcomplaint_proc"]).filter(
        pl.col("cc_system").is_in([
            'GI/Abdominal', 'Cardiac', 'Neurological', 'Trauma/Injury', 'Respiratory',
            'Musculoskeletal', 'Psychiatric/MH', 'Infectious', 'Genitourinary',
            'Substance/Tox', 'Dermatological', 'Endocrine/Metabolic', 'Hematology/Oncology'
        ])
    )

    ctx.register("edstays", lf_ed)
    ctx.register("patients", lf_patients)
    ctx.register("triage", lf_triage)

    sql_base = """
        SELECT e.subject_id, e.hadm_id, e.stay_id,
               e.intime AS ed_intime, e.outtime AS ed_outtime,
               e.disposition, e.gender, e.race, e.arrival_transport,
               p.anchor_age, t.cc_system, t.chiefcomplaint_proc
        FROM edstays AS e
        INNER JOIN patients AS p ON e.subject_id = p.subject_id
        INNER JOIN triage AS t ON e.subject_id = t.subject_id AND e.stay_id = t.stay_id
    """
    ctx.execute(sql_base).sink_parquet("ed_base_cohort_with_cc.parquet")
    print("  ✅ Base cohort built.")

    print("--- [Step 2: Create Labels (Admit OR Expire Definition)] ---")
    file_adm = ensure_parquet("admissions", MIMIC_DIRS)
    ctx.register("base", pl.scan_parquet("ed_base_cohort_with_cc.parquet"))
    ctx.register("admissions", pl.scan_parquet(file_adm).select(["hadm_id", "hospital_expire_flag"]))

    sql_labels = """
        SELECT b.*,
               CASE
                   WHEN b.hadm_id IS NOT NULL THEN 1
                   WHEN COALESCE(a.hospital_expire_flag, 0) = 1 THEN 1
                   WHEN b.disposition = 'EXPIRED' THEN 1
                   ELSE 0
               END AS target_admit_or_expire
        FROM base AS b LEFT JOIN admissions AS a ON b.hadm_id = a.hadm_id
    """
    ctx.execute(sql_labels).sink_parquet("ed_cohort_with_labels.parquet")
    lbl_dist = pl.scan_parquet("ed_cohort_with_labels.parquet").select(["target_admit_or_expire"]).collect()
    print(f"  ✅ Labels created (Severe: {lbl_dist['target_admit_or_expire'].sum():,}).")


def build_timepoint_master(ctx, win_label: str, win_hours: float):
    """Steps 3-7 for a single elapsed-time window from intime.

    Builds vitals summary and first-within-window labs restricted to
    [intime - grace, intime + win_hours], then merges into a per-timepoint
    master table.
    """
    print(f"\n########## Building master for time point: {win_label} "
          f"(intime .. intime+{win_hours}h) ##########")

    # Polars' INTERVAL parser only accepts integer + unit (no decimals like
    # '0.5 hour'), so express every window in whole minutes.
    lower = f"INTERVAL '{int(round(PRE_ARRIVAL_GRACE_H * 60))} minute'"
    upper = f"INTERVAL '{int(round(win_hours * 60))} minute'"

    vitals_out = f"ed_vitals_summary_{win_label}.parquet"
    minimal_out = f"ed_minimal_model_{win_label}.parquet"
    lab_raw_out = f"lab_raw_continuous_{win_label}.parquet"
    master_out = f"ed_master_dataset_{win_label}.parquet"

    # ---- Step 3 & 4: Vitals within window ----
    file_v = ensure_parquet("vitalsign", MIMIC_DIRS)

    ctx.register("ed_base_cohort", pl.scan_parquet("ed_cohort_with_labels.parquet").select(
        ["stay_id", "ed_intime", "ed_outtime"]
    ).with_columns([
        pl.col("ed_intime").str.to_datetime(strict=False),
        pl.col("ed_outtime").str.to_datetime(strict=False)
    ]))

    ctx.register("vitalsign", pl.scan_parquet(file_v).select(
        ["stay_id", "charttime", "sbp", "dbp", "heartrate", "resprate", "o2sat", "temperature"]
    ).with_columns([
        pl.col("charttime").str.to_datetime(strict=False),
        pl.col("sbp").cast(pl.Float64, strict=False),
        pl.col("dbp").cast(pl.Float64, strict=False),
        pl.col("heartrate").cast(pl.Float64, strict=False),
        pl.col("resprate").cast(pl.Float64, strict=False),
        pl.col("o2sat").cast(pl.Float64, strict=False),
        pl.col("temperature").cast(pl.Float64, strict=False)
    ]))

    sql_v_raw = f"""
        SELECT v.stay_id, v.charttime, v.sbp, v.dbp, v.heartrate, v.resprate, v.o2sat, v.temperature
        FROM vitalsign AS v INNER JOIN ed_base_cohort AS b ON v.stay_id = b.stay_id
        WHERE v.charttime >= b.ed_intime - {lower}
          AND v.charttime <= b.ed_intime + {upper}
    """
    lf_v_precise = ctx.execute(sql_v_raw)

    (
        lf_v_precise.sort(["stay_id", "charttime"]).group_by("stay_id").agg([
            pl.col("sbp").min().alias("sbp_min"),
            pl.col("sbp").max().alias("sbp_max"),
            pl.col("heartrate").min().alias("heartrate_min"),
            pl.col("heartrate").max().alias("heartrate_max"),
            pl.col("resprate").max().alias("resprate_max"),
            pl.col("o2sat").min().alias("o2sat_min"),
            pl.col("temperature").max().alias("temperature_max")
        ]).sink_parquet(vitals_out)
    )
    print(f"  ✅ Vitals summarized -> {vitals_out}")

    # ---- Step 5: Minimal model integration ----
    ctx.register("base", pl.scan_parquet("ed_cohort_with_labels.parquet"))
    ctx.register("vitals", pl.scan_parquet(vitals_out))
    ctx.register("t_orig", pl.scan_parquet(ensure_parquet("triage", MIMIC_DIRS)).select(["stay_id", "acuity", "pain"]))
    ctx.register("t_topic", pl.scan_parquet("ed_triage_with_topics_v5_fixed.parquet").select(
        ["stay_id", "cc_topic", "cc_keywords"]))

    sql_min = """
        SELECT
            b.*,
            v.sbp_min, v.sbp_max, v.heartrate_min, v.heartrate_max, v.resprate_max, v.o2sat_min, v.temperature_max,
            t_orig.acuity AS triage_acuity,
            t_orig.pain AS triage_pain,
            t_topic.cc_topic,
            t_topic.cc_keywords
        FROM base AS b
        LEFT JOIN vitals AS v ON b.stay_id = v.stay_id
        LEFT JOIN t_orig ON b.stay_id = t_orig.stay_id
        LEFT JOIN t_topic ON b.stay_id = t_topic.stay_id
    """
    ctx.execute(sql_min).sink_parquet(minimal_out)
    print(f"  ✅ Minimal model skeleton merged -> {minimal_out}")

    # ---- Step 6 & 7: Labs within window, first value per category ----
    ctx.register("base_cohort", pl.scan_parquet("ed_cohort_with_labels.parquet").select(
        ["subject_id", "stay_id", "ed_intime"]
    ).with_columns([
        pl.col("ed_intime").str.to_datetime(strict=False)
    ]))

    ctx.register("labevents", pl.scan_parquet(ensure_parquet("labevents", MIMIC_DIRS)).select(
        ["subject_id", "itemid", "charttime", "valuenum", "valueuom"]
    ).with_columns([
        pl.col("charttime").str.to_datetime(strict=False),
        pl.col("valuenum").cast(pl.Float64, strict=False)
    ]))

    ctx.register("d_labitems", pl.scan_parquet(ensure_parquet("d_labitems", MIMIC_DIRS)).select(["itemid", "label"]))

    sql_lab = f"""
        SELECT b.stay_id, b.subject_id, l.itemid, d.label AS lab_name, l.charttime, l.valuenum, l.valueuom,
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
        INNER JOIN d_labitems AS d ON l.itemid = d.itemid
        WHERE l.charttime >= b.ed_intime - {lower}
          AND l.charttime <= b.ed_intime + {upper}
          AND l.valuenum IS NOT NULL
    """
    ctx.execute(sql_lab).filter(pl.col("lab_category") != "Other").sink_parquet(lab_raw_out)

    ctx.register("lab_raw", pl.scan_parquet(lab_raw_out))
    sql_pivot = """
        WITH RankedLabs AS (
            SELECT stay_id, lab_category, valuenum,
                   ROW_NUMBER() OVER(PARTITION BY stay_id, lab_category ORDER BY charttime) AS rn
            FROM lab_raw
        ), PivotedLabs AS (
            SELECT stay_id,
                   MAX(CASE WHEN lab_category = 'WBC' THEN valuenum END) AS wbc,
                   MAX(CASE WHEN lab_category = 'Hemoglobin' THEN valuenum END) AS hb,
                   MAX(CASE WHEN lab_category = 'Platelet' THEN valuenum END) AS plt,
                   MAX(CASE WHEN lab_category = 'Creatinine' THEN valuenum END) AS cr,
                   MAX(CASE WHEN lab_category = 'Sodium' THEN valuenum END) AS na,
                   MAX(CASE WHEN lab_category = 'Potassium' THEN valuenum END) AS k,
                   MAX(CASE WHEN lab_category = 'Bicarbonate' THEN valuenum END) AS hco3,
                   MAX(CASE WHEN lab_category = 'Lactate' THEN valuenum END) AS lactate,
                   MAX(CASE WHEN lab_category = 'Glucose' THEN valuenum END) AS glucose
            FROM RankedLabs WHERE rn = 1 GROUP BY stay_id
        )
        SELECT stay_id, wbc, hb, plt, cr, na, k, hco3, lactate, glucose,
               CAST(CASE WHEN wbc IS NOT NULL THEN 1 ELSE 0 END AS TINYINT) AS wbc_measured,
               CAST(CASE WHEN hb IS NOT NULL THEN 1 ELSE 0 END AS TINYINT) AS hb_measured,
               CAST(CASE WHEN plt IS NOT NULL THEN 1 ELSE 0 END AS TINYINT) AS plt_measured,
               CAST(CASE WHEN cr IS NOT NULL THEN 1 ELSE 0 END AS TINYINT) AS cr_measured,
               CAST(CASE WHEN lactate IS NOT NULL THEN 1 ELSE 0 END AS TINYINT) AS lactate_measured
        FROM PivotedLabs
    """
    lf_pivoted = ctx.execute(sql_pivot)
    pl.scan_parquet(minimal_out).join(lf_pivoted, on="stay_id", how="left").sink_parquet(master_out)
    print(f"  ✅ Master dataset for {win_label} -> {master_out}")
    return master_out


def build_outcomes(ctx):
    """Step 8 & 9: composite outcome. Time-point independent, runs once."""
    print("\n--- [Step 8 & 9: Create Outcome Components & Composite Target Label] ---")

    ctx.register("base_c", pl.scan_parquet("ed_cohort_with_labels.parquet").select(
        ["stay_id", "hadm_id", "ed_intime"]
    ).with_columns([
        pl.col("ed_intime").str.to_datetime(strict=False)
    ]))

    ctx.register("icustays", pl.scan_parquet(ensure_parquet("icustays", MIMIC_DIRS)).select(
        ["hadm_id", "intime"]
    ).rename({"intime": "icu_intime"}).with_columns([
        pl.col("icu_intime").str.to_datetime(strict=False)
    ]))

    ctx.register("admissions", pl.scan_parquet(ensure_parquet("admissions", MIMIC_DIRS)).select(
        ["hadm_id", "hospital_expire_flag"]
    ).with_columns([
        pl.col("hospital_expire_flag").cast(pl.Int64, strict=False)
    ]))

    sql_oc = """
        WITH IcuEvents AS (
            SELECT b.stay_id, MAX(CASE WHEN i.icu_intime >= b.ed_intime AND i.icu_intime <= b.ed_intime + INTERVAL '24 hour' THEN 1 ELSE 0 END) AS icu_24h
            FROM base_c AS b LEFT JOIN icustays AS i ON b.hadm_id = i.hadm_id GROUP BY b.stay_id
        ), MortalityEvents AS (
            SELECT b.stay_id, MAX(CASE WHEN a.hospital_expire_flag = 1 THEN 1 ELSE 0 END) AS death_hosp
            FROM base_c AS b LEFT JOIN admissions AS a ON b.hadm_id = a.hadm_id GROUP BY b.stay_id
        )
        SELECT b.stay_id, b.hadm_id, COALESCE(icu.icu_24h, 0) AS outcome_icu_24h, COALESCE(mort.death_hosp, 0) AS outcome_mortality
        FROM base_c AS b LEFT JOIN IcuEvents AS icu ON b.stay_id = icu.stay_id LEFT JOIN MortalityEvents AS mort ON b.stay_id = mort.stay_id
    """
    ctx.execute(sql_oc).sink_parquet("outcome_components.parquet")

    ctx.register("outcomes", pl.scan_parquet("outcome_components.parquet"))
    sql_composite = """
        SELECT stay_id, hadm_id, outcome_icu_24h AS icu_24h, 0 AS vent_24h, 0 AS pressor_24h, outcome_mortality AS death_hosp,
               CASE WHEN outcome_icu_24h = 1 OR outcome_mortality = 1 THEN 1 ELSE 0 END AS early_critical_illness
        FROM outcomes
    """
    ctx.execute(sql_composite).sink_parquet("outcome_critical_illness.parquet")
    y_dist = pl.scan_parquet("outcome_critical_illness.parquet").select(["early_critical_illness"]).collect()
    print(f"  ✅ Composite targets computed. Critical illness positive rate: "
          f"{y_dist['early_critical_illness'].sum() / len(y_dist)*100:.1f}%")


def run_data_pipeline():
    ctx = pl.SQLContext()

    # Time-point independent stages run once.
    build_base_and_labels(ctx)
    build_outcomes(ctx)

    # Per-time-point feature snapshots.
    for win_label, win_hours in TIME_WINDOWS.items():
        build_timepoint_master(ctx, win_label, win_hours)

    print("\n🎉 All data preparation tasks successfully executed!")
    print("   Per-timepoint masters: " +
          ", ".join(f"ed_master_dataset_{w}.parquet" for w in TIME_WINDOWS))


if __name__ == "__main__":
    run_data_pipeline()