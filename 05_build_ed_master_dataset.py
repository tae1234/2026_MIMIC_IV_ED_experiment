"""
================================================================================
05_build_ed_master_dataset.py
Description: Full pipeline for ED cohort extraction, vital/lab aggregation,
             composite outcome generation, and model master table building.

             *** TIME-POINT AWARE VERSION ***
             Vitals/labs are aggregated separately for elapsed-time windows of
             0.5h, 1h, and 2h measured from ED arrival (intime), producing one
             master table per time point:
                 ed_master_dataset_0.5h.parquet
                 ed_master_dataset_1h.parquet
                 ed_master_dataset_2h.parquet
             so that downstream scripts (06, 07) can compare ML algorithms
             (LR / RF / XGB / ResMLP) across subgroups and time points.

             Cohort selection is done in two stages (Fig. 1):
               Stage 1 "Cohort selection": adults >= 18 years with a matching
                 ED-stay and demographic record, a valid arrival time,
                 departure time not before arrival, and available triage
                 acuity and pain score.
               Stage 2 "Chief complaint NLP & subgrouping": chief complaint
                 mapped to one of the 13 core CEDIS categories.
================================================================================
"""

import os
import glob
import polars as pl

import paths  # makes WORK_DIR the working directory

MIMIC_DIRS = paths.MIMIC_DIRS  # set via the MIMIC_DIR environment variable (see paths.py)

# Time points (hours elapsed from ED intime) to build feature snapshots for.
TIME_WINDOWS = {"0.5h": 0.5, "1h": 1.0, "2h": 2.0}

# Lower bound grace period (hours) before intime, to capture pre-registration /
# EMS labs that clinically belong to the arrival episode. Set to 0.0 to forbid
# any pre-arrival data. Kept small so it does not contaminate the 0.5h snapshot.
PRE_ARRIVAL_GRACE_H = 0.25  # 15 minutes

# The 13 CEDIS "core" clinical subgroups the analytic cohort is restricted to
# (must match Table 4 / cc_system values produced by 04_reassign_cedis.py).
CORE_CEDIS_CATEGORIES = [
    'GI/Abdominal', 'Cardiac', 'Neurological', 'Trauma/Injury', 'Respiratory',
    'Musculoskeletal', 'Psychiatric/MH', 'Infectious', 'Genitourinary',
    'Substance/Tox', 'Dermatological', 'Endocrine/Metabolic', 'Hematology/Oncology'
]

# Figure 1 Box 2 "Cohort selection" -- minimum age to be included as an adult.
MIN_ADULT_AGE = 18


def ensure_parquet(table_name: str, search_dirs: list) -> str:
    """CSV(.gz) -> parquet conversion (shared helper in paths.py)."""
    return paths.ensure_parquet(table_name, search_dirs)


def build_base_and_labels(ctx):
    """Step 1 & 2: cohort + labels. Time-point independent, runs once.

    Runs in two passes:
      Pass A (diagnostic): a LEFT-JOIN of the full triage-with-topics universe
        against edstays, patients, and the ORIGINAL triage table (for
        acuity/pain), with one flag column per exclusion reason, applied in
        priority order so the Figure-1 Box-2 / Box-3 counts can be printed.
      Pass B (persisted result): a single INNER JOIN + WHERE combining every
        condition from Pass A -- produces the exact same row set as Pass A's
        final filter, just without the intermediate reason bookkeeping.
    """
    print("--- [Step 1: Build Base Cohort — two-stage exclusion accounting] ---")
    file_edstays = ensure_parquet("edstays", MIMIC_DIRS)
    file_patients = ensure_parquet("patients", MIMIC_DIRS)
    file_triage_topics = "ed_triage_with_topics_v5_fixed.parquet"
    file_triage_orig = ensure_parquet("triage", MIMIC_DIRS)

    core_list_sql = ", ".join(f"'{c}'" for c in CORE_CEDIS_CATEGORIES)

    lf_ed = pl.scan_parquet(file_edstays).select(
        ["subject_id", "hadm_id", "stay_id", "intime", "outtime", "disposition", "gender", "race", "arrival_transport"])
    lf_patients = pl.scan_parquet(file_patients).select(["subject_id", "anchor_age"])
    lf_triage_all = pl.scan_parquet(file_triage_topics).select(
        ["subject_id", "stay_id", "cc_system", "chiefcomplaint_proc"])
    lf_triage_orig = pl.scan_parquet(file_triage_orig).select(["stay_id", "acuity", "pain"])

    n_universe = lf_triage_all.select(pl.len()).collect().item()

    ctx.register("edstays", lf_ed)
    ctx.register("patients", lf_patients)
    ctx.register("triage_all", lf_triage_all)
    ctx.register("triage_orig", lf_triage_orig)

    # ---- Pass A: diagnostic LEFT JOIN with one flag per exclusion reason ----
    sql_flags = f"""
        SELECT
            t.subject_id, t.stay_id, t.cc_system,
            e.intime AS ed_intime, e.outtime AS ed_outtime,
            p.anchor_age, tr.acuity, tr.pain,
            CASE WHEN e.stay_id IS NULL THEN 1 ELSE 0 END AS flag_no_edstay_match,
            CASE WHEN e.stay_id IS NOT NULL AND e.intime IS NULL
                 THEN 1 ELSE 0 END AS flag_missing_intime,
            CASE WHEN e.stay_id IS NOT NULL AND e.intime IS NOT NULL
                      AND e.outtime IS NOT NULL AND e.outtime < e.intime
                 THEN 1 ELSE 0 END AS flag_time_inconsistent,
            CASE WHEN p.subject_id IS NULL THEN 1 ELSE 0 END AS flag_no_patient_match,
            CASE WHEN p.subject_id IS NOT NULL AND p.anchor_age < {MIN_ADULT_AGE}
                 THEN 1 ELSE 0 END AS flag_not_adult,
            CASE WHEN tr.acuity IS NULL OR tr.pain IS NULL
                 THEN 1 ELSE 0 END AS flag_missing_key_vars,
            CASE WHEN t.cc_system IS NULL OR t.cc_system NOT IN ({core_list_sql})
                 THEN 1 ELSE 0 END AS flag_cc_not_core
        FROM triage_all AS t
        LEFT JOIN edstays     AS e  ON t.subject_id = e.subject_id AND t.stay_id = e.stay_id
        LEFT JOIN patients    AS p  ON t.subject_id = p.subject_id
        LEFT JOIN triage_orig AS tr ON t.stay_id = tr.stay_id
    """
    ctx.execute(sql_flags).sink_parquet("_exclusion_flags_debug.parquet")

    flags = pl.scan_parquet("_exclusion_flags_debug.parquet").collect()

    # Apply reasons in priority order so a row is only ever attributed to the
    # FIRST reason it trips (avoids double-counting across overlapping flags).
    remaining = flags

    n_no_edstay = remaining.filter(pl.col("flag_no_edstay_match") == 1).height
    remaining = remaining.filter(pl.col("flag_no_edstay_match") == 0)

    n_missing_intime = remaining.filter(pl.col("flag_missing_intime") == 1).height
    remaining = remaining.filter(pl.col("flag_missing_intime") == 0)

    n_time_inconsistent = remaining.filter(pl.col("flag_time_inconsistent") == 1).height
    remaining = remaining.filter(pl.col("flag_time_inconsistent") == 0)

    n_no_patient = remaining.filter(pl.col("flag_no_patient_match") == 1).height
    remaining = remaining.filter(pl.col("flag_no_patient_match") == 0)

    n_not_adult = remaining.filter(pl.col("flag_not_adult") == 1).height
    remaining = remaining.filter(pl.col("flag_not_adult") == 0)

    n_missing_key_vars = remaining.filter(pl.col("flag_missing_key_vars") == 1).height
    remaining = remaining.filter(pl.col("flag_missing_key_vars") == 0)

    n_after_cohort_selection = remaining.height  # == Figure 1 Box 2 N

    n_cc_not_core = remaining.filter(pl.col("flag_cc_not_core") == 1).height
    n_final = remaining.filter(pl.col("flag_cc_not_core") == 0).height  # == Figure 1 Box 3 N

    print(f"\n  [Figure 1 — Box 2 'Cohort selection': exclusion reasons, priority order]")
    print(f"    Universe (raw ED arrivals w/ chief complaint)       : {n_universe:>10,}")
    print(f"    (-) No matching ED-stay record                      : {n_no_edstay:>10,}")
    print(f"    (-) ED-stay found but ed_intime missing/unparseable : {n_missing_intime:>10,}")
    print(f"    (-) ed_outtime < ed_intime (temporal inconsistency) : {n_time_inconsistent:>10,}")
    print(f"    (-) No matching demographic (patients) record       : {n_no_patient:>10,}")
    print(f"    (-) anchor_age < {MIN_ADULT_AGE} (not adult)                     : {n_not_adult:>10,}")
    print(f"    (-) Missing key triage variables (acuity/pain null) : {n_missing_key_vars:>10,}")
    print(f"    {'-'*68}")
    print(f"    = N after cohort selection (Box 2)                  : {n_after_cohort_selection:>10,}")

    print(f"\n  [Figure 1 — Box 3 'Chief complaint NLP & subgrouping': exclusion reasons]")
    print(f"    (-) Chief complaint not mapped to 13 core CEDIS cat.: {n_cc_not_core:>10,}")
    print(f"    {'-'*68}")
    print(f"    = Final analytic cohort (Box 3)                     : {n_final:>10,}")

    total_excluded = (n_no_edstay + n_missing_intime + n_time_inconsistent +
                      n_no_patient + n_not_adult + n_missing_key_vars + n_cc_not_core)
    print(f"\n    (sanity check: universe - all exclusions = {n_universe - total_excluded:,}, "
          f"should equal Box 3 final N above)")
    print(f"    NOTE: this N may differ from the previously-reported 272,258 -- the age/")
    print(f"    timestamp/key-variable filters above are new. Treat this run's N as the")
    print(f"    new source of truth and re-run 3_finalize_analysis_data.py + ")
    print(f"    4_run_all_experiments.py against it.\n")

    os.remove("_exclusion_flags_debug.parquet")

    # ---- Pass B: the actual cohort build, combining every Pass-A condition ----
    lf_triage_core = lf_triage_all.filter(pl.col("cc_system").is_in(CORE_CEDIS_CATEGORIES))
    ctx.register("triage", lf_triage_core)
    ctx.register("triage_orig2", lf_triage_orig)

    sql_base = f"""
        SELECT e.subject_id, e.hadm_id, e.stay_id,
               e.intime AS ed_intime, e.outtime AS ed_outtime,
               e.disposition, e.gender, e.race, e.arrival_transport,
               p.anchor_age, t.cc_system, t.chiefcomplaint_proc
        FROM edstays AS e
        INNER JOIN patients    AS p  ON e.subject_id = p.subject_id
        INNER JOIN triage      AS t  ON e.subject_id = t.subject_id AND e.stay_id = t.stay_id
        INNER JOIN triage_orig2 AS tr ON e.stay_id = tr.stay_id
        WHERE e.intime IS NOT NULL
          AND e.outtime IS NOT NULL AND e.outtime >= e.intime
          AND p.anchor_age >= {MIN_ADULT_AGE}
          AND tr.acuity IS NOT NULL AND tr.pain IS NOT NULL
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


# ---- Outcome component itemids (MIMIC-IV ICU module) ----
# Invasive mechanical ventilation (procedureevents). NIV intentionally excluded
# for a stricter critical-illness definition; add 225794 to include NIV.
VENT_ITEMIDS = [225792]  # Invasive Ventilation
# Vasopressors (inputevents). starttime marks initiation.
PRESSOR_ITEMIDS = [
    221906,  # Norepinephrine
    221289,  # Epinephrine
    221749,  # Phenylephrine
    221662,  # Dopamine
    222315,  # Vasopressin
    221653,  # Dobutamine
]


def build_outcomes(ctx):
    """Step 8 & 9: 4-component composite outcome. Time-point independent, runs once.

    early_critical_illness = 1 if ANY of the following occurs within 24h of ED
    arrival (intime):
        (1) icu_24h     : ICU admission
        (2) vent_24h    : initiation of invasive mechanical ventilation
        (3) pressor_24h : vasopressor administration
        (4) death_hosp  : in-hospital mortality (no 24h window)

    NOTE: vent/pressor are recorded only in the ICU module, so in practice their
    positives are largely a subset of ICU admissions. This is a data limitation,
    not a coding choice — state it in Methods.
    """
    print("\n--- [Step 8 & 9: Create 4-Component Composite Outcome] ---")

    vent_ids = ",".join(str(i) for i in VENT_ITEMIDS)
    pressor_ids = ",".join(str(i) for i in PRESSOR_ITEMIDS)

    # base cohort with subject_id (needed to bridge ED stay -> ICU-module events)
    ctx.register("base_c", pl.scan_parquet("ed_cohort_with_labels.parquet").select(
        ["subject_id", "stay_id", "hadm_id", "ed_intime"]
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

    # procedureevents: invasive ventilation start times (subject_id + starttime)
    ctx.register("procedureevents", pl.scan_parquet(ensure_parquet("procedureevents", MIMIC_DIRS)).select(
        ["subject_id", "itemid", "starttime"]
    ).with_columns([
        pl.col("itemid").cast(pl.Int64, strict=False),
        pl.col("starttime").str.to_datetime(strict=False),
    ]))

    # inputevents: vasopressor start times (subject_id + starttime)
    ctx.register("inputevents", pl.scan_parquet(ensure_parquet("inputevents", MIMIC_DIRS)).select(
        ["subject_id", "itemid", "starttime"]
    ).with_columns([
        pl.col("itemid").cast(pl.Int64, strict=False),
        pl.col("starttime").str.to_datetime(strict=False),
    ]))

    sql_oc = f"""
        WITH
        VentSource AS (
            SELECT subject_id, starttime
            FROM procedureevents
            WHERE itemid IN ({vent_ids})
        ),
        PressorSource AS (
            SELECT subject_id, starttime
            FROM inputevents
            WHERE itemid IN ({pressor_ids})
        ),
        IcuEvents AS (
            SELECT b.stay_id,
                   MAX(CASE WHEN i.icu_intime >= b.ed_intime
                             AND i.icu_intime <= b.ed_intime + INTERVAL '24 hour'
                            THEN 1 ELSE 0 END) AS icu_24h
            FROM base_c AS b
            LEFT JOIN icustays AS i
                   ON b.hadm_id = i.hadm_id
            GROUP BY b.stay_id
        ),
        VentEvents AS (
            SELECT b.stay_id,
                   MAX(CASE WHEN p.starttime >= b.ed_intime
                             AND p.starttime <= b.ed_intime + INTERVAL '24 hour'
                            THEN 1 ELSE 0 END) AS vent_24h
            FROM base_c AS b
            LEFT JOIN VentSource AS p
                   ON p.subject_id = b.subject_id
            GROUP BY b.stay_id
        ),
        PressorEvents AS (
            SELECT b.stay_id,
                   MAX(CASE WHEN iv.starttime >= b.ed_intime
                             AND iv.starttime <= b.ed_intime + INTERVAL '24 hour'
                            THEN 1 ELSE 0 END) AS pressor_24h
            FROM base_c AS b
            LEFT JOIN PressorSource AS iv
                   ON iv.subject_id = b.subject_id
            GROUP BY b.stay_id
        ),
        MortalityEvents AS (
            SELECT b.stay_id,
                   MAX(CASE WHEN a.hospital_expire_flag = 1 THEN 1 ELSE 0 END) AS death_hosp
            FROM base_c AS b
            LEFT JOIN admissions AS a
                   ON b.hadm_id = a.hadm_id
            GROUP BY b.stay_id
        )
        SELECT b.stay_id, b.hadm_id,
               COALESCE(icu.icu_24h, 0)        AS outcome_icu_24h,
               COALESCE(vent.vent_24h, 0)      AS outcome_vent_24h,
               COALESCE(press.pressor_24h, 0)  AS outcome_pressor_24h,
               COALESCE(mort.death_hosp, 0)    AS outcome_mortality
        FROM base_c AS b
        LEFT JOIN IcuEvents       AS icu   ON b.stay_id = icu.stay_id
        LEFT JOIN VentEvents      AS vent  ON b.stay_id = vent.stay_id
        LEFT JOIN PressorEvents   AS press ON b.stay_id = press.stay_id
        LEFT JOIN MortalityEvents AS mort  ON b.stay_id = mort.stay_id
    """
    ctx.execute(sql_oc).sink_parquet("outcome_components.parquet")

    ctx.register("outcomes", pl.scan_parquet("outcome_components.parquet"))
    sql_composite = """
        SELECT stay_id, hadm_id,
               outcome_icu_24h     AS icu_24h,
               outcome_vent_24h    AS vent_24h,
               outcome_pressor_24h AS pressor_24h,
               outcome_mortality   AS death_hosp,
               CASE WHEN outcome_icu_24h = 1
                      OR outcome_vent_24h = 1
                      OR outcome_pressor_24h = 1
                      OR outcome_mortality = 1
                    THEN 1 ELSE 0 END AS early_critical_illness
        FROM outcomes
    """
    ctx.execute(sql_composite).sink_parquet("outcome_critical_illness.parquet")

    # Component-wise + composite prevalence report
    comp = pl.scan_parquet("outcome_critical_illness.parquet").select(
        ["icu_24h", "vent_24h", "pressor_24h", "death_hosp", "early_critical_illness"]
    ).collect()
    n = len(comp)
    print(f"  ✅ 4-component composite computed (N={n:,}). Component positives:")
    for c in ["icu_24h", "vent_24h", "pressor_24h", "death_hosp"]:
        s = int(comp[c].sum())
        print(f"       - {c:<12}: {s:>7,} ({s/n*100:.2f}%)")
    pos = int(comp["early_critical_illness"].sum())
    print(f"     Composite early_critical_illness: {pos:,} ({pos/n*100:.2f}%)")


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