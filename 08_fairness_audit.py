"""
================================================================================
08_fairness_audit.py -- Supplementary Table S4
================================================================================
AUROC and AUPRC of the T2.0 XGBoost model (fitted in 07_run_experiments.py and
saved as xgboost_t2_final.pkl) in the independent test set, stratified by age
group, sex and race.

Output (OUT_DIR): TableS4_Fairness_Audit.csv
================================================================================
"""
import joblib
import pandas as pd
import polars as pl
from sklearn.metrics import average_precision_score, roc_auc_score

import paths  # makes WORK_DIR the working directory

SPLIT_YEAR = 2180
MODEL_PATH = "xgboost_t2_final.pkl"
OUTPUT_CSV = paths.OUT_DIR / "TableS4_Fairness_Audit.csv"
TARGET = "early_critical_illness"

FEATURES_NUM = [
    "anchor_age", "triage_acuity", "triage_pain",
    "sbp_min", "sbp_max", "heartrate_max", "resprate_max",
    "o2sat_min", "temperature_max",
    "wbc", "hb", "plt", "cr", "na", "k", "hco3", "lactate", "glucose",
    "wbc_measured", "hb_measured", "plt_measured",
    "cr_measured", "lactate_measured",
]
FEATURES_CAT = ["gender", "race", "arrival_transport", "cc_system"]

RACE_MAPPING = {
    # White
    "WHITE":                                    "White",
    "WHITE - OTHER EUROPEAN":                   "White",
    "WHITE - RUSSIAN":                          "White",
    "WHITE - EASTERN EUROPEAN":                 "White",
    "WHITE - BRAZILIAN":                        "White",
    "PORTUGUESE":                               "White",
    # Black
    "BLACK/AFRICAN AMERICAN":                   "Black",
    "BLACK/AFRICAN":                            "Black",
    "BLACK/CAPE VERDEAN":                       "Black",
    "BLACK/CARIBBEAN ISLAND":                   "Black",
    # Hispanic
    "HISPANIC/LATINO - PUERTO RICAN":           "Hispanic",
    "HISPANIC/LATINO - DOMINICAN":              "Hispanic",
    "HISPANIC OR LATINO":                       "Hispanic",
    "HISPANIC/LATINO - GUATEMALAN":             "Hispanic",
    "HISPANIC/LATINO - SALVADORAN":             "Hispanic",
    "HISPANIC/LATINO - COLOMBIAN":              "Hispanic",
    "HISPANIC/LATINO - CENTRAL AMERICAN":       "Hispanic",
    "HISPANIC/LATINO - MEXICAN":                "Hispanic",
    "HISPANIC/LATINO - HONDURAN":               "Hispanic",
    "HISPANIC/LATINO - CUBAN":                  "Hispanic",
    "SOUTH AMERICAN":                           "Hispanic",
    # Asian
    "ASIAN":                                    "Asian",
    "ASIAN - CHINESE":                          "Asian",
    "ASIAN - SOUTH EAST ASIAN":                 "Asian",
    "ASIAN - ASIAN INDIAN":                     "Asian",
    "ASIAN - KOREAN":                           "Asian",
    # Other
    "OTHER":                                    "Other",
    "AMERICAN INDIAN/ALASKA NATIVE":            "Other",
    "NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER":"Other",
    # Unmapped values (UNKNOWN, UNABLE TO OBTAIN, PATIENT DECLINED TO ANSWER, ...)
    # become NaN and are excluded from the race strata.
}

SEX_MAPPING = {"F": "Female", "M": "Male"}
AGE_ORDER = ["18-39", "40-64", "65 years and older"]


def evaluate_subgroup(data, group_col, category_label, order):
    results = []
    for group in order:
        subset = data[data[group_col] == group]
        y_true, y_prob = subset[TARGET], subset["pred_prob"]
        if y_true.nunique() < 2:
            print(f"   {category_label} - {group}: single outcome class, skipped")
            continue
        results.append({
            "Category": category_label, "Subgroup": str(group), "N": len(subset),
            "Events": int(y_true.sum()),
            "AUROC": round(roc_auc_score(y_true, y_prob), 4),
            "AUPRC": round(average_precision_score(y_true, y_prob), 4),
        })
    return results


def run_fairness_audit():
    lf_edstays = pl.scan_parquet("edstays.parquet").select(["stay_id", "subject_id"]).with_columns(
        pl.col("stay_id").cast(pl.Int64, strict=False))
    lf_patients = pl.scan_parquet("patients.parquet").select(["subject_id", "anchor_year"]).with_columns(
        pl.col("subject_id").cast(pl.Int64, strict=False))
    df_year_lookup = lf_edstays.join(lf_patients, on="subject_id", how="inner").select(
        ["stay_id", "anchor_year"]).collect().to_pandas()

    df = pl.scan_parquet("analysis_master_2h.parquet").collect().to_pandas()
    if "anchor_year" in df.columns:
        df = df.drop(columns=["anchor_year"])
    df = pd.merge(df, df_year_lookup, on="stay_id", how="left")

    avail_num = [c for c in FEATURES_NUM if c in df.columns]
    avail_cat = [c for c in FEATURES_CAT if c in df.columns]
    for col in avail_num:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df_test = df.loc[df["anchor_year"] > SPLIT_YEAR].copy()
    df_test["race_group"] = df_test["race"].map(RACE_MAPPING)
    df_test["age_group"] = pd.cut(df_test["anchor_age"], bins=[17, 39, 64, 150], labels=AGE_ORDER)
    df_test["sex_label"] = df_test["gender"].map(SEX_MAPPING)
    print(f"Test set N = {len(df_test):,}; race not reported/unknown (excluded from race strata): "
          f"{df_test['race_group'].isna().sum():,}")

    model_xgb = joblib.load(MODEL_PATH)  # T2.0 XGBoost pipeline from 07_run_experiments.py
    df_test["pred_prob"] = model_xgb.predict_proba(df_test[avail_num + avail_cat])[:, 1]
    print(f"Overall test AUROC {roc_auc_score(df_test[TARGET], df_test['pred_prob']):.4f}, "
          f"AUPRC {average_precision_score(df_test[TARGET], df_test['pred_prob']):.4f}")

    audit = []
    audit += evaluate_subgroup(df_test, "age_group", "Age group", AGE_ORDER)
    audit += evaluate_subgroup(df_test, "sex_label", "Sex", ["Female", "Male"])
    audit += evaluate_subgroup(df_test, "race_group", "Race", ["White", "Black", "Hispanic", "Asian", "Other"])
    df_audit = pd.DataFrame(audit)
    print(df_audit.to_string(index=False))
    df_audit.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    run_fairness_audit()
