# AI-derived Information Phenotypes for Early Risk Stratification in the Emergency Department

Analysis code for *AI-derived Information Phenotypes for Early Risk Stratification in the
Emergency Department: A Retrospective Study* (MIMIC-IV-ED).

## Data access

The study uses MIMIC-IV and MIMIC-IV-ED (PhysioNet). Access requires credentialing and a
signed Data Use Agreement. **No data are included in this repository**, and no row-level
data or derived row-level files may be committed.

## Setup

```bash
pip install -r requirements.txt
```

Set the three folders with environment variables:

| Variable | Content |
|---|---|
| `MIMIC_DIR` | Folder with the MIMIC-IV / MIMIC-IV-ED `*.csv.gz` files (searched recursively, e.g. `ed/`, `hosp/`, `icu/`) |
| `WORK_DIR` | Intermediate row-level files (parquet, caches, predictions, fitted models). Never share. |
| `OUT_DIR` | Aggregate tables and figures |

Windows (cmd):

```bat
set MIMIC_DIR=D:\mimic-iv
set WORK_DIR=D:\ed_work
set OUT_DIR=D:\ed_outputs
python run_all.py
```

macOS / Linux:

```bash
export MIMIC_DIR=~/mimic-iv WORK_DIR=~/ed_work OUT_DIR=~/ed_outputs
python run_all.py          # all steps
python run_all.py 7        # restart from step 07
```

## Pipeline

| Step | Script | Manuscript |
|---|---|---|
| 00 | `00_convert_mimic_to_parquet.py` – CSV(.gz) → parquet | – |
| 01 | `01_preprocess_abbrev.py` – chief-complaint abbreviation expansion | Methods: chief complaint preprocessing |
| 02 | `02_hdbscan_grid_search.py` – S-PubMedBert embeddings, UMAP, HDBSCAN grid (13 × 13 configurations), composite score | Methods: hyperparameter optimization; Table S2 |
| 03 | `03_topic_modeling.py` – BERTopic (UMAP n_neighbors = 15, n_components = 10; HDBSCAN min_cluster_size = 600, min_samples = 10, leaf) | Methods: chief complaint categorization |
| 04 | `04_reassign_cedis.py` – topic → CEDIS mapping and rule-based reassignment (13 subgroups) | Methods: chief complaint categorization; Table S3 |
| 05 | `05_build_ed_master_dataset.py` – cohort selection, composite outcome, window-specific predictors | Methods: study population, outcome, predictors; Fig. 1 |
| 06 | `06_finalize_analysis_data.py` – outcome merge, clipping of implausible values | Methods: predictors |
| 07 | `07_run_experiments.py` – XGBoost at T0.5/T1.0/T2.0, LR, RF, ResMLP, DeLong test, SHAP | Tables 2–3, Fig. 2, Fig. 4 |
| 08 | `08_fairness_audit.py` – performance by age group, sex and race | Table S4 |
| 09 | `09_tables_figures.py` – baseline characteristics, subgroup phenotypes | Table 1, Table 4, Fig. 3, Table S2 |

Shared modules: `paths.py` (folders), `deid_cleaner.py` (removal of de-identification
placeholders from chief-complaint text).

## Analysis as implemented

- **Cohort**: adult ED encounters (≥ 18 years) with valid arrival/departure times, available
  triage acuity and pain score, and a chief complaint mapped to one of 13 CEDIS subgroups.
  Repeat visits are analysed as separate encounters.
- **Outcome** (`early_critical_illness`): ICU admission, invasive mechanical ventilation or
  vasopressor administration within 24 h of ED arrival, or in-hospital death.
- **Observation windows**: for T0.5, T1.0 and T2.0 the same predictors are computed from data
  recorded from 15 min before arrival up to 30 min, 1 h and 2 h after arrival: demographics,
  mode of arrival, triage acuity, pain score, CEDIS subgroup, minimum/maximum vital signs, and
  the first available laboratory results (WBC, hemoglobin, platelets, creatinine, sodium,
  potassium, bicarbonate, lactate, glucose) with measurement indicators. Missing values are
  median-imputed.
- **Split**: de-identified anchor year ≤ 2180 → training; > 2180 → test.
- **Models** (T2.0 comparison; XGBoost also at T0.5 and T1.0):
  - LR: L2, C = 1.0, L-BFGS, max 1,000 iterations, balanced class weights, sigmoid calibration (5-fold).
  - RF: 300 trees, max depth 15, balanced class weights, isotonic calibration (5-fold).
  - XGBoost: 300 trees, learning rate 0.05, max depth 6, min child weight 5, subsample 0.8,
    column sample 0.8, log-loss.
  - ResMLP: input layer (128 units) + two residual blocks (128 units; batch normalization,
    GELU, 30% dropout); AdamW (lr 0.001, weight decay 1e-4); class-weighted BCE loss; up to
    30 epochs with early stopping (patience 5) on AUPRC in a 10% stratified validation
    subset of the training set.
- **Metrics**: AUROC, AUPRC, Brier score; specificity at the test-set threshold giving 90%
  sensitivity; DeLong test vs XGBoost.
- **Information phenotypes**: ΔAUROC and ΔAUPRC = T2.0 − T0.5 per CEDIS subgroup (subgroups
  with ≥ 50 test encounters and both outcome classes); non-monotonic AUROC → anomalous
  trajectory; otherwise cut-offs ΔAUPRC 0.07 and ΔAUROC 0.04.

## Reproducibility

Seeds are fixed (42). Results can still differ slightly between runs on different hardware or
package versions (GPU embeddings, UMAP, PyTorch). Record the package versions used with
`pip freeze > requirements_lock.txt`.
