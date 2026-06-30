# AI-Derived Information Phenotypes for Early Risk Stratification in the Emergency Department

Machine-learning pipeline for early critical-illness risk stratification in the
emergency department (ED), built on the **MIMIC-IV-ED** dataset. The pipeline
clusters chief complaints into clinical subgroups, builds time-resolved feature
snapshots (0.5 h / 1 h / 2 h from ED arrival), trains and compares four model
families, and produces the tables and figures used in the manuscript.

> **Outcome (label):** `early_critical_illness` — ICU admission within 24 h of ED
> arrival **OR** initiation of invasive mechanical ventilation within 24 h **OR**
> vasopressor administration within 24 h **OR** in-hospital mortality.

---

## Table of contents

- [Pipeline overview](#pipeline-overview)
- [Repository layout](#repository-layout)
- [Requirements](#requirements)
- [Input data](#input-data)
- [How to run](#how-to-run)
  - [Stage A — Chief-complaint NLP & subgroup discovery](#stage-a--chief-complaint-nlp--subgroup-discovery)
  - [Stage B — Dataset construction](#stage-b--dataset-construction)
  - [Stage C — Experiments](#stage-c--experiments)
  - [Stage D — Figures](#stage-d--figures)
- [Key design decisions](#key-design-decisions)
- [Outputs](#outputs)
- [Reproducibility notes](#reproducibility-notes)
- [Troubleshooting](#troubleshooting)

---

## Pipeline overview

```
 ┌──────────────────────── Stage A: Chief-Complaint NLP ─────────────────────────┐
 │ g01_preprocess_abbrev      triage.parquet ─► triage_abbr.parquet              │
 │ g02_topic_modeling         triage_abbr.parquet ─► ed_triage_with_topics_v5    │
 │ g03_reassign_cedis         ─► ed_triage_with_topics_v5_fixed.parquet          │
 │ g03b_hdbscan_grid_search   (HDBSCAN parameter search, optional)               │
 │ g03c_grid_search_validation(validation orchestrator, optional)                │
 └───────────────────────────────────────────────────────────────────────────────┘
                                   │  ed_triage_with_topics_v5_fixed.parquet
                                   ▼
 ┌──────────────────────── Stage B: Dataset Construction ────────────────────────┐
 │ 1_build_ed_master_dataset  ─► ed_master_dataset_{0.5h,1h,2h}.parquet          │
 │                                outcome_critical_illness.parquet               │
 │ 3_finalize_analysis_data   ─► analysis_master_{0.5h,1h,2h}.parquet            │
 └───────────────────────────────────────────────────────────────────────────────┘
                                   │  analysis_master_{0.5h,1h,2h}.parquet
                                   ▼
 ┌──────────────────────── Stage C: Experiments ─────────────────────────────────┐
 │ 4_run_all_experiments                                                         │
 │   1. Subgroup time trends (XGBoost @ 0.5/1/2h) + global SHAP                  │
 │   2. ML baselines (LR / RF / XGBoost)                                          │
 │   3. PyTorch Residual MLP (ResMLP)                                             │
 │   4. Uncertainty-based selective prediction (risk-coverage)                    │
 └───────────────────────────────────────────────────────────────────────────────┘
                                   │  prediction CSVs + metric tables
                                   ▼
 ┌──────────────────────── Stage D: Figures ─────────────────────────────────────┐
 │ fig1_subgroup_timepoint_trajectory   (phenotype 4×2 panels)                   │
 │ fig2_model_roc_pr_overlay            (ROC + PR overlay, 4 models)             │
 │ fig3_xgboost_shap_beeswarm           (global SHAP beeswarm)                   │
 └───────────────────────────────────────────────────────────────────────────────┘
```

The four **information phenotypes** (Information-Dependent, Precision-Dependent,
Triage-Sufficient, Anomalous Trajectory) describe how much each clinical subgroup
benefits from adding vitals/labs over time relative to triage-only information.

---

## Repository layout

| Stage | Script | Role |
|-------|--------|------|
| A | `g01_preprocess_abbrev_git.py` | Expand ED chief-complaint abbreviations (e.g. `cp` → `chest pain`) via word-boundary regex |
| A | `g02_topic_modeling_git.py` | BERTopic clustering of chief complaints (PubMedBERT embeddings + UMAP + HDBSCAN) |
| A | `g03_reassign_cedis_git.py` | Post-hoc CEDIS subgroup reassignment from raw CC text (corrects c-TF-IDF mislabeling) |
| A | `g03b_hdbscan_grid_search_git.py` | Targeted HDBSCAN parameter grid search (optional tuning) |
| A | `g03c_grid_search_with_validation_git.py` | Orchestrator running topic validation across grid combinations (optional) |
| B | `1_build_ed_master_dataset.py` | Cohort, labels, time-resolved vitals/labs, composite outcome → per-timepoint masters |
| B | `3_finalize_analysis_data.py` | Outcome join, outlier clipping → `analysis_master_{0.5h,1h,2h}.parquet` |
| C | `4_run_all_experiments.py` | All four experiments + prediction CSVs + metric tables + SHAP |
| D | `fig1_subgroup_timepoint_trajectory.py` | Per-phenotype AUROC/AUPRC trajectory panels |
| D | `fig2_model_roc_pr_overlay.py` | Four-model ROC and PR curve overlay |
| D | `fig3_xgboost_shap_beeswarm.py` | XGBoost (2 h) global SHAP beeswarm |

> A `2_train_early_warning_models.py` exploratory script (T0 vs T2 only) also
> exists; the canonical experiment driver is `4_run_all_experiments.py`.

---

## Requirements

- Python 3.10+
- Core: `polars`, `pandas`, `numpy`, `pyarrow`, `scikit-learn`, `xgboost`
- Deep learning: `torch`
- Explainability / plots: `shap`, `matplotlib`, `seaborn`
- NLP stage: `sentence-transformers`, `bertopic`, `umap-learn`, `hdbscan`, `tqdm`

```bash
pip install polars pandas numpy pyarrow scikit-learn xgboost torch \
            shap matplotlib seaborn sentence-transformers bertopic \
            umap-learn hdbscan tqdm
```

A CUDA-capable GPU is recommended for Stage A embeddings and the PyTorch ResMLP,
but the pipeline falls back to CPU.

---

## Input data

This project uses **MIMIC-IV** and **MIMIC-IV-ED**, which require credentialed
access via [PhysioNet](https://physionet.org/). Data are **not** included in this
repository.

Place the required source tables (CSV or `.csv.gz`) so that
`1_build_ed_master_dataset.py` can find them under one of:
`mimic/ed`, `mimic/hosp`, `mimic/icu`, or `mimic/`. The script auto-converts them
to Parquet on first run. Tables used include: `edstays`, `patients`, `triage`,
`vitalsign`, `admissions`, `labevents`, `d_labitems`, `icustays`,
`procedureevents`, and `inputevents`.

The NLP stage additionally expects a `triage.parquet` (from MIMIC-IV-ED `triage`).

---

## How to run

Run from a single working directory; intermediate Parquet/CSV files are written
there and consumed by later steps.

### Stage A — Chief-complaint NLP & subgroup discovery

```bash
python g01_preprocess_abbrev_git.py      # triage.parquet -> triage_abbr.parquet
python g02_topic_modeling_git.py         # -> ed_triage_with_topics_v5.parquet
python g03_reassign_cedis_git.py         # -> ed_triage_with_topics_v5_fixed.parquet
```

Optional HDBSCAN tuning (only if re-deriving clustering parameters):

```bash
python g03b_hdbscan_grid_search_git.py        # writes grid_search_results_v3/
python g03c_grid_search_with_validation_git.py
```

Stage A produces **`ed_triage_with_topics_v5_fixed.parquet`**, which carries the
`cc_system` (CEDIS subgroup) column consumed by Stage B.

### Stage B — Dataset construction

```bash
python 1_build_ed_master_dataset.py   # cohort, labels, time-resolved features, outcome
python 3_finalize_analysis_data.py    # -> analysis_master_{0.5h,1h,2h}.parquet
```

`1_build_ed_master_dataset.py` builds **three** feature snapshots, aggregating
vitals/labs over `[arrival − 15 min, arrival + X]` for X ∈ {0.5 h, 1 h, 2 h}.
Time-point–independent stages (cohort, labels, composite outcome) run once; only
vitals/labs aggregation repeats per window.

The composite outcome is written to `outcome_critical_illness.parquet` and contains
the four component indicators `icu_24h`, `vent_24h`, `pressor_24h`, and
`death_hosp`, plus the final binary label `early_critical_illness`.

### Stage C — Experiments

```bash
python 4_run_all_experiments.py
```

Runs all four experiments end to end and writes prediction CSVs and metric tables.

### Stage D — Figures

```bash
python fig1_subgroup_timepoint_trajectory.py   # needs analysis_master_*, edstays, patients
python fig2_model_roc_pr_overlay.py            # needs model_predictions_2h.csv + pytorch_predictions_2h.csv
python fig3_xgboost_shap_beeswarm.py           # needs analysis_master_2h, edstays, patients
```

Run Stage C before Stage D — `fig2` consumes the prediction CSVs produced by
`4_run_all_experiments.py`.

---

## Key design decisions

**Composite early critical illness outcome.** The primary label,
`early_critical_illness`, is positive when any of the following components is
present: ICU admission within 24 h of ED arrival, invasive mechanical ventilation
initiation within 24 h, vasopressor administration within 24 h, or in-hospital
mortality. Ventilation events are extracted from the ICU `procedureevents` table
and vasopressor events from the ICU `inputevents` table. Because these ICU-module
tables are primarily populated after ICU-level care begins, ventilation and
vasopressor positives may substantially overlap with ICU admission; this should
be stated as a data-source limitation when reporting the outcome definition.

**Time-resolved feature windows.** Vitals and labs are aggregated separately for
0.5 h, 1 h, and 2 h after ED arrival (`intime`), yielding three parallel master
tables. A small 15-minute pre-arrival grace window captures EMS/registration labs;
set `PRE_ARRIVAL_GRACE_H = 0.0` in `1_build_ed_master_dataset.py` to disable it.

**Pseudo–out-of-time (OOT) split.** All experiments and figures order encounters
by `anchor_year` (a de-identified calendar-time surrogate in MIMIC-IV) and use the
earliest 80 % for training and the latest 20 % for testing. Ties at the cutoff are
broken by `stay_id` so the split is deterministic and non-overlapping. This
preserves a train-on-earlier / test-on-later structure without a hard-coded year
threshold that could leave a split empty.

**Information phenotypes (subgroup classification).** The 13 CEDIS subgroups are
grouped into four phenotypes based on how AUROC/AUPRC evolve from 0.5 h to 2 h.
`fig1` maps subgroups to phenotypes per the manuscript's Table 4; verify the Δ
values in `subgroup_timepoint_metrics.csv` against that classification if the
split or cohort changes.

**Class imbalance.** Tree models use `scale_pos_weight = N_neg / N_pos`; the
ResMLP uses `BCEWithLogitsLoss(pos_weight=…)`. LR and RF baselines are probability-
calibrated (`CalibratedClassifierCV`).

---

## Outputs

| File | Produced by | Contents |
|------|-------------|----------|
| `ed_master_dataset_{0.5h,1h,2h}.parquet` | `1_` | Per-timepoint feature masters |
| `outcome_critical_illness.parquet` | `1_` | Composite outcome (`early_critical_illness`, `icu_24h`, `vent_24h`, `pressor_24h`, `death_hosp`) |
| `analysis_master_{0.5h,1h,2h}.parquet` | `3_` | Cleaned, outcome-joined analysis tables |
| `Table_Subgroup_Performance_Evolution.csv` | `4_` (Exp 1) | Subgroup × time-window AUROC/AUPRC/Brier/Spec@90 |
| `Figure4_Global_SHAP_Beeswarm.png` | `4_` (Exp 1) | Global SHAP beeswarm (XGBoost 2 h) |
| `model_predictions_2h.csv` | `4_` (Exp 2) | LR / RF / XGBoost test predictions + target |
| `pytorch_predictions_2h.csv` | `4_` (Exp 3) | ResMLP test predictions |
| `subgroup_timepoint_metrics.csv` | `fig1` | Per-subgroup 0.5/1/2 h AUROC/AUPRC + Δ + phenotype |
| `Figure1_Subgroup_Phenotype_Trajectory.{png,pdf}` | `fig1` | Phenotype-paneled trajectories |
| `Figure2_Model_ROC_PR_Overlay.{png,pdf}` | `fig2` | Four-model ROC + PR overlay |
| `Figure3_XGBoost_SHAP_Beeswarm.{png,pdf}` | `fig3` | XGBoost global SHAP beeswarm |

---

## Reproducibility notes

- Seeds are fixed (`random_state=42`, `torch.manual_seed(42)`), but exact metrics
  can vary slightly with hardware, library versions, and GPU non-determinism.
- The NLP stage uses the `pritamdeka/S-PubMedBert-MS-MARCO` sentence embedding
  model; the first run downloads weights.
- Reference performance (2 h window, XGBoost): AUROC ≈ 0.914, AUPRC ≈ 0.566, with
  model ordering LR < RF < ResMLP < XGBoost.
- Risk-coverage (Experiment 4): confident-subset AUROC increases monotonically as
  coverage decreases, supporting selective deferral of uncertain cases to labs.

---

## Troubleshooting

### Polars SQL `only equi-join constraints are currently supported`

If Stage B fails during composite outcome construction with an error similar to:

```text
polars.exceptions.SQLInterfaceError: only equi-join constraints are currently supported
```

the usual cause is a non-equi filter condition inside a SQL `JOIN ... ON` clause,
for example:

```sql
LEFT JOIN procedureevents AS p
       ON p.subject_id = b.subject_id AND p.itemid IN (...)
```

Polars SQL currently expects the `ON` clause to contain only equi-join predicates.
Filter item IDs before the join instead, for example by using pre-filtered CTEs:

```sql
VentSource AS (
    SELECT subject_id, starttime
    FROM procedureevents
    WHERE itemid IN (...)
),
PressorSource AS (
    SELECT subject_id, starttime
    FROM inputevents
    WHERE itemid IN (...)
)
```

Then join those filtered sources using only equality conditions such as
`p.subject_id = b.subject_id`. If the run failed midway, delete any partially
written outcome files before rerunning:

```text
outcome_components.parquet
outcome_critical_illness.parquet
```

---

## Citation

If you use this code, please cite the associated manuscript:

> *AI-Derived Information Phenotypes for Early Risk Stratification in the
> Emergency Department* (manuscript in preparation / under review).

This work uses MIMIC-IV and MIMIC-IV-ED (Johnson et al., PhysioNet). Please follow
the PhysioNet credentialing and data use agreement.
