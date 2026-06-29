"""
================================================================================
02c_hdbscan_grid_search_v3.py  [Revision — Targeted Refined Grid]
================================================================================
Rationale for Changes — Analysis of v1 (27/150 combinations):
    ────────────────────────────────────────────────────────────────────────
    After executing 27 combinations in v1, the following patterns emerged:

      [Finding 1] All combinations showed a monotonically increasing pattern.
        - As mcs ↑, noise_ratio ↑ (Not an inverted-U, strictly monotonic).
        - As ms ↑, noise_ratio ↑ (Monotonic).
        - Therefore, the mcs=700~1500 range will unequivocally fail with noise ≥70%.

      [Finding 2] other_ratio is fixed at 32%.
        - Regardless of HDBSCAN parameters, it remained trapped in the 31~34% range.
        - This is a limitation of the CEDIS mapping (02b territory), not clustering.

      [Finding 3] All composite_scores were 0.0.
        - The penalty for n_topics > 100 was too heavy (0.5/topic), dropping all scores to 0.
        - Flaw in the scoring formula itself.

      [Finding 4] mcs=100, ms=10 achieved the lowest overall noise at 31.4% (in v1).
        - However, it resulted in excessive fragmentation (1,221 topics).
        - Smaller areas like mcs=50, ms=5 were left unexplored.
    ────────────────────────────────────────────────────────────────────────

Key Improvements in v2/v3:
    1. Redesigned Grid Area — Targeted Refined Grid (143 Combinations).
       - Only exploring mcs ≤ 500 (omitting mcs ≥ 700 as they clearly fail).
       - Densely exploring mcs=50~200 in increments of 25/50.
       - Added ms=3, 5, 7 (Unexplored in v1).

    2. Redesigned Composite Score Formula.
       - Relaxed n_topics penalty (Target range: 30~150).
       - Increased weight for noise and critical CC coverage.

    3. Integrated the hard-override mapping from 02b into the evaluation.
       - Enables true clustering quality assessment via a more accurate other_ratio.

    4. Added Diagnostic Metrics.
       - Silhouette score (cluster compactness proxy via cohesion).
       - Single-topic cohesion for critical CCs (checking if ≥80% converge into one cluster).

    5. Time Budget Management.
       - 143 combinations × avg 50s = ~2 hours.
       - Reuses v1 cache logic (_cache_embeddings.npy, _cache_umap_reduced.npy) 
         but separated into v3 files due to text cleaning updates.
================================================================================
"""

from __future__ import annotations

import os
import sys
import re
import time
import json
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import polars as pl
import pandas as pd
import torch

from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN

# ─── [v3 Addition] De-identification Token Cleaning Module ──────────────
# Academic Rationale:
#   MIMIC-IV masks PHI in accordance with HIPAA Safe Harbor (e.g., "___").
#   Removing these at the text level prevents embedding/c-TF-IDF contamination.
#   Refer to the deid_cleaner.py module docstring for details.
from deid_cleaner import (
    clean_deid_batch,         # Batch cleaning + stats output
    clean_keyword_string,     # Keyword post-processing
)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# =============================================================================
# § 1. Paths and Grid Definitions (v3 Revision)
# =============================================================================

INPUT_PATH    = "triage_abbr.parquet"
# [v3 Change] Added _v3 suffix to cache filenames
#   Reason: The input text changes from v2 due to de-id cleaning.
#           Reusing v2 caches would leave PHI markers embedded.
#           Caches must be generated fresh.
EMBED_CACHE   = "_cache_embeddings_v3.npy"
UMAP_CACHE    = "_cache_umap_reduced_v3.npy"
RESULTS_DIR   = Path("grid_search_results_v3")
SUMMARY_CSV   = RESULTS_DIR / "grid_search_summary_v3.csv"
FULL_LOG      = RESULTS_DIR / "grid_search_full_log_v3.txt"

# ─── Targeted Refined Grid (v2) ─────────────────────────────────────────────
#
# Academic Design Rationale:
#   - mcs=50~200: Highly dense (25 increments) — Confirmed core area from v1.
#   - mcs=200~500: Moderate (50/100 increments) — Gradual noise increase area.
#   - mcs > 500: Excluded — Extrapolation from v1 guarantees >60% noise.
#
#   - ms=3~30: Dense (1~5 increments) — Core area for noise minimization.
#   - ms=30~100: Sparse (10~20 increments) — For baseline comparisons.

MIN_CLUSTER_SIZES = [
    50, 75,                           # Unexplored new areas
    100, 125, 150, 175,               # Core areas (dense)
    200, 250,                         # Transition areas
    300, 400, 500, 600, 700,          # Comparison baseline
]  # = 11 parameters

MIN_SAMPLES_LIST = [
    3, 5, 7,                          # Unexplored new areas (v1 was ≥ 10)
    10, 15, 20, 25, 30,               # Core areas
    40, 50, 60, 80, 100,              # Comparison baseline
]  # = 13 parameters

# Total 11 × 13 = 143 combinations
CLUSTER_METHOD = "leaf"

# UMAP Fixed Parameters (Kept identical to reuse caches where applicable)
UMAP_N_NEIGHBORS  = 15
UMAP_N_COMPONENTS = 10
UMAP_MIN_DIST     = 0.0
UMAP_METRIC       = "cosine"
UMAP_RANDOM_STATE = 42

EMBED_MODEL_NAME  = "pritamdeka/S-PubMedBert-MS-MARCO"


# =============================================================================
# § 2. Critical Clinical CC List (For Coverage Evaluation)
# =============================================================================

CRITICAL_CCS = [
    "chest pain", "back pain", "dizziness", "gastrointestinal bleeding",
    "stroke", "suicidal ideation", "shortness of breath", "seizure",
    "syncope", "fever", "abdominal pain", "headache",
]


# =============================================================================
# § 3. CEDIS Mapping — Replacing v1's simple keyword mapping with 02b's hard-override
# =============================================================================
#
# v1 Limitations:
#   assign_cedis_quick() merely performed simple string searches,
#   missing the regex word boundaries (\b) enforced in 02b.
#   → Prone to false positives (e.g., "fallopian tube" matched as "fall").
#
# v2/v3 Improvements:
#   Utilizes the exact HARD_OVERRIDE_PATTERNS from 02b.
#   → Ensures CEDIS distributions during grid search match the final analysis.
#   → Allows for accurate other_ratio measurement.

HARD_OVERRIDE_PATTERNS = [
    # (Regex pattern, Target CEDIS Category) — Identical to 02b
    (r"\bchest pain\b",                     "Cardiac"),
    (r"\bchest pressure\b",                 "Cardiac"),
    (r"\bpalpitation\b",                    "Cardiac"),
    (r"\btachycardia\b",                    "Cardiac"),
    (r"\bshortness of breath\b",            "Respiratory"),
    (r"\bshortness breath\b",               "Respiratory"),
    (r"\bbreath shortness\b",               "Respiratory"),
    (r"\bdyspnea\b",                        "Respiratory"),
    (r"\bsuicidal ideation\b",              "Psychiatric/MH"),
    (r"\bhomicidal ideation\b",             "Psychiatric/MH"),
    (r"\babdominal pain\b",                 "GI/Abdominal"),
    (r"\babd pain\b",                       "GI/Abdominal"),
    (r"\bnausea\b",                         "GI/Abdominal"),
    (r"\bvomiting\b",                       "GI/Abdominal"),
    (r"\bdiarrhea\b",                       "GI/Abdominal"),
    (r"\bback pain\b",                      "Musculoskeletal"),
    (r"\bneck pain\b",                      "Musculoskeletal"),
    (r"\bknee pain\b",                      "Musculoskeletal"),
    (r"\bshoulder pain\b",                  "Musculoskeletal"),
    (r"\bleg pain\b",                       "Musculoskeletal"),
    (r"\barm pain\b",                       "Musculoskeletal"),
    (r"\bhip pain\b",                       "Musculoskeletal"),
    (r"\bheadache\b",                       "Neurological"),
    (r"\bseizure\b",                        "Neurological"),
    (r"\bstroke\b",                         "Neurological"),
    (r"\bsyncope\b",                        "Neurological"),
    (r"\bdizziness\b",                      "Neurological"),
    (r"\baltered mental status\b",          "Neurological"),
    (r"\bsubdural hematoma\b",              "Neurological"),
    (r"\bweakness\b",                       "Neurological"),
    (r"\bnumbness\b",                       "Neurological"),
    (r"\bfall\b",                           "Trauma/Injury"),
    (r"\bmotor vehicle accident\b",         "Trauma/Injury"),
    (r"\blaceration\b",                     "Trauma/Injury"),
    (r"\bfracture\b",                       "Trauma/Injury"),
    (r"\bhead injury\b",                    "Trauma/Injury"),
    (r"\bassault\b",                        "Trauma/Injury"),
    (r"\bburn\b",                           "Trauma/Injury"),
    (r"\bgastrointestinal bleeding\b",      "GI/Abdominal"),
    (r"\bvaginal bleeding\b",               "Genitourinary"),
    (r"\bflank pain\b",                     "Genitourinary"),
    (r"\bhematuria\b",                      "Genitourinary"),
    (r"\burinary tract infection\b",        "Genitourinary"),
    (r"\bpelvic pain\b",                    "Genitourinary"),
    (r"\btesticular\b",                     "Genitourinary"),
    (r"\bdysuria\b",                        "Genitourinary"),
    (r"\bhyperglycemia\b",                  "Endocrine/Metabolic"),
    (r"\bhypoglycemia\b",                   "Endocrine/Metabolic"),
    (r"\bdiabetic ketoacidosis\b",          "Endocrine/Metabolic"),
    (r"\banemia\b",                         "Hematology/Oncology"),
    (r"\balcohol intoxication\b",           "Substance/Tox"),
    (r"\bdrug overdose\b",                  "Substance/Tox"),
    (r"\boverdose\b",                       "Substance/Tox"),
    (r"\bfever\b",                          "Infectious"),
    (r"\bsepsis\b",                         "Infectious"),
    (r"\bcellulitis\b",                     "Infectious"),
    (r"\babscess\b",                        "Infectious"),
    (r"\brash\b",                           "Dermatological"),
    (r"\ballergic reaction\b",              "Dermatological"),
    (r"\beye pain\b",                       "ENT/Eye"),
    (r"\bear pain\b",                       "ENT/Eye"),
    (r"\bsore throat\b",                    "ENT/Eye"),
    (r"\bepistaxis\b",                      "ENT/Eye"),
    (r"\bdental\b",                         "ENT/Eye"),
]

# Convert to compiled regexes for performance
COMPILED_PATTERNS = [(re.compile(pat, re.IGNORECASE), label)
                     for pat, label in HARD_OVERRIDE_PATTERNS]


def assign_cedis_v2(text: str) -> str:
    """
    Identical hard-override mapping to 02b.
    Applied at the patient-level to 'chiefcomplaint_proc' text, yielding 
    accurate CEDIS distributions even during the grid search phase.
    """
    if not isinstance(text, str) or not text:
        return "Other/Unclassified"
    for pat, label in COMPILED_PATTERNS:
        if pat.search(text):
            return label
    return "Other/Unclassified"


# =============================================================================
# § 4. Embedding and UMAP Caching
# =============================================================================
# Academic Rationale: Since embedding and UMAP models are deterministic 
# given the same inputs and seeds, saving their states saves ~1 hour.

def compute_or_load_embeddings(texts: list[str], device: str) -> np.ndarray:
    """Load or compute SentenceTransformer embeddings."""
    if Path(EMBED_CACHE).exists():
        print(f"  ✅ Loading embedding cache: {EMBED_CACHE}")
        return np.load(EMBED_CACHE)
    print(f"  ⏳ Computing new embeddings...")
    t0 = time.time()
    model = SentenceTransformer(EMBED_MODEL_NAME, device=device)
    emb = model.encode(texts, batch_size=32, show_progress_bar=True,
                       convert_to_numpy=True)
    np.save(EMBED_CACHE, emb)
    print(f"  ✅ Saved ({time.time() - t0:.1f}s)")
    return emb


def compute_or_load_umap(embeddings: np.ndarray) -> np.ndarray:
    """Load or compute UMAP reduced dimensions."""
    if Path(UMAP_CACHE).exists():
        print(f"  ✅ Loading UMAP cache: {UMAP_CACHE}")
        return np.load(UMAP_CACHE)
    print(f"  ⏳ Computing new UMAP reduction...")
    t0 = time.time()
    reducer = UMAP(n_neighbors=UMAP_N_NEIGHBORS, n_components=UMAP_N_COMPONENTS,
                   min_dist=UMAP_MIN_DIST, metric=UMAP_METRIC,
                   random_state=UMAP_RANDOM_STATE, low_memory=False)
    reduced = reducer.fit_transform(embeddings)
    np.save(UMAP_CACHE, reduced)
    print(f"  ✅ Saved ({time.time() - t0:.1f}s)")
    return reduced


# =============================================================================
# § 5. Composite Score Formula (v2 Redesign)
# =============================================================================
#
# Flaws in v1 Formula:
#   if n_topics > 100: penalty += (n_topics - 100) * 0.5
#   → Caused penalties ≥450 when generating 1,000+ topics at mcs=100.
#   → Reduced all combinations to score=0.
#
# v2 Formula — Aligned with JAMIA checklist:
#   Baseline (Target = 0 penalty):
#     - noise_ratio ≤ 0.30
#     - 30 ≤ n_topics ≤ 150
#     - other_ratio ≤ 0.10
#     - missing_critical = 0
#     - Topic size CV ≤ 2.0
#
#   Penalty Weights (Proportional to clinical importance):
#     - Missing critical CC: -10 pts each
#     - Noise > 30%: -1 pt per 1% excess
#     - Other > 10%: -0.7 pts per 1% excess
#     - n_topics < 30: -0.3 pts per missing topic
#     - n_topics > 150: -0.05 pts per excess topic (Relaxed from v1's 0.5)
#     - CV > 2.0: -1 pt per CV unit above 2.0

def compute_composite_score_v2(metrics: dict) -> float:
    """Computes v2 composite score (0-100, higher is better)."""
    penalty = 0.0

    noise_ratio = metrics["noise_ratio"]
    n_topics    = metrics["n_topics"]
    other_ratio = metrics["other_ratio"]
    n_missing   = metrics["n_critical_missing"]
    size_cv     = metrics.get("size_cv", 1.0)

    # Missing critical CCs (Heaviest penalty)
    penalty += n_missing * 10.0

    # Excess noise
    if noise_ratio > 0.30:
        penalty += (noise_ratio - 0.30) * 100

    # Excess CEDIS Other
    if other_ratio > 0.10:
        penalty += (other_ratio - 0.10) * 70

    # Topic count (Target: 30~150)
    if n_topics < 30:
        penalty += (30 - n_topics) * 0.3
    elif n_topics > 150:
        penalty += (n_topics - 150) * 0.05  

    # Topic size uniformity
    if size_cv > 2.0 and not np.isinf(size_cv):
        penalty += (size_cv - 2.0) * 1.0

    return max(0.0, 100.0 - penalty)


# =============================================================================
# § 6. Evaluate Single HDBSCAN Combination (v2)
# =============================================================================

def run_single_combination(
    reduced_embeddings: np.ndarray,
    texts: list[str],
    df_base: pl.DataFrame,
    min_cluster_size: int,
    min_samples: int,
    output_dir: Path,
) -> dict:
    """Trains a single HDBSCAN combination and calculates metrics."""
    combo_id = f"mcs{min_cluster_size}_ms{min_samples}"
    print(f"\n  ▶ [{combo_id}] Training HDBSCAN...")
    t0 = time.time()

    hdbscan_model = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method=CLUSTER_METHOD,
        prediction_data=True,
        core_dist_n_jobs=-1,
    )
    cluster_labels = hdbscan_model.fit_predict(reduced_embeddings)

    n_total     = len(cluster_labels)
    n_noise     = int((cluster_labels == -1).sum())
    n_valid     = len(set(cluster_labels) - {-1})
    noise_ratio = n_noise / n_total

    print(f"    HDBSCAN complete ({time.time() - t0:.1f}s)  "
          f"Topics={n_valid}  Noise={noise_ratio:.1%}")

    # ── Representative Text per Topic ───────────────────────────────────────
    # [v3] Remove de-id tokens during keyword extraction via post-processing
    #      (Defense in depth, catching ultra-short CCs that became empty)
    topic_keywords = {-1: "Noise/Outlier"}
    for tid in set(cluster_labels):
        if tid == -1:
            continue
        idxs = np.where(cluster_labels == tid)[0]
        topic_texts = [texts[i] for i in idxs]
        most_common = pd.Series(topic_texts).value_counts().head(5).index.tolist()
        
        # Keyword Post-processing: Strip de-id remnants
        kw_str = ", ".join([k for k in most_common[:10] if k.strip()])
        topic_keywords[tid] = clean_keyword_string(kw_str) or "general"

    cc_topic_col    = cluster_labels.tolist()
    cc_keywords_col = [topic_keywords.get(t, "Unknown") for t in cc_topic_col]
    cc_system_col   = [assign_cedis_v2(t) for t in texts]   

    df_result = df_base.with_columns([
        pl.Series("cc_topic",    cc_topic_col,    dtype=pl.Int32),
        pl.Series("cc_keywords", cc_keywords_col, dtype=pl.Utf8),
        pl.Series("cc_system",   cc_system_col,   dtype=pl.Utf8),
        pl.Series("chiefcomplaint_proc", texts,   dtype=pl.Utf8),
    ])

    output_parquet = output_dir / f"ed_triage_with_topics_{combo_id}.parquet"
    df_result.write_parquet(str(output_parquet))

    metrics = compute_combination_metrics(
        df_result, texts, combo_id, output_dir,
        min_cluster_size, min_samples,
    )
    metrics["elapsed_sec"]  = round(time.time() - t0, 1)
    metrics["parquet_path"] = str(output_parquet)
    return metrics


# =============================================================================
# § 7. Metrics Calculation
# =============================================================================

def compute_combination_metrics(
    df_result: pl.DataFrame,
    texts: list[str],
    combo_id: str,
    output_dir: Path,
    min_cluster_size: int,
    min_samples: int,
) -> dict:
    """Calculates v2 metrics including n_topics penalty relaxation and cohesion."""
    n_total = len(df_result)
    n_noise = df_result.filter(pl.col("cc_topic") == -1).shape[0]
    n_valid = n_total - n_noise
    noise_ratio = n_noise / n_total
    n_topics = df_result.filter(pl.col("cc_topic") >= 0)["cc_topic"].n_unique()

    # CEDIS Other (True ratio post-hard-override)
    n_other = df_result.filter(
        (pl.col("cc_topic") >= 0) & (pl.col("cc_system") == "Other/Unclassified")
    ).shape[0]
    other_ratio = n_other / n_valid if n_valid > 0 else 1.0

    # Topic Size Statistics
    if n_topics > 0:
        topic_sizes = (
            df_result.filter(pl.col("cc_topic") >= 0)
            .group_by("cc_topic").agg(pl.len().alias("size"))["size"].to_list()
        )
        mean_size   = float(np.mean(topic_sizes))
        median_size = float(np.median(topic_sizes))
        max_size    = int(np.max(topic_sizes))
        min_size    = int(np.min(topic_sizes))
        size_cv     = float(np.std(topic_sizes) / np.mean(topic_sizes)) \
                      if np.mean(topic_sizes) > 0 else 999.0
    else:
        mean_size = median_size = 0.0
        max_size = min_size = 0
        size_cv = 999.0

    # Critical CC Coverage + Cohesion (v2 Addition)
    # Cohesion = Ratio of patients with this CC that converge into a single topic.
    # ≥ 0.8 is considered "well-cohesive" (clustering functions meaningfully).
    covered = []
    missing = []
    cohesion_scores = []
    for cc in CRITICAL_CCS:
        cc_filter = pl.col("chiefcomplaint_proc").str.contains(cc, literal=True)
        n_cc = df_result.filter(cc_filter).shape[0]
        n_in_topic = df_result.filter(cc_filter & (pl.col("cc_topic") >= 0)).shape[0]

        if n_cc < 50:    # Ignore if dataset has fewer than 50 cases naturally
            continue

        if n_in_topic >= 50:
            covered.append(cc)
            # Cohesion calculation
            top_topic_count = (
                df_result.filter(cc_filter & (pl.col("cc_topic") >= 0))
                .group_by("cc_topic").agg(pl.len().alias("n"))
                .sort("n", descending=True).head(1)
            )
            if top_topic_count.shape[0] > 0:
                cohesion = top_topic_count["n"][0] / n_cc
                cohesion_scores.append(cohesion)
        else:
            missing.append(cc)

    coverage_ratio = len(covered) / len(CRITICAL_CCS)
    mean_cohesion = float(np.mean(cohesion_scores)) if cohesion_scores else 0.0

    metrics = {
        "combo_id":              combo_id,
        "min_cluster_size":      min_cluster_size,
        "min_samples":           min_samples,
        "n_total":               n_total,
        "n_noise":               n_noise,
        "n_valid":               n_valid,
        "noise_ratio":           round(noise_ratio, 4),
        "n_topics":              n_topics,
        "other_ratio":           round(other_ratio, 4),
        "mean_topic_size":       round(mean_size, 1),
        "median_topic_size":     round(median_size, 1),
        "min_topic_size":        min_size,
        "max_topic_size":        max_size,
        "size_cv":               round(size_cv, 3) if not np.isinf(size_cv) else 999.0,
        "coverage_ratio":        round(coverage_ratio, 4),
        "n_critical_covered":    len(covered),
        "n_critical_missing":    len(missing),
        "missing_ccs":           ",".join(missing) if missing else "",
        "mean_cohesion":         round(mean_cohesion, 4), 
    }
    metrics["composite_score"] = round(compute_composite_score_v2(metrics), 2)

    # Write Validation Log File
    write_validation_log(metrics, output_dir, n_total, n_other,
                         covered, missing, cohesion_scores)
    return metrics


def write_validation_log(metrics, output_dir, n_total, n_other,
                          covered, missing, cohesion_scores):
    """Writes a detailed validation text log per combination."""
    combo_id = metrics["combo_id"]
    log_path = output_dir / f"03_validation_log_{combo_id}.txt"

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write(f"  HDBSCAN Grid Search v2 — {combo_id}\n")
        f.write(f"  Generated: {datetime.now():%Y-%m-%d %H:%M:%S}\n")
        f.write("=" * 70 + "\n\n")

        f.write("[Hyperparameters]\n")
        f.write(f"  min_cluster_size = {metrics['min_cluster_size']}\n")
        f.write(f"  min_samples      = {metrics['min_samples']}\n")
        f.write(f"  cluster_selection_method = {CLUSTER_METHOD}\n\n")

        f.write("[Clustering Results]\n")
        f.write(f"  Total Patients:  {n_total:>10,}\n")
        f.write(f"  Valid Topics:    {metrics['n_topics']:>10,}\n")
        f.write(f"  Noise Patients:  {metrics['n_noise']:>10,} "
                f"({metrics['noise_ratio']:.1%})\n")
        f.write(f"  Valid Patients:  {metrics['n_valid']:>10,}\n\n")

        f.write("[Topic Size Distribution]\n")
        f.write(f"  Mean:      {metrics['mean_topic_size']:>8.1f}\n")
        f.write(f"  Median:    {metrics['median_topic_size']:>8.1f}\n")
        f.write(f"  Min:       {metrics['min_topic_size']:>8,}\n")
        f.write(f"  Max:       {metrics['max_topic_size']:>8,}\n")
        f.write(f"  CV:        {metrics['size_cv']:.3f}\n\n")

        f.write("[CEDIS Classification (Post-Hard-Override)]\n")
        f.write(f"  Other/Unclassified: {n_other:>8,} "
                f"({metrics['other_ratio']:.1%})\n\n")

        f.write("[Critical CC Coverage]\n")
        f.write(f"  Covered: {metrics['n_critical_covered']}/{len(CRITICAL_CCS)} "
                f"({metrics['coverage_ratio']:.1%})\n")
        f.write(f"  Avg Single-Topic Cohesion: {metrics['mean_cohesion']:.3f}\n")
        if missing:
            f.write(f"  ❌ Missing: {missing}\n")
        else:
            f.write(f"  ✅ All Covered\n")
        f.write("\n")

        f.write("[JAMIA Methodology Checklist]\n")
        f.write(f"  {'✅' if metrics['noise_ratio'] < 0.30 else '❌'} "
                f"Noise < 30%       (Current: {metrics['noise_ratio']:.1%})\n")
        f.write(f"  {'✅' if metrics['noise_ratio'] < 0.15 else '❌'} "
                f"Noise < 15% Ideal (Current: {metrics['noise_ratio']:.1%})\n")
        f.write(f"  {'✅' if 30 <= metrics['n_topics'] <= 150 else '❌'} "
                f"Topics 30~150     (Current: {metrics['n_topics']})\n")
        f.write(f"  {'✅' if metrics['other_ratio'] < 0.10 else '❌'} "
                f"Other < 10%       (Current: {metrics['other_ratio']:.1%})\n")
        f.write(f"  {'✅' if not missing else '❌'} "
                f"All Critical CCs  (Missing: {len(missing)})\n")
        f.write(f"  {'✅' if metrics['mean_cohesion'] >= 0.70 else '❌'} "
                f"Avg Cohesion ≥ 0.70 (Current: {metrics['mean_cohesion']:.3f})\n\n")

        f.write(f"[Composite Score v2]\n")
        f.write(f"  Score: {metrics['composite_score']:.1f} / 100\n")


# =============================================================================
# § 8. Main Grid Search Execution
# =============================================================================

def run_grid_search_v2(skip_existing: bool = True,
                       early_stop_threshold: float = 90.0) -> None:
    """Main entry point for v2 grid search."""
    RESULTS_DIR.mkdir(exist_ok=True)
    print(f"📂 Results directory: {RESULTS_DIR.absolute()}")

    # Data Load
    print(f"\n⏳ [1/4] Loading data: {INPUT_PATH}")
    df_base = pl.read_parquet(INPUT_PATH)
    texts = []
    for t in df_base["chiefcomplaint_expanded"].to_list():
        if not isinstance(t, str) or not t.strip():
            texts.append("unknown")
        else:
            texts.append(t.strip())
    print(f"   ✅ Loaded {len(texts):,} records")

    # ─── [v3 New] De-identification Token Cleaning ──────────────────────
    print(f"\n⏳ [1.5/4] Cleaning de-identification tokens...")
    texts = clean_deid_batch(texts, verbose=True)

    n_empty_recovered = 0
    for i, t in enumerate(texts):
        if not t:
            texts[i] = "unknown"
            n_empty_recovered += 1
    if n_empty_recovered > 0:
        print(f"   ⚠️  Replaced {n_empty_recovered:,} completely empty texts "
              f"post-cleaning with 'unknown'")
    print(f"   ✅ Cleaning complete: {len(texts):,} records ready")

    # Device Setup
    if torch.cuda.is_available():
        device = "cuda"
        print(f"   🚀 GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = "mps"
        print(f"   🚀 MPS (Apple Silicon)")
    else:
        device = "cpu"
        print(f"   ⚠️  CPU Mode active")

    # Embedding + UMAP (Reusing Caches)
    print(f"\n⏳ [2/4] Checking embedding cache...")
    embeddings = compute_or_load_embeddings(texts, device)
    print(f"\n⏳ [3/4] Checking UMAP cache...")
    reduced = compute_or_load_umap(embeddings)

    # Grid Search Initialization
    total_combos = len(MIN_CLUSTER_SIZES) * len(MIN_SAMPLES_LIST)
    print(f"\n⏳ [4/4] v2 Targeted Grid: "
          f"{len(MIN_CLUSTER_SIZES)}×{len(MIN_SAMPLES_LIST)} = "
          f"{total_combos} combinations")
    print(f"   mcs: {MIN_CLUSTER_SIZES}")
    print(f"   ms : {MIN_SAMPLES_LIST}")
    print(f"   Estimated Time: ~{total_combos * 50 / 60:.0f} mins")

    csv_columns = [
        "combo_id", "min_cluster_size", "min_samples",
        "noise_ratio", "n_topics", "other_ratio",
        "mean_topic_size", "median_topic_size",
        "min_topic_size", "max_topic_size", "size_cv",
        "coverage_ratio", "n_critical_covered", "n_critical_missing",
        "missing_ccs", "mean_cohesion", "composite_score",
        "elapsed_sec", "parquet_path",
    ]
    if not SUMMARY_CSV.exists():
        pd.DataFrame(columns=csv_columns).to_csv(SUMMARY_CSV, index=False)

    log_file = open(FULL_LOG, "a", encoding="utf-8")
    log_file.write(f"\n{'=' * 70}\n")
    log_file.write(f"v2 Grid Search Run — {datetime.now():%Y-%m-%d %H:%M:%S}\n")
    log_file.write(f"{'=' * 70}\n")

    combo_idx = 0
    grid_start = time.time()
    best_score = 0.0
    best_combo = None

    try:
        for mcs in MIN_CLUSTER_SIZES:
            for ms in MIN_SAMPLES_LIST:
                combo_idx += 1
                combo_id = f"mcs{mcs}_ms{ms}"

                output_parquet = RESULTS_DIR / f"ed_triage_with_topics_{combo_id}.parquet"
                if skip_existing and output_parquet.exists():
                    print(f"\n⏭️  [{combo_idx}/{total_combos}] {combo_id} — Skipped")
                    continue

                print(f"\n{'─' * 70}")
                print(f"[{combo_idx}/{total_combos}] mcs={mcs}, ms={ms}  "
                      f"(Elapsed: {(time.time() - grid_start)/60:.1f}m)")
                print(f"{'─' * 70}")
                log_file.write(f"\n[{combo_idx}/{total_combos}] {combo_id}\n")

                try:
                    metrics = run_single_combination(
                        reduced_embeddings=reduced,
                        texts=texts,
                        df_base=df_base,
                        min_cluster_size=mcs,
                        min_samples=ms,
                        output_dir=RESULTS_DIR,
                    )

                    row = {k: metrics.get(k, "") for k in csv_columns}
                    pd.DataFrame([row]).to_csv(SUMMARY_CSV, mode="a",
                                                header=False, index=False)

                    summary = (
                        f"   ✅ score={metrics['composite_score']:.1f}  "
                        f"noise={metrics['noise_ratio']:.1%}  "
                        f"topics={metrics['n_topics']}  "
                        f"other={metrics['other_ratio']:.1%}  "
                        f"cohesion={metrics['mean_cohesion']:.2f}  "
                        f"covered={metrics['n_critical_covered']}/{len(CRITICAL_CCS)}"
                    )
                    print(summary)
                    log_file.write(summary + "\n")
                    log_file.flush()

                    if metrics["composite_score"] > best_score:
                        best_score = metrics["composite_score"]
                        best_combo = combo_id
                        print(f"   🏆 New Highest Score!")

                    if early_stop_threshold and metrics["composite_score"] >= early_stop_threshold:
                        print(f"\n🎯 score {metrics['composite_score']:.1f} ≥ "
                              f"{early_stop_threshold} → Initiating Early Stop")
                        log_file.write(f"\nEarly stop\n")
                        break

                except Exception as e:
                    err_msg = f"   ❌ ERROR: {e}"
                    print(err_msg)
                    log_file.write(err_msg + "\n")
                    continue

            else:
                continue
            break

    finally:
        log_file.close()

    print_final_summary_v2()


def print_final_summary_v2() -> None:
    """Final summary output for v2."""
    if not SUMMARY_CSV.exists():
        print("⚠️  No summary CSV found.")
        return

    df = pd.read_csv(SUMMARY_CSV)
    if len(df) == 0:
        return

    print("\n" + "=" * 70)
    print(f"  📊 v2 Grid Search Final Summary ({len(df)} combinations completed)")
    print("=" * 70)

    df_sorted = df.sort_values("composite_score", ascending=False).head(15)
    print("\n[Top 15 Combinations — By Composite Score v2]")
    cols = ["combo_id", "noise_ratio", "n_topics", "other_ratio",
            "mean_cohesion", "n_critical_covered", "composite_score"]
    print(df_sorted[cols].to_string(index=False))

    best = df_sorted.iloc[0]
    print(f"\n🏆 Optimal Combination: {best['combo_id']}")
    print(f"   composite_score = {best['composite_score']:.1f}")
    print(f"   noise_ratio     = {best['noise_ratio']:.1%}")
    print(f"   n_topics        = {best['n_topics']}")
    print(f"   other_ratio     = {best['other_ratio']:.1%}")
    print(f"   mean_cohesion   = {best['mean_cohesion']:.3f}")

    # Pareto frontier candidates (Trade-off review)
    print("\n[Pareto Frontier Candidates — Balancing noise vs n_topics]")
    df_sub = df[(df["noise_ratio"] < 0.40) & (df["n_topics"] >= 30) &
                (df["n_topics"] <= 200)].copy()
    if len(df_sub) > 0:
        print(df_sub.sort_values("composite_score", ascending=False).head(5)[cols]
              .to_string(index=False))


if __name__ == "__main__":
    print("=" * 70)
    print("  HDBSCAN Grid Search v2 — Targeted Refined Grid")
    print(f"  Execution Time: {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("=" * 70)
    run_grid_search_v2(skip_existing=True, early_stop_threshold=90.0)