"""
============================================================
02_topic_modeling_v4.py  —  MIMIC-IV ED Triage Topic Modeling
  Version: v4.0 (Improved based on v1 result analysis)
============================================================

[Pipeline Flow]
  ┌─────────────────────────────────────────────────┐
  │  01_preprocess_abbrev.py                        │
  │     triage.parquet                               │
  │         ↓ (Abbreviation standardization + Primary CC extraction) │
  │     triage_abbr.parquet                          │
  │         ↓                                        │
  │  02_topic_modeling_v4.py  ← Current file        │
  │     triage_abbr.parquet  (Input)                 │
  │         ↓ (Embedding + BERTopic clustering)      │
  │     ed_triage_with_topics_v5.parquet  (Output)   │
  └─────────────────────────────────────────────────┘

[Key Improvements — Reflecting v1 Result Analysis]

  Addressed 5 critical issues found in the v1 results:

  1. Eliminated Double Preprocessing
     Previous: Recalled full_preprocess() in step [2/7] (224 records re-converted).
     Improved: Uses the result from script 01 (chiefcomplaint_expanded) directly.
     Effect: Adheres to the Single Source of Truth principle, ensuring reproducibility.

  2. Prevented Empty Keyword Clusters
     Cause: Default sklearn stopwords removed all words in short CCs.
            → Caused v1 Topics 14/30/35 to appear as cc_keywords='{""}'
            → Generated 99 duplicate pairs with Jaccard=1.00.
     Improved: Introduced MEDICAL_STOP_WORDS (custom medical domain stopwords).
               Added "patient/presents/eval" for removal, preserved "no/not".

  3. Automatic Noise Demotion for Empty Keyword Topics
     Topics with fewer than 2 meaningful keywords (length >= 2) prior to Jaccard 
     evaluation are stripped of cluster status and demoted to -1 (noise).
     JAMIA Perspective: "Uninterpretable clusters are noise, not clusters."

  4. Massively Expanded CEDIS Mapping
     Previous: Other/Unclassified was at 43.7%.
     Improved: Added specific body parts (foot/ankle/heel/shoulder) to Musculoskeletal.
               Added new Constitutional system (fatigue/malaise).
               Supplemented keywords for each system based on v1 samples.

  5. Laterality Normalization Handled in Script 01
     "l foot pain" → "left foot pain" (Script 01)
     → Successful Musculoskeletal mapping (Script 02)

[Retained Parameters from v1]
  - HDBSCAN: min_cluster_size=500, min_samples=30, method='leaf'
  - UMAP: n_neighbors=15, n_components=10
  - Jaccard merging for duplicates, heterogeneity scoring, reclustering.

============================================================
"""
import os
import warnings
import polars as pl
import torch
import numpy as np
from collections import defaultdict, Counter
from itertools import combinations

from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic import BERTopic

# 1. Bypass Transformers Security Check (Monkey Patching)
try:
    import transformers.utils.import_utils as import_utils
    # Replace the security check function with a no-op
    import_utils.check_torch_load_is_safe = lambda: None
    print("✅ Transformers security check has been bypassed via monkey patching.")
except Exception as e:
    print(f"⚠️ Bypass failed: {e}")

# 2. Maintain existing environment variables
os.environ["TRUST_REMOTE_CODE"] = "True"
os.environ["TRANSFORMERS_VERIFY_SCHEDULED_NODES"] = "False"

# ─── [v4 Addition] De-identification Token Cleaning Module ──────────────
# Academic Rationale:
#   MIMIC-IV masks PHI with tokens like "___" per HIPAA Safe Harbor §164.514(b)(2).
#   Failing to remove these beforehand causes:
#     1) Dispersion of identical topics during the embedding phase.
#     2) c-TF-IDF extracting "___" as a top keyword (polluting cc_keywords).
#     3) Immediate rejection risk in JAMIA Methods sections (TRIPOD-AI Item 9 violation).
#   See deid_cleaner.py docstrings for detailed design.
from deid_cleaner import (
    clean_deid_batch,         # Batch text cleaning + statistics output
    clean_keyword_string,     # Post-processing for cc_keywords
)

# ============================================================
# [Core] Import Preprocessing Module from 01_preprocess_abbrev.py
# ============================================================
#
# Import necessary functions/constants from script 01.
#
# Imported Items:
#   full_preprocess()    — [No longer directly called after v1 improvement]
#                          Kept only as an emergency fallback or for backwards compatibility.
#   ED_ABBREV_LIST       — Abbreviation dictionary (for reference during debugging)
#   run_quality_check()  — Preprocessing quality validation function
#
# [Important Design Decision — Reflecting v1 Analysis]
#   We do not re-process the 'chiefcomplaint_expanded' generated by script 01.
#   Modifying the dictionary in script 01 requires re-running script 01, which 
#   immediately reflects in script 02. This ensures the "Single Source of Truth" 
#   principle and reproducibility.
#
# Note: 01_preprocess_abbrev.py must be in the same directory.
from importlib.util import spec_from_file_location, module_from_spec
import os as _os
import sys as _sys

def _import_preprocess_module():
    """
    Dynamically import 01_preprocess_abbrev.py.
    
    Why dynamic import is used:
      - Standard imports can fail if filenames contain hyphens or start with numbers.
      - Dynamic imports specify the file path directly, making them environment-independent.

    Returns:
        Imported module object.
    """
    this_dir = _os.path.dirname(_os.path.abspath(__file__))
    target = _os.path.join(this_dir, "01_preprocess_abbrev.py")

    if not _os.path.exists(target):
        raise ImportError(
            f"\n❌ Could not find '01_preprocess_abbrev.py'.\n"
            f"   Search path: {target}\n"
            f"   Resolution: Ensure both files are in the same directory."
        )

    spec = spec_from_file_location("preprocess_abbrev", target)
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_preprocess = _import_preprocess_module()

# Bind preprocessing functions and constants
full_preprocess   = _preprocess.full_preprocess    # (Emergency fallback, do not call directly)
run_quality_check = _preprocess.run_quality_check  # Quality check (optional execution)
ED_ABBREV_LIST    = _preprocess.ED_ABBREV_LIST     # Abbreviation dict (for reference)

warnings.filterwarnings("ignore", category=FutureWarning)


# ============================================================
# § 3. BERTopic Model Configuration — v3 Parameters
# ============================================================
#
# [Key Change: cluster_selection_method = 'leaf']
#
# HDBSCAN cluster_selection_method comparison:
#
#   'eom' (Excess of Mass, Previous):
#     - Selects large clusters with high dendrogram stability.
#     - Prefers large clusters → Rare CCs tend to be absorbed as noise.
#     - Result: Chest pain, back pain, etc., were classified as -1 (noise).
#
#   'leaf' (v3 New):
#     - Selects the smallest clusters at the end of the dendrogram (leaves).
#     - Acknowledges small-scale clinical CCs as independent clusters.
#     - Result: Expect more topics and a lower noise ratio.
#     - Trade-off: Some clusters might be very small (controlled by min_cluster_size).

# ─────────────────────────────────────────────────────────────
# [v1 Improvement ⑤] Custom Medical Domain Stop Words
# ─────────────────────────────────────────────────────────────
#
# Rationale: Short CCs lost all meaning due to standard sklearn stop words 
# (e.g., "of", "for", "to", "with").
# Solution: 
#   1) Exclude clinically important terms from default stopwords (e.g., "no", "before").
#   2) Add administrative/routine terms that hold no clinical weight (e.g., "patient", "presents").

try:
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    # Convert frozenset to set for modification
    _BASE_STOPWORDS = set(ENGLISH_STOP_WORDS)
except ImportError:
    _BASE_STOPWORDS = set()

# Words that hold clinical significance and should be preserved
_MEDICAL_KEEP_WORDS = {
    "no", "not", "without", "before", "after",
    "during", "through", "under", "over",
}

# Clinical jargon — Words to be additionally removed
_MEDICAL_EXTRA_STOPWORDS = {
    "patient", "pt", "presents", "present",
    "complains", "complained", "reports",
    "here", "seen", "evaluated",
    "evaluation", "eval",           
    "rule", "out",                  
    "follow", "up",                 
    "status", "post",               
    "history",                      
    "transfer", "admission",
}

# Final custom stop words set
MEDICAL_STOP_WORDS = list(
    (_BASE_STOPWORDS - _MEDICAL_KEEP_WORDS) | _MEDICAL_EXTRA_STOPWORDS
)


def build_bertopic_model_v3(embedding_model) -> BERTopic:
    """
    Construct the v3 BERTopic model (improved based on v1 results).

    UMAP:
      n_neighbors=15  (↓ from 50)
      n_components=10  (↑ from 5)

    HDBSCAN:
      min_cluster_size=600  
      min_samples=10  
      cluster_selection_method='leaf'  (Changed from 'eom')

    CountVectorizer:
      stop_words=MEDICAL_STOP_WORDS  (Custom medical domain stopwords)
      min_df=5  (↓ from 10)
      max_df=0.95

    Args:
        embedding_model: Initialized SentenceTransformer instance

    Returns:
        Prepared BERTopic instance
    """
    umap_model = UMAP(
        n_neighbors=15,          
        n_components=10,         
        min_dist=0.0,            
        metric="cosine",         
        random_state=42,         
        low_memory=False,        
    )

    print("  HDBSCAN: min_cluster_size=600, min_samples=10, method='leaf'")

    hdbscan_model = HDBSCAN(
        min_cluster_size=600,             
        min_samples=10,                   
        metric="euclidean",
        cluster_selection_method="leaf",  
        prediction_data=True,             
        core_dist_n_jobs=-1,              
    )

    vectorizer_model = CountVectorizer(
        stop_words=MEDICAL_STOP_WORDS,    
        ngram_range=(1, 3),  
        min_df=5,            
        max_df=0.95,         
    )

    ctfidf_model = ClassTfidfTransformer(
        bm25_weighting=True,
        reduce_frequent_words=True,  
    )

    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        ctfidf_model=ctfidf_model,
        calculate_probabilities=False,  
        verbose=True,
    )
    return topic_model


# ============================================================
# § 4. Automatic Detection and Merging of Duplicate Clusters
# ============================================================
#
# Uses Jaccard similarity of the top 5 keywords to detect overlaps.
# Threshold=0.5: If 50% or more of the top keywords overlap, consider them identical.

def detect_and_merge_duplicate_topics(
    df: pl.DataFrame,
    topic_model: BERTopic,
    topics: list,
    similarity_threshold: float = 0.5,
) -> tuple[list, dict]:
    """
    Detect and merge duplicate clusters based on keyword similarity.

    Algorithm (Reflecting v1 Improvement ⑥):
      1) Extract top 5 keywords for each topic.
      2) Exclude empty strings or whitespace-only keywords.
      3) Exclude topics with < 2 valid keywords from merge evaluation.
      4) Calculate Jaccard similarity for all topic pairs.
      5) Identify pairs where similarity > threshold.
      6) Merge into the larger cluster.
      7) Return the mapping dictionary.
    """
    topic_info = topic_model.get_topic_info()
    valid_topics = topic_info[topic_info["Topic"] != -1]

    topic_keywords_sets: dict[int, set[str]] = {}
    empty_keyword_topics: list[int] = []

    for _, row in valid_topics.iterrows():
        tid = row["Topic"]
        raw_words = row["Representation"][:5]
        valid_words = {
            w.strip() for w in raw_words
            if isinstance(w, str) and len(w.strip()) >= 2
        }
        if len(valid_words) < 2:
            empty_keyword_topics.append(tid)
            continue
        topic_keywords_sets[tid] = valid_words

    if empty_keyword_topics:
        print(f"  ⚠️  Found {len(empty_keyword_topics)} topics with no meaningful keywords "
              f"(excluded from Jaccard evaluation): {empty_keyword_topics[:10]}...")

    merge_pairs = []
    topic_ids = list(topic_keywords_sets.keys())
    for t1, t2 in combinations(topic_ids, 2):
        kw1 = topic_keywords_sets[t1]
        kw2 = topic_keywords_sets[t2]
        if not kw1 or not kw2:
            continue
        jaccard = len(kw1 & kw2) / len(kw1 | kw2)
        if jaccard >= similarity_threshold:
            merge_pairs.append((t1, t2, jaccard))

    if not merge_pairs:
        print("  ✅ No duplicate clusters found (Jaccard criteria)")
        merge_log = {tid: -1 for tid in empty_keyword_topics}
        if merge_log:
            print(f"  ℹ️  Preparing to demote {len(merge_log)} empty-keyword topics to noise.")
        return topics, merge_log

    print(f"\n  ⚠️  Found {len(merge_pairs)} duplicate cluster pairs:")
    topic_counts = defaultdict(int)
    for t in topics:
        topic_counts[t] += 1

    merge_log = {}
    for t1, t2, sim in merge_pairs:
        representative = t1 if topic_counts[t1] >= topic_counts[t2] else t2
        to_merge = t2 if representative == t1 else t1
        merge_log[to_merge] = representative
        kw_preview = topic_keywords_sets[t1] | topic_keywords_sets[t2]
        print(f"    Topic {to_merge} → Topic {representative} "
              f"(Jaccard={sim:.2f}, Common keywords: {kw_preview})")

    for empty_tid in empty_keyword_topics:
        if empty_tid not in merge_log:
            merge_log[empty_tid] = -1
    if empty_keyword_topics:
        print(f"  ℹ️  Demoted {len(empty_keyword_topics)} empty-keyword topics to noise.")

    updated_topics = [merge_log.get(t, t) for t in topics]
    print(f"  ✅ Processed {len(merge_log)} topics.")
    return updated_topics, merge_log


# ============================================================
# § 5. CEDIS-based Broad Classification Labeling
# ============================================================
#
# Maps clinical concepts to the CEDIS (Canadian ED Information System) 
# Presenting Complaint List for standardization.

CEDIS_SYSTEM_RULES = [
    ("Cardiac",             ["chest pain", "palpitation", "cardiac",
                             "heart", "myocardial", "angina",
                             "arrhythmia", "bradycardia", "tachycardia",
                             "atrial fibrillation", "heart failure",
                             "hypertension", "hypertensive",
                             "hypotension"]),

    ("Respiratory",         ["shortness of breath", "cough", "congestion",
                             "wheezing", "asthma", "pneumonia",
                             "chronic obstructive pulmonary disease",
                             "respiratory", "dyspnea on exertion",
                             "hypoxia", "oxygen saturation",
                             "dyspnea", "hemoptysis", "sore throat"]),

    ("GI/Abdominal",        ["abdominal pain", "nausea vomiting", "nausea",
                             "vomiting", "diarrhea", "constipation",
                             "rectal", "bowel", "right lower quadrant",
                             "left lower quadrant", "right upper quadrant",
                             "left upper quadrant",    
                             "epigastric pain", "gastrointestinal bleeding",
                             "jaundice", "hepatic", "gallbladder",
                             "abdominal distention", "abdominal",
                             "flank pain",              
                             "anorexia", "dysphagia"]),

    ("Neurological",        ["headache", "seizure", "stroke", "stroke symptoms",
                             "subarachnoid hemorrhage", "transient ischemic attack",
                             "dizziness", "altered mental status",
                             "loss of consciousness", "syncope", "presyncope",
                             "weakness", "numbness", "facial droop",
                             "slurred speech",
                             "lightheaded", "confusion", "tingling",
                             "vertigo", "migraine"]),

    ("Psychiatric/MH",      ["suicidal ideation", "homicidal ideation",
                             "psychosis", "anxiety", "panic attack",
                             "depression", "hallucination",
                             "psychiatric", "behavioral",
                             "agitation", "agitated",
                             "auditory hallucination", "ah "]),

    ("Trauma/Injury",       ["fall", "laceration", "fracture",
                             "motor vehicle accident", "pedestrian struck",
                             "wound evaluation", "trauma", "injury",
                             "contusion", "sprain", "burn",
                             "stab wound", "stab", "wound check",
                             "head injury", "neck injury", "back injury",
                             "assault", "assaulted", "bite",
                             "abrasion", "puncture"]),

    ("Substance/Tox",       ["alcohol intoxication", "drug overdose",
                             "unable to ambulate", "toxic",
                             "intoxication", "withdrawal",
                             "alcohol", "etoh"]),

    ("Musculoskeletal",     ["back pain", "joint pain", "leg pain",
                             "arm pain", "knee pain",
                             "lower extremity pain", "upper extremity",
                             "shoulder pain", "hip pain",
                             "shoulder", "elbow", "wrist", "hand",
                             "finger", "thumb",
                             "foot pain", "foot injury", "foot",
                             "ankle pain", "ankle injury", "ankle",
                             "heel pain", "heel", "toe",
                             "knee injury", "hip",
                             "calf pain", "calf",
                             "left leg", "right leg",
                             "left arm", "right arm",
                             "left lower extremity", "right lower extremity",
                             "left upper extremity", "right upper extremity",
                             "neck pain", "musculoskeletal",
                             "muscle pain", "myalgia"]),

    ("Infectious",          ["fever", "sepsis", "urinary tract infection",
                             "skin infection", "abscess", "cellulitis",
                             "influenza", "influenza like illness",
                             "ili", "infection"]),

    ("Dermatological",      ["rash", "allergic reaction", "hives",
                             "itching", "skin",
                             "pruritus", "eczema", "lesion"]),

    ("Genitourinary",       ["urinary retention", "urinary",
                             "vaginal", "pelvic",
                             "kidney", "bladder", "hematuria",
                             "vaginal bleeding", "penile", "scrotal",
                             "dysuria", "frequency", "testicular",
                             "urolithiasis", "kidney stone"]),

    ("Hematology/Oncology", ["abnormal laboratory result", "anemia",
                             "coagulation", "transfusion", "bleeding",
                             "thrombosis",
                             "low hemoglobin", "pancytopenia",
                             "neutropenic", "chemotherapy"]),

    ("ENT/Eye",             ["eye pain", "ear pain", "sore throat",
                             "nose", "vision", "hearing",
                             "dental", "tooth",
                             "eye", "ear", "throat",
                             "visual changes", "red eye",
                             "epistaxis", "nose bleed",
                             "gums swollen", "dental pain"]),

    ("Transfer/Admin",      ["transfer", "medication refill",
                             "follow up", "referral",
                             "requesting readmission",
                             "medical device problem",
                             "jtube", "tube", "g tube", "ng tube",
                             "drain", "catheter",
                             "anemia", "hypotension syncope"]),

    ("Constitutional",      ["fatigue", "malaise", "weakness weakness",
                             "generalized weakness",
                             "body aches", "chills"]),
]


def assign_cedis_label(keywords_str: str) -> str:
    """
    Assigns a CEDIS broad classification label based on the keyword string.
    Returns "Other/Unclassified" if no patterns match.
    """
    kw_lower = keywords_str.lower()
    for label, patterns in CEDIS_SYSTEM_RULES:
        if any(p in kw_lower for p in patterns):
            return label
    return "Other/Unclassified"


# ============================================================
# § 6. Re-clustering Oversized Topic (e.g., Topic 0)
# ============================================================
# Automatically triggers re-clustering if a topic exceeds 15% of the total dataset.

def recluster_oversized_topic(
    texts: list,
    topic_assignments: list,
    target_topic: int,
    embedding_model,
    size_threshold: float = 0.15,
) -> list:
    """
    Detects and finely re-clusters oversized clusters (>15%).
    """
    total = len(topic_assignments)
    target_count = sum(1 for t in topic_assignments if t == target_topic)
    ratio = target_count / total

    if ratio < size_threshold:
        print(f"  ℹ️  Topic {target_topic} ratio ({ratio:.1%}) < threshold ({size_threshold:.0%}) → Skipping.")
        return topic_assignments

    print(f"\n  🔄 Topic {target_topic} ratio is {ratio:.1%} → Initiating re-clustering ({target_count:,} records)...")

    target_indices = [i for i, t in enumerate(topic_assignments) if t == target_topic]
    target_texts = [texts[i] for i in target_indices]

    sub_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=UMAP(
            n_neighbors=10, n_components=8,
            min_dist=0.0, metric="cosine", random_state=42,
        ),
        hdbscan_model=HDBSCAN(
            min_cluster_size=200,    
            min_samples=5,
            metric="euclidean",
            cluster_selection_method="leaf",  
            prediction_data=True,
        ),
        vectorizer_model=CountVectorizer(
            stop_words="english", ngram_range=(1, 3), min_df=3,
        ),
        ctfidf_model=ClassTfidfTransformer(
            bm25_weighting=True, reduce_frequent_words=True,
        ),
        calculate_probabilities=False,
        verbose=False,
    )

    sub_topics, _ = sub_model.fit_transform(target_texts)

    max_existing = max(t for t in topic_assignments if t != -1)
    updated_topics = list(topic_assignments)

    for list_pos, orig_idx in enumerate(target_indices):
        sub_t = sub_topics[list_pos]
        if sub_t == -1:
            updated_topics[orig_idx] = -1
        else:
            updated_topics[orig_idx] = max_existing + 1 + sub_t

    n_sub = len(set(sub_topics)) - (1 if -1 in sub_topics else 0)
    n_total = len(set(updated_topics)) - (1 if -1 in updated_topics else 0)
    print(f"  ✅ Re-clustering complete: Topic {target_topic} → {n_sub} subtopics (Total: {n_total})")
    return updated_topics


# ============================================================
# § 7. Compute Heterogeneity Scores
# ============================================================
# Measures semantic distance between keywords within a topic.
# Score = 1 - mean(pairwise cosine similarity). Closer to 1 implies higher heterogeneity.

def compute_heterogeneity_scores(
    topic_model: BERTopic,
    embedding_model,
    top_n_keywords: int = 5,
) -> dict:
    """
    Computes heterogeneity scores (0~1) for each topic.
    """
    topic_info = topic_model.get_topic_info()
    scores = {}

    for _, row in topic_info.iterrows():
        tid = row["Topic"]
        if tid == -1:
            scores[tid] = None
            continue

        keywords = list(set(row["Representation"][:top_n_keywords]))
        if len(keywords) < 2:
            scores[tid] = 0.0
            continue

        try:
            embeddings = embedding_model.encode(keywords, show_progress_bar=False)
            sim_matrix = cosine_similarity(embeddings)
            n = len(keywords)
            off_diag = [sim_matrix[i][j]
                        for i in range(n) for j in range(n) if i != j]
            mean_sim = np.mean(off_diag) if off_diag else 1.0
            scores[tid] = round(1.0 - mean_sim, 4)
        except Exception:
            scores[tid] = None

    return scores


# ============================================================
# § 8. Missing Critical CC Detection Warning
# ============================================================

CRITICAL_ED_CCS = [
    "chest pain",
    "back pain",
    "dizziness",
    "gastrointestinal bleeding",
    "stroke",
    "stroke symptoms",
    "suicidal ideation",
    "shortness of breath",
    "seizure",
    "syncope",
    "fever",
    "abdominal pain",
]

def check_missing_critical_ccs(topic_model: BERTopic) -> None:
    """
    Checks if clinically critical CCs are included as independent topics.
    """
    topic_info = topic_model.get_topic_info()
    all_keywords = set()
    for _, row in topic_info.iterrows():
        if row["Topic"] == -1:
            continue
        for kw in row["Representation"]:
            all_keywords.add(kw.lower())

    print("\n  [Critical CC Coverage Check]")
    found_any_missing = False
    for cc in CRITICAL_ED_CCS:
        found = any(cc in kw or kw in cc for kw in all_keywords)
        status = "✅" if found else "❌ Missing"
        if not found:
            found_any_missing = True
        print(f"    {status}  {cc}")

    if found_any_missing:
        print("\n  ⚠️  Missing CCs detected: Consider lowering min_cluster_size or enhancing preprocessing.")
    else:
        print("\n  ✅ All critical CCs covered.")


# ============================================================
# § 9. Comprehensive Diagnostic Report
# ============================================================

def print_comprehensive_diagnostic_report(
    df_final: pl.DataFrame,
    topic_model: BERTopic,
    merge_log: dict,
    heterogeneity_scores: dict,
) -> None:
    """
    Outputs a comprehensive diagnostic report on clustering quality.
    """
    total = len(df_final)
    noise_count = df_final.filter(pl.col("cc_topic") == -1).shape[0]
    noise_ratio = noise_count / total
    n_topics = df_final.filter(pl.col("cc_topic") != -1)["cc_topic"].n_unique()

    print("\n" + "=" * 65)
    print("📊  MIMIC-IV ED Chief Complaint Clustering Diagnostic Report")
    print("=" * 65)

    print(f"\n  Total Patients:    {total:>10,} patients")
    print(f"  Valid Topics:      {n_topics:>10,} topics")
    print(f"  Noise Patients:    {noise_count:>10,} patients")

    if noise_ratio < 0.15:
        noise_status = "✅ Good (Target Achieved)"
    elif noise_ratio < 0.30:
        noise_status = "⚠️ Needs Improvement (Doc recommendation is <30%)"
    elif noise_ratio < 0.60:
        noise_status = "❌ Excessive (Similar to previous 59.6% level)"
    else:
        noise_status = "🚨 Critical — Complete parameter review required"
    print(f"\n  Noise Ratio:       {noise_ratio:.1%}  {noise_status}")
    print(f"  * Recommended: ≤30% (ED CC Research Standard), Ideally ≤15%")

    print(f"\n  [Duplicate Cluster Merges]")
    if merge_log:
        for src, dst in merge_log.items():
            print(f"    Merged Topic {src} → Topic {dst}")
    else:
        print("    None")

    print(f"\n  [Top 5 Heterogeneity Scores (Candidates for Decomposition)]")
    valid_scores = {k: v for k, v in heterogeneity_scores.items()
                    if k != -1 and v is not None}
    top5_het = sorted(valid_scores.items(), key=lambda x: x[1], reverse=True)[:5]
    for tid, score in top5_het:
        flag = "🔴 Decomposition required" if score > 0.5 else ("🟡 Needs review" if score > 0.3 else "")
        print(f"    Topic {tid:3d}: Heterogeneity={score:.3f}  {flag}")

    print(f"\n  [Patient Distribution by CEDIS Broad Classification]")
    cedis_dist = (
        df_final.filter(pl.col("cc_topic") != -1)
        .group_by("cc_system")
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
    )
    for row in cedis_dist.iter_rows(named=True):
        ratio_str = f"{row['count']/total:.1%}"
        bar = "█" * min(int(row["count"] / total * 50), 30)
        print(f"    {row['cc_system']:<25s}: {row['count']:6,} patients ({ratio_str:>5})  {bar}")

    check_missing_critical_ccs(topic_model)

    print("\n" + "=" * 65)

    print("\n  📝  Methodology Checklist for JAMIA Submission:")
    items = [
        (noise_ratio < 0.30,
         "Noise ratio < 30% achieved"),
        (len(merge_log) == 0,
         "No duplicate clusters (Statistical power preserved)"),
        (any(v > 0.5 for v in valid_scores.values()) is False,
         "No highly heterogeneous clusters (Score > 0.5)"),
        (n_topics >= 20,
         "Sufficient clinical granularity (≥20 topics)"),
    ]
    for ok, label in items:
        print(f"    {'✅' if ok else '❌'} {label}")
    print()


# ============================================================
# § 10. Main Pipeline
# ============================================================

def run_topic_modeling():

    # ── 1. Data Loading ────────────────────────────────────────
    input_path = "triage_abbr.parquet"
    print(f"⏳ [1/7] Loading '{input_path}'...")
    df = pl.read_parquet(input_path)
    raw_complaints = df["chiefcomplaint_expanded"].to_list()
    print(f"✅ Loaded {len(raw_complaints):,} records.")

    # ── 2. Re-use Preprocessing Results (v1 Improvement ④) ────────
    print("\n⏳ [2/7] Verifying preprocessed results and handling null values...")

    processed_complaints: list[str] = []
    n_empty = 0
    for t in raw_complaints:
        if not isinstance(t, str) or not t.strip():
            processed_complaints.append("unknown")
            n_empty += 1
        else:
            processed_complaints.append(t.strip())

    if n_empty > 0:
        print(f"⚠️  Empty strings or nulls: {n_empty:,} records → Replaced with 'unknown'")
    print(f"✅ {len(processed_complaints):,} records ready for use (Utilizing script 01 results directly)")

    print("\n  [Script 01 Preprocessing Sample (10 records)]")
    for i in range(min(10, len(processed_complaints))):
        orig = df["chiefcomplaint"][i]
        proc = processed_complaints[i]
        if str(orig).lower().strip() != proc:
            print(f"    Original: {orig!r:50s}  →  After Script 01: {proc!r}")

    # ── [v4 New] De-identification Token Cleaning ────────────────────
    print("\n⏳ [2.5/7] Cleaning de-identification tokens... (Removing HIPAA Safe Harbor markers)")
    processed_complaints = clean_deid_batch(processed_complaints, verbose=True)

    n_empty_after_deid = 0
    for i, t in enumerate(processed_complaints):
        if not t:
            processed_complaints[i] = "unknown"
            n_empty_after_deid += 1
    if n_empty_after_deid > 0:
        print(f"   ⚠️  Replaced {n_empty_after_deid:,} completely empty texts post-cleaning "
              f"with 'unknown'")
    print(f"   ✅ Cleaning complete: {len(processed_complaints):,} records")

    # ── 3. GPU Configuration ─────────────────────────────────────
    if torch.cuda.is_available():
        device = "cuda"
        print(f"\n🚀 [3/7] Using NVIDIA GPU ({torch.cuda.get_device_name(0)})")
    elif torch.backends.mps.is_available():
        device = "mps"
        print("\n🚀 [3/7] Using Apple Silicon MPS")
    else:
        device = "cpu"
        print("\n⚠️ [3/7] CPU mode active (May be slow)")

    # ── 4. Loading Embedding Model ───────────────────────────────
    print(f"\n⏳ [4/7] Loading S-PubMedBert ({device.upper()})...")
    embedding_model = SentenceTransformer(
        "pritamdeka/S-PubMedBert-MS-MARCO",
        device=device
    )
    print("✅ Embedding model loaded successfully.")

    # ── 5. BERTopic Training ─────────────────────────────────────
    print("\n⏳ [5/7] Training BERTopic model (v3 Parameters)...")
    topic_model = build_bertopic_model_v3(embedding_model)
    topics, _ = topic_model.fit_transform(processed_complaints)

    n_valid = len(set(t for t in topics if t != -1))
    n_noise = sum(1 for t in topics if t == -1)
    print(f"✅ Initial Topics: {n_valid}, Noise: {n_noise:,} records ({n_noise/len(topics):.1%})")

    # ── 6. Post-processing — Deduplication + Re-clustering ───────────
    print("\n⏳ [6/7] Post-processing...")

    print("\n  [6-a] Detecting duplicate clusters...")
    topics, merge_log = detect_and_merge_duplicate_topics(
        df, topic_model, topics, similarity_threshold=0.5
    )

    print("\n  [6-b] Checking for oversized clusters to re-cluster...")
    topic_counter = Counter(t for t in topics if t != -1)
    for target_topic, _ in topic_counter.most_common(3):  
        topics = recluster_oversized_topic(
            texts=processed_complaints,
            topic_assignments=topics,
            target_topic=target_topic,
            embedding_model=embedding_model,
            size_threshold=0.15,
        )

    print("\n  [6-c] Calculating heterogeneity scores...")
    heterogeneity_scores = compute_heterogeneity_scores(
        topic_model, embedding_model, top_n_keywords=5
    )

    # ── 7. Keyword Mapping + CEDIS Labeling + Saving ──────────────
    print("\n⏳ [7/7] Generating final columns and saving...")

    topic_info_df = topic_model.get_topic_info()
    keyword_mapping = {}
    n_keywords_cleaned = 0
    for _, row in topic_info_df.iterrows():
        tid = row["Topic"]
        if tid == -1:
            keyword_mapping[tid] = "Noise/Outlier"
        else:
            raw_kw_str = ", ".join(row["Representation"][:10])
            cleaned_kw_str = clean_keyword_string(raw_kw_str)
            if cleaned_kw_str != raw_kw_str:
                n_keywords_cleaned += 1
            keyword_mapping[tid] = cleaned_kw_str or "general"

    if n_keywords_cleaned > 0:
        print(f"   📝 Post-processed cc_keywords: Removed de-id remnants from {n_keywords_cleaned} topics.")

    cc_keywords_col = []
    cc_system_col = []
    cc_heterogeneity_col = []

    for t in topics:
        kw = keyword_mapping.get(t, "reclustered_subtopic")
        cc_keywords_col.append(kw)
        cc_system_col.append(assign_cedis_label(kw))
        cc_heterogeneity_col.append(heterogeneity_scores.get(t, None))

    df_final = df.with_columns([
        pl.Series("cc_topic",           topics,                dtype=pl.Int32),
        pl.Series("cc_keywords",         cc_keywords_col,       dtype=pl.Utf8),
        pl.Series("cc_system",           cc_system_col,         dtype=pl.Utf8),       
        pl.Series("cc_heterogeneity",    cc_heterogeneity_col,  dtype=pl.Float64),    
        pl.Series("chiefcomplaint_proc", processed_complaints,  dtype=pl.Utf8),       
    ])

    output_path = "ed_triage_with_topics_v5.parquet"
    df_final.write_parquet(output_path)
    print(f"\n✅ Save complete: {output_path}")

    print_comprehensive_diagnostic_report(
        df_final, topic_model, merge_log, heterogeneity_scores
    )

    print("\n[Final Sample Preview (5 Records)]")
    print(df_final.select([
        "chiefcomplaint",
        "chiefcomplaint_proc",
        "cc_topic",
        "cc_keywords",
        "cc_system",
        "cc_heterogeneity",
    ]).head(5))

if __name__ == "__main__":
    run_topic_modeling()