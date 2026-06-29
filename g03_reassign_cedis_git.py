"""
================================================================================
02b_reassign_cedis_v2.py  [New — Post-Hoc CEDIS Reassignment]
================================================================================
Purpose:
    Preserves the BERTopic clustering results from 02_topic_modeling_v4.py,
    but reassigns the cc_system (CEDIS broad classification) labels based on
    the "actual CC text" rather than the c-TF-IDF keywords.

Root Cause:
    ────────────────────────────────────────────────────────────────
    The CEDIS assignment logic in the original topic modeling script relies solely
    on the c-TF-IDF output (cc_keywords). However, the ClassTfidfTransformer's
    reduce_frequent_words=True option down-weights core clinical terms like 
    "chest pain", "dyspnea", and "abdominal pain", causing the representative 
    keywords to degrade into noise words like "wall", "yesterday", or "eating".

    As a result, approximately 52,000 patients (12.3%) were misclassified.
    Critical examples:
      - Topic 0 (16,432 patients, chest pain) → Constitutional (Correct: Cardiac)
      - Topic 2 (11,025 patients, dyspnea)    → Other/Unclassified (Correct: Respiratory)
      - Topic 4 (6,787 patients, SI)          → Musculoskeletal (Correct: Psychiatric/MH)
      - Topic 5 (6,700 patients, abd pain)    → Cardiac (Correct: GI/Abdominal)

Academic Rationale:
    The CEDIS ontology (Grafstein E, et al. CJEM 2008;10(1):26-27) is designed
    for mapping based on the original Presenting Complaint text. Mapping based 
    on c-TF-IDF statistically adjusted keywords deviates from the design intent.

Core Algorithm:
    Stage 1: Extract the modal (most frequent) chiefcomplaint_proc text for each topic.
    Stage 2: Apply the expanded CEDIS rules (v4, adding Endocrine, Hematology, etc.).
    Stage 3: Patient-level hard override for strong clinical markers 
             (e.g., forcing "Cardiac" if "chest pain" is present in the text).

Input:
    ed_triage_with_topics_v5.parquet

Output:
    ed_triage_with_topics_v5_fixed.parquet

Execution:
    python 02b_reassign_cedis_v2.py
================================================================================
"""

from __future__ import annotations

import polars as pl
import re
from collections import Counter

INPUT_PATH  = "ed_triage_with_topics_v5.parquet"
OUTPUT_PATH = "ed_triage_with_topics_v5_fixed.parquet"


# =============================================================================
# § 1. Expanded CEDIS Rules (v4)
# =============================================================================
#
# [Key changes from previous versions]
#   1. Added Endocrine/Metabolic category (resolves hyperglycemia misclassifications).
#   2. Detailed Hematology/Oncology (resolves anemia misclassifications).
#   3. Removed overly broad patterns like "hypotension" from Cardiac to prevent
#      absorbing non-cardiac cases like abdominal pain.
#   4. Added SI/HI abbreviations directly to Psychiatric/MH.
#   5. Moved "fall" from Neurological to Trauma/Injury.
#   6. Expanded "dyspnea" and "shortness of breath" variants in Respiratory.
#   7. Reordered rules: Specific → General.

CEDIS_SYSTEM_RULES_V4 = [
    # 1. Psychiatric/Mental Health
    # Placed first as SI/HI are clinically urgent and must not be confused with others.
    ("Psychiatric/MH",      [
        "suicidal ideation", "suicidal",
        "homicidal ideation", "homicidal",
        "psychosis", "psychotic",
        "depression", "depressive",
        "hallucination", "hallucinating",
        "panic attack", "anxiety",
        "psychiatric", "mental health crisis",
    ]),

    # 2. Substance/Toxicology
    ("Substance/Tox",       [
        "alcohol intoxication", "alcohol", "etoh",
        "drug overdose", "overdose", "toxic ingestion",
        "intoxication", "withdrawal",
    ]),

    # 3. Endocrine/Metabolic
    ("Endocrine/Metabolic", [
        "hyperglycemia", "hypoglycemia",
        "diabetic ketoacidosis", "dka",
        "dehydration", "electrolyte",
        "thyroid", "adrenal",
    ]),

    # 4. Hematology/Oncology
    ("Hematology/Oncology", [
        "anemia", "anaemia",
        "abnormal laboratory result", "abnormal labs",
        "coagulation", "transfusion", "thrombosis",
        "chemotherapy", "chemo", "neutropenic",
    ]),

    # 5. Cardiac
    ("Cardiac",             [
        "chest pain", "chest pressure", "chest tightness", "chest discomfort",
        "palpitation", "palpitations",
        "cardiac arrest", "cardiac",
        "angina", "myocardial",
        "atrial fibrillation", "afib", "arrhythmia",
        "bradycardia", "tachycardia", "tachy",
        "heart failure", "chf",
    ]),

    # 6. Respiratory
    ("Respiratory",         [
        "shortness of breath", "shortness breath",
        "breath shortness", "breathing", "breathe",
        "dyspnea", "sob",
        "cough", "wheezing",
        "asthma", "pneumonia",
        "chronic obstructive pulmonary disease", "copd",
        "respiratory distress", "hypoxia", "hypoxemia",
        "hemoptysis",
    ]),

    # 7. Neurological
    ("Neurological",        [
        "headache", "migraine",
        "seizure",
        "stroke", "stroke symptoms", "cva",
        "subarachnoid hemorrhage", "sah",
        "subdural hematoma", "sdh",
        "transient ischemic attack", "tia",
        "dizziness", "vertigo", "dizzy", "lightheaded",
        "altered mental status", "ams", "confusion",
        "loss of consciousness", "loc",
        "syncope", "presyncope",
        "weakness", "numbness",
        "facial droop", "slurred speech",
    ]),

    # 8. Trauma/Injury
    ("Trauma/Injury",       [
        "fall",
        "motor vehicle accident", "mvc", "mva",
        "pedestrian struck",
        "laceration", "wound evaluation",
        "fracture", "fx",
        "head injury", "trauma",
        "assault", "contusion", "sprain",
        "burn", "burns",
        "bite",
    ]),

    # 9. GI/Abdominal
    ("GI/Abdominal",        [
        "abdominal pain", "abd pain",
        "nausea vomiting", "nausea", "vomiting",
        "diarrhea", "constipation",
        "rectal", "bowel",
        "right lower quadrant", "left lower quadrant",
        "right upper quadrant", "left upper quadrant",
        "epigastric pain", "epigastric",
        "gastrointestinal bleeding", "gi bleed",
        "jaundice", "hepatic", "gallbladder",
        "ercp",
    ]),

    # 10. Genitourinary
    ("Genitourinary",       [
        "flank pain", "flank",
        "hematuria",
        "urinary tract infection", "uti",
        "urinary retention",
        "vaginal bleeding", "vaginal",
        "pelvic pain", "pelvic",
        "testicular",
        "kidney", "bladder",
        "dysuria",
    ]),

    # 11. Infectious
    ("Infectious",          [
        "fever",
        "sepsis",
        "influenza like illness", "ili",
        "skin infection", "cellulitis", "abscess",
        "myalgia",
    ]),

    # 12. Musculoskeletal
    ("Musculoskeletal",     [
        "back pain", "low back pain", "lbp",
        "neck pain",
        "knee pain", "shoulder pain", "hip pain",
        "leg pain", "arm pain", "foot pain", "hand pain",
        "wrist pain", "ankle pain", "rib pain",
        "elbow pain",
        "joint pain",
        "lower extremity pain", "upper extremity pain",
        "musculoskeletal",
    ]),

    # 13. ENT/Eye
    ("ENT/Eye",             [
        "eye pain", "visual changes", "vision loss",
        "ear pain", "hearing",
        "sore throat", "throat",
        "nosebleed", "epistaxis",
        "dental", "tooth pain", "toothache",
        "sinus pain", "sinusitis",
    ]),

    # 14. Dermatological
    ("Dermatological",      [
        "rash",
        "allergic reaction", "anaphylaxis", "hives",
        "pruritus", "itching",
        "ulcer",
    ]),

    # 15. Constitutional
    ("Constitutional",      [
        "fatigue", "malaise", "weakness generalized",
        "failure to thrive", "weight loss",
        "unresponsive", "lethargy",
    ]),

    # 16. Transfer/Admin
    ("Transfer/Admin",      [
        "transfer", "medication refill",
        "follow up", "referral",
        "catheter", "tube", "foley",
    ]),
]


# =============================================================================
# § 2. Strong Clinical Markers — Patient-Level Hard Override
# =============================================================================
#
# These markers are clinically the most critical and unambiguous CCs in the ED.
# If these are present in 'chiefcomplaint_proc', they force a correct CEDIS 
# reassignment at the individual patient level, regardless of the topic's overall label.

HARD_OVERRIDE_PATTERNS = [
    # (Regex pattern, Target CEDIS Category)
    (r"\bchest pain\b",                     "Cardiac"),
    (r"\bchest pressure\b",                 "Cardiac"),
    (r"\bshortness of breath\b",            "Respiratory"),
    (r"\bshortness breath\b",               "Respiratory"),
    (r"\bbreath shortness\b",               "Respiratory"),  
    (r"\bdyspnea\b",                        "Respiratory"),
    (r"\bsuicidal ideation\b",              "Psychiatric/MH"),
    (r"\bhomicidal ideation\b",             "Psychiatric/MH"),
    (r"\babdominal pain\b",                 "GI/Abdominal"),
    (r"\babd pain\b",                       "GI/Abdominal"),
    (r"\bback pain\b",                      "Musculoskeletal"),
    (r"\bheadache\b",                       "Neurological"),
    (r"\bseizure\b",                        "Neurological"),
    (r"\bstroke\b",                         "Neurological"),
    (r"\bsyncope\b",                        "Neurological"),
    (r"\bdizziness\b",                      "Neurological"),
    (r"\baltered mental status\b",          "Neurological"),
    (r"\bsubdural hematoma\b",              "Neurological"),
    (r"\bfall\b",                           "Trauma/Injury"),
    (r"\bmotor vehicle accident\b",         "Trauma/Injury"),
    (r"\bgastrointestinal bleeding\b",      "GI/Abdominal"),
    (r"\bvaginal bleeding\b",               "Genitourinary"),
    (r"\bflank pain\b",                     "Genitourinary"),
    (r"\bhematuria\b",                      "Genitourinary"),
    (r"\burinary tract infection\b",        "Genitourinary"),
    (r"\bhyperglycemia\b",                  "Endocrine/Metabolic"),
    (r"\bhypoglycemia\b",                   "Endocrine/Metabolic"),
    (r"\bdiabetic ketoacidosis\b",          "Endocrine/Metabolic"),
    (r"\banemia\b",                         "Hematology/Oncology"),
    (r"\balcohol intoxication\b",           "Substance/Tox"),
    (r"\bdrug overdose\b",                  "Substance/Tox"),
    (r"\bfever\b",                          "Infectious"),
    (r"\brash\b",                           "Dermatological"),
    (r"\ballergic reaction\b",              "Dermatological"),
]


# =============================================================================
# § 3. CEDIS Rule Matching Functions
# =============================================================================

def apply_cedis_rules(text: str, rules: list) -> str | None:
    """
    Applies CEDIS rules sequentially to the given text.
    Returns the category on the first match, or None if no match is found.
    """
    if not text:
        return None
    for label, patterns in rules:
        if any(pat in text for pat in patterns):
            return label
    return None


def apply_hard_override(text: str) -> str | None:
    """
    Patient-level hard override for strong clinical markers.
    Strictly adheres to word boundaries (\\b) via regex.
    """
    if not text:
        return None
    for pattern, label in HARD_OVERRIDE_PATTERNS:
        if re.search(pattern, text, flags=re.IGNORECASE):
            return label
    return None


# =============================================================================
# § 4. 2-Stage CEDIS Reassignment Logic
# =============================================================================

def compute_topic_representative_text(df: pl.DataFrame) -> dict[int, str]:
    """
    Calculates the "representative CC text" for each topic by concatenating 
    the top 10 most frequent (modal) 'chiefcomplaint_proc' values.
    
    This is clinically more meaningful than c-TF-IDF outputs because it reflects
    actual patient terminology without the adverse down-weighting of frequent terms.
    """
    print("  📊 Extracting representative CC text for topics...")
    topic_texts: dict[int, str] = {}

    for tid in df["cc_topic"].unique().sort().to_list():
        if tid == -1:
            topic_texts[tid] = ""
            continue

        topic_df = df.filter(pl.col("cc_topic") == tid)
        cc_series = topic_df["chiefcomplaint_proc"].to_list()
        
        # Top 10 modal values
        top_ccs = [c for c, _ in Counter(cc_series).most_common(10)]
        topic_texts[tid] = " | ".join(top_ccs).lower()

    return topic_texts


def reassign_cedis_for_topic(
    topic_id: int,
    representative_text: str,
) -> tuple[str, str]:
    """
    Reassigns CEDIS for a single topic.

    Returns:
        (new_cc_system, confidence)
    """
    if topic_id == -1:
        return "Other/Unclassified", "noise"

    # Stage 1: Hard override (Prioritize strong markers)
    label = apply_hard_override(representative_text)
    if label:
        return label, "high"

    # Stage 2: Expanded CEDIS rules
    label = apply_cedis_rules(representative_text, CEDIS_SYSTEM_RULES_V4)
    if label:
        return label, "medium"

    # Stage 3: Fallback
    return "Other/Unclassified", "low"


# =============================================================================
# § 5. Main Reassignment Pipeline
# =============================================================================

def run_reassignment() -> None:
    """
    Executes the 2-Stage Post-Hoc CEDIS Reassignment.
    """
    print("=" * 70)
    print("  MIMIC-IV ED CEDIS Label Post-Hoc Reassignment")
    print("  (Switching from c-TF-IDF noise → actual CC text-based mapping)")
    print("=" * 70)

    # ── 1. Load Data ──────────────────────────────────────────────────────
    print(f"\n⏳ [1/5] Loading '{INPUT_PATH}'...")
    df = pl.read_parquet(INPUT_PATH)
    n_total = len(df)
    print(f"   ✅ Loaded {n_total:,} records")
    print(f"   Columns: {df.columns}")

    required = {"cc_topic", "chiefcomplaint_proc", "cc_system"}
    missing = required - set(df.columns)
    if missing:
        print(f"   ❌ Missing required columns: {missing}")
        return

    # ── 2. Extract Topic Representative CC Text ───────────────────────────
    print(f"\n⏳ [2/5] Extracting representative CC text for topics...")
    topic_rep_texts = compute_topic_representative_text(df)
    print(f"   ✅ Processed {len(topic_rep_texts):,} topics")

    # ── 3. Topic-Level CEDIS Reassignment ─────────────────────────────────
    print(f"\n⏳ [3/5] Reassigning CEDIS at the topic level...")
    topic_new_cedis: dict[int, str] = {}
    topic_confidence: dict[int, str] = {}
    for tid, rep_text in topic_rep_texts.items():
        new_label, conf = reassign_cedis_for_topic(tid, rep_text)
        topic_new_cedis[tid]  = new_label
        topic_confidence[tid] = conf

    # ── 4. Apply Reassignments to DataFrame ───────────────────────────────
    print(f"\n⏳ [4/5] Applying reassignment results to DataFrame...")

    # Backup original cc_system
    df_new = df.with_columns([
        pl.col("cc_system").alias("cc_system_orig"),
    ])

    # Apply topic-level mapping
    df_new = df_new.with_columns([
        pl.col("cc_topic").map_elements(
            lambda t: topic_new_cedis.get(t, "Other/Unclassified"),
            return_dtype=pl.Utf8,
        ).alias("cc_system_topic_level"),
        pl.col("cc_topic").map_elements(
            lambda t: topic_confidence.get(t, "low"),
            return_dtype=pl.Utf8,
        ).alias("cc_system_confidence"),
    ])

    # Patient-level hard override (Highest priority)
    def _row_override(proc_text: str, topic_label: str) -> str:
        if not isinstance(proc_text, str):
            return topic_label
        override = apply_hard_override(proc_text)
        return override if override else topic_label

    new_cc_system = [
        _row_override(row["chiefcomplaint_proc"], row["cc_system_topic_level"])
        for row in df_new.iter_rows(named=True)
    ]

    df_new = df_new.with_columns([
        pl.Series("cc_system", new_cc_system, dtype=pl.Utf8),
    ])

    # ── 5. Generate Stats and Save ────────────────────────────────────────
    print(f"\n⏳ [5/5] Generating statistics and saving...")

    n_changed = df_new.filter(
        pl.col("cc_system") != pl.col("cc_system_orig")
    ).shape[0]
    pct_changed = n_changed / n_total * 100
    print(f"\n   🔄 cc_system changed for {n_changed:,} out of {n_total:,} patients ({pct_changed:.1f}%)")

    # Before vs After Distribution
    print("\n   📊 cc_system Distribution Change (Before → After):")
    before = (
        df_new.group_by("cc_system_orig")
        .agg(pl.len().alias("n"))
        .sort("n", descending=True)
    )
    after = (
        df_new.group_by("cc_system")
        .agg(pl.len().alias("n"))
        .sort("n", descending=True)
    )
    all_labels = set(before["cc_system_orig"].to_list()) | set(after["cc_system"].to_list())
    print(f"   {'CEDIS Category':<22}  {'Before':>10}  {'After':>10}  {'ΔChange':>10}")
    print(f"   {'-'*22}  {'-'*10}  {'-'*10}  {'-'*10}")

    before_dict = dict(zip(before["cc_system_orig"].to_list(), before["n"].to_list()))
    after_dict  = dict(zip(after["cc_system"].to_list(),       after["n"].to_list()))

    for label in sorted(all_labels, key=lambda x: -after_dict.get(x, 0)):
        b = before_dict.get(label, 0)
        a = after_dict.get(label, 0)
        delta = a - b
        sign = "+" if delta >= 0 else ""
        print(f"   {label:<22}  {b:>10,}  {a:>10,}  {sign}{delta:>+10,}")

    # Display critical fix examples (useful for the Methods section of the paper)
    print("\n   📝 Key Correction Examples (Misclassified → Corrected):")
    critical_fixes = [
        ("chest pain",         "Cardiac"),
        ("dyspnea",            "Respiratory"),
        ("suicidal ideation",  "Psychiatric/MH"),
        ("abdominal pain",     "GI/Abdominal"),
        ("fall",               "Trauma/Injury"),
        ("dizziness",          "Neurological"),
        ("hyperglycemia",      "Endocrine/Metabolic"),
        ("anemia",             "Hematology/Oncology"),
    ]
    for cc_keyword, expected_cedis in critical_fixes:
        n = df_new.filter(
            (pl.col("chiefcomplaint_proc").str.contains(cc_keyword)) &
            (pl.col("cc_system") == expected_cedis)
        ).shape[0]
        print(f"     '{cc_keyword:<20}' → {expected_cedis:<22}: {n:>7,} patients")

    # Drop intermediate columns and save
    df_final = df_new.drop(["cc_system_topic_level"])
    df_final.write_parquet(OUTPUT_PATH)

    print(f"\n✅ Save Complete: {OUTPUT_PATH}")
    print(f"   Key Output Columns:")
    print(f"     - cc_system             : [Reassigned] CEDIS category")
    print(f"     - cc_system_orig        : [Preserved] Original topic modeling output")
    print(f"     - cc_system_confidence  : high/medium/low/noise")
    print()
    print("=" * 70)
    print("  Next Steps:")
    print("    → Re-run 03_validate_topics.py with the fixed file for quality validation.")
    print("    → Update the input filename in step02_create_ed_triage_v2.py to")
    print(f"      '{OUTPUT_PATH}'.")
    print("=" * 70)

if __name__ == "__main__":
    run_reassignment()