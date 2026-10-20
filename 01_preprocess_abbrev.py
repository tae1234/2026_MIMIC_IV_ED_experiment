"""
================================================================================
01_preprocess_abbrev.py -- ED chief-complaint abbreviation expansion
================================================================================
Input : triage.parquet (MIMIC-IV-ED triage; created by 00_convert_mimic_to_parquet.py)
Output: triage_abbr.parquet (adds 'chiefcomplaint_expanded')

Lower-cases each chief complaint and expands ED abbreviations with
word-boundary regular expressions (e.g. '\bcp\b' -> 'chest pain', leaving
'cpr' untouched). ED_ABBREV_LIST, run_quality_check() and full_preprocess()
are imported by 03_topic_modeling.py.
================================================================================
"""


import paths  # makes WORK_DIR the working directory (see paths.py)
import polars as pl
import re
import time

# =========================================================
# MIMIC-IV-ED Custom Emergency Chief Complaint Dictionary
# Uses the regular expression \b (Word Boundary) to replace terms only when they are independent words.
# (e.g., '\bcp\b' is replaced with 'chest pain', leaving 'cpr' untouched.)
# =========================================================
ED_ABBREV_DICT = {
    # 1. Cardiovascular
    r"\bcp\b": "chest pain",
    r"\bchf\b": "congestive heart failure",
    r"\bafib\b": "atrial fibrillation",
    r"\bcad\b": "coronary artery disease",
    r"\bpalps\b": "palpitations",

    # 2. Respiratory
    r"\bsob\b": "shortness of breath",
    r"\bdoe\b": "dyspnea on exertion",
    r"\bcopd\b": "chronic obstructive pulmonary disease",
    r"\buri\b": "upper respiratory infection",
    r"\bpna\b": "pneumonia",
    r"\bpe\b": "pulmonary embolism",
    r"\bcough\b": "cough",

    # 3. Gastrointestinal / Abdominal
    r"\babd\b": "abdominal",
    r"\bn/v\b": "nausea and vomiting",
    r"\bn/v/d\b": "nausea vomiting and diarrhea",
    r"\bgib\b": "gastrointestinal bleed",
    r"\bruq\b": "right upper quadrant",
    r"\bluq\b": "left upper quadrant",
    r"\brlq\b": "right lower quadrant",
    r"\bllq\b": "left lower quadrant",

    # 4. Neurological / Psychiatric
    r"\bams\b": "altered mental status",
    r"\bh/a\b": "headache",
    r"\bha\b": "headache",
    r"\bsi\b": "suicidal ideation",
    r"\bhi\b": "homicidal ideation",
    r"\bsz\b": "seizure",
    r"\bcva\b": "cerebrovascular accident",
    r"\btia\b": "transient ischemic attack",
    r"\bloc\b": "loss of consciousness",
    r"\bod\b": "overdose",
    r"\betoh\b": "alcohol",
    r"\bsah\b": "subarachnoid hemorrhage",

    # 5. Trauma / Musculoskeletal
    r"\bmva\b": "motor vehicle accident",
    r"\bmvc\b": "motor vehicle collision",
    r"\bglf\b": "ground level fall",
    r"\blle\b": "left lower extremity",
    r"\brle\b": "right lower extremity",
    r"\blue\b": "left upper extremity",
    r"\brue\b": "right upper extremity",
    r"\blbp\b": "low back pain",
    r"\bfx\b": "fracture",
    r"\blac\b": "laceration",
    r"\blacs\b": "lacerations",

    # 6. General / Infection / Endocrine
    r"\bhtn\b": "hypertension",
    r"\bdka\b": "diabetic ketoacidosis",
    r"\buti\b": "urinary tract infection",
    r"\bss\b": "sickle cell",
    r"\bfever\b": "fever",
    r"\bmed\b": "medication",
    r"\bmeds\b": "medications",

    # 7. Clinical Shorthand
    r"\beval\b": "evaluation",
    r"\bc/o\b": "complains of",
    r"\bs/p\b": "status post",
    r"\br/o\b": "rule out",
    r"\bhx\b": "history",
    r"\bh/o\b": "history of",
    r"\bf/u\b": "follow up",
    r"\bfu\b": "follow up",
    r"\bpt\b": "patient"
}


def expand_clinical_abbreviations():
    print("⏳ 1. Loading dataset...")

    df = (
        pl.scan_parquet("triage.parquet")
        .filter(pl.col("chiefcomplaint").is_not_null())
        .collect()
    )

    chief_complaints_raw = df["chiefcomplaint"].to_list()
    print(f"✅ Loaded {len(chief_complaints_raw)} records successfully.")

    print("\n⏳ 2. Decoding abbreviations using regex...")
    start_time = time.time()

    processed_complaints = []

    for text in chief_complaints_raw:
        # Safely handle non-string values
        if not isinstance(text, str):
            processed_complaints.append("")
            continue

        # Unify to lowercase to prevent case fragmentation
        text_lower = text.lower()

        # Word replacement via dictionary mapping
        for pattern, full_word in ED_ABBREV_DICT.items():
            text_lower = re.sub(pattern, full_word, text_lower, flags=re.IGNORECASE)

        # Remove trailing and leading whitespaces
        processed_complaints.append(text_lower.strip())

    elapsed = time.time() - start_time
    print(f"✅ Abbreviation decoding complete. (Time elapsed: {elapsed:.2f}s)")

    print("\n⏳ 3. Saving results to intermediate parquet file...")

    # Add the expanded text column to the dataframe
    df_abbr = df.with_columns(
        pl.Series("chiefcomplaint_expanded", processed_complaints)
    )

    output_checkpoint = "triage_abbr.parquet"
    df_abbr.write_parquet(output_checkpoint)
    print(f"✅ Intermediate file saved: {output_checkpoint}\n")

    # Verify conversion results
    print("=== 👀 Checking conversion sample (containing 'cp' or 'sob') ===")
    sample_check = df_abbr.filter(
        pl.col("chiefcomplaint").str.to_lowercase().str.contains("cp|sob")
    ).select(["chiefcomplaint", "chiefcomplaint_expanded"]).head(10)
    print(sample_check)

    return df_abbr


# =========================================================
# Compatibility layer for 03_topic_modeling.py's dynamic import.
# 03_topic_modeling.py imports these three names at module-load time but (per its own
# comments) does not call full_preprocess() in its main path -- it only
# reads the chiefcomplaint_expanded column that expand_clinical_abbreviations()
# already wrote to triage_abbr.parquet. These exist so the import succeeds.
# =========================================================

# Reference copy of the abbreviation table as (pattern, expansion) pairs,
# matching 03_topic_modeling.py's comment "ED_ABBREV_LIST — Abbreviation dictionary (for
# reference during debugging)".
ED_ABBREV_LIST = list(ED_ABBREV_DICT.items())


def run_quality_check(parquet_path: str = "triage_abbr.parquet") -> dict:
    """Sanity-checks the abbreviation-expansion output.

    Not called by 03_topic_modeling.py's main path (validation-only), but kept real/useful
    rather than a stub: reports row-count integrity and how many known
    abbreviations still appear un-expanded in a sample, so you notice if a
    future edit to ED_ABBREV_DICT breaks the word-boundary regexes.
    """
    print("\n⏳ [run_quality_check] Validating abbreviation-expansion output...")

    df = pl.read_parquet(parquet_path)
    n_total = len(df)
    n_empty_expanded = df.filter(pl.col("chiefcomplaint_expanded") == "").height

    # Spot-check: how many rows still contain a raw abbreviation token that
    # should have been expanded (word-boundary match against the *expanded*
    # column -- a non-zero count usually means a new abbreviation appeared
    # that ED_ABBREV_DICT doesn't cover yet, not a bug in the regex itself).
    sample_patterns = list(ED_ABBREV_DICT.keys())[:15]
    leftover_hits = 0
    for pattern in sample_patterns:
        leftover_hits += df.filter(
            pl.col("chiefcomplaint_expanded").str.contains(pattern)
        ).height

    stats = {
        "n_total_rows": n_total,
        "n_empty_expanded": n_empty_expanded,
        "leftover_abbrev_hits_sampled": leftover_hits,
    }

    print(f"  Total rows           : {n_total:,}")
    print(f"  Empty expanded text   : {n_empty_expanded:,}")
    print(f"  Leftover abbrev hits  : {leftover_hits:,} (sampled first {len(sample_patterns)} patterns)")
    print("✅ [run_quality_check] Done.\n")
    return stats


def full_preprocess():
    """Fallback entry point kept for API compatibility with older callers.

    03_topic_modeling.py does not call this directly (it reads triage_abbr.parquet's
    chiefcomplaint_expanded column instead), but if anything else in the
    pipeline still calls full_preprocess(), this delegates to the exact same
    logic as expand_clinical_abbreviations() so results stay identical.
    """
    return expand_clinical_abbreviations()


if __name__ == "__main__":
    expand_clinical_abbreviations()
    run_quality_check()