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

if __name__ == "__main__":
    expand_clinical_abbreviations()