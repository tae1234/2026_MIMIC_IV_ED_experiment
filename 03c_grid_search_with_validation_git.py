"""
================================================================================
02d_grid_search_with_validation.py  [New — Validation Integration Wrapper]
================================================================================
Purpose:
    Complements 02c_hdbscan_grid_search_v3.py by executing the full validation 
    module (03_validate_topics.py) for each grid combination and saving the 
    results into separate text files.

    This script acts as an orchestrator linking 02c and 03.

  Execution Flow (per combination):
    1. Loads the generated `ed_triage_with_topics_mcs{N}_ms{M}.parquet` from 02c.
    2. Imports the `inspect_topic_results()` function from 03_validate_topics.py 
       and executes the identical validation logic.
    3. Redirects stdout to `grid_search_results/03_validation_log_mcs{N}_ms{M}.txt`.
    4. Compiles a comprehensive ranking after validating all combinations.

  Usage:
    [Method A] Run 02c alone, then run this script for validation.
        $ python 02c_hdbscan_grid_search_v3.py
        $ python 02d_grid_search_with_validation.py

    [Method B] Call automatically from within 02c (if integrated).

  Notes:
    - 03_validate_topics.py must be located in the same directory.
    - Because 03_validate_topics.py auto-searches for candidate input files, 
      this script temporarily creates a standard filename (via copy) before 
      calling the validation function for each iteration.
================================================================================
"""

from __future__ import annotations

import os
import sys
import shutil
import importlib.util
from io import StringIO
from pathlib import Path
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime

import pandas as pd

# ── Path Definitions (Maintained identical to 02c) ────────────────────────
RESULTS_DIR = Path("grid_search_results_v3")
SUMMARY_CSV = RESULTS_DIR / "grid_search_summary_v3.csv"
VALIDATE_SCRIPT = "03_validate_topics.py"


# =============================================================================
# § 1. Dynamically Load 03_validate_topics.py Module
# =============================================================================
# Academic Rationale — Why use importlib instead of a standard import?
#   Filenames starting with numbers (like 03_validate_topics.py) cannot be 
#   imported using standard Python import statements.
#   importlib.util is used to load a file from an arbitrary path as a module.

def load_validate_module():
    """Dynamically loads 03_validate_topics.py to make its functions accessible."""
    if not Path(VALIDATE_SCRIPT).exists():
        raise FileNotFoundError(
            f"{VALIDATE_SCRIPT} not found in the current directory. "
            f"This script must be in the same directory as 02d."
        )

    spec = importlib.util.spec_from_file_location("validate_mod", VALIDATE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# =============================================================================
# § 2. Single Combination Validation
# =============================================================================
# The inspect_topic_results() function in 03_validate_topics.py auto-detects
# hardcoded filenames (e.g., "ed_triage_with_topics_v5.parquet").
# Therefore, we temporarily copy the grid combination file to a standard filename 
# before executing the validation.

def validate_single_combination(parquet_path: Path, combo_id: str) -> dict:
    """
    Performs validation for a single grid combination using 03_validate_topics.py.

    Procedure:
      1. Create a copy with a temporary standard filename.
      2. Execute 03_validate_topics.py -> inspect_topic_results().
      3. Capture stdout and save it to a text file.
      4. Clean up temporary files.
    """
    # The filename that script 03 prioritizes when searching
    standard_name = "ed_triage_with_topics_v5_fixed.parquet"
    standard_path = Path(standard_name)

    # Backup existing file (if any)
    backup_path = None
    if standard_path.exists():
        backup_path = Path(f"{standard_name}.grid_backup")
        shutil.move(str(standard_path), str(backup_path))

    try:
        # Copy the grid result to the standard location
        # Academic Note: Copy is used instead of symlink to avoid potential 
        # permission issues on certain operating systems (especially Windows).
        shutil.copy(str(parquet_path), str(standard_path))

        # ── Execute 03_validate_topics.py + Capture stdout ────────────
        validate_mod = load_validate_module()

        log_file = RESULTS_DIR / f"03_validation_full_{combo_id}.txt"
        with open(log_file, "w", encoding="utf-8") as f:
            f.write(f"{'=' * 70}\n")
            f.write(f"  03_validate_topics.py Execution Results\n")
            f.write(f"  Combination ID: {combo_id}\n")
            f.write(f"  Parquet: {parquet_path}\n")
            f.write(f"  Execution Time: {datetime.now():%Y-%m-%d %H:%M:%S}\n")
            f.write(f"{'=' * 70}\n\n")

            # Redirect both standard output and standard error to the file
            try:
                with redirect_stdout(f), redirect_stderr(f):
                    validate_mod.inspect_topic_results()
            except SystemExit:
                # If 03_validate calls sys.exit() (on critical failure)
                f.write("\n[Validation Aborted — sys.exit called]\n")
            except Exception as e:
                f.write(f"\n[Error during validation]: {type(e).__name__}: {e}\n")

        return {"combo_id": combo_id, "log_file": str(log_file), "ok": True}

    finally:
        # ── Cleanup ──────────────────────────────────────────────────
        if standard_path.exists():
            standard_path.unlink()
        if backup_path and backup_path.exists():
            shutil.move(str(backup_path), str(standard_path))


# =============================================================================
# § 3. Batch Execute 03 Validation for All Combinations
# =============================================================================

def validate_all_combinations() -> None:
    """Performs 03 validation for all Parquet files in grid_search_results/."""
    print("=" * 70)
    print(f"  Batch Validation of Grid Search Results (03_validate_topics.py)")
    print(f"  Execution Time: {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("=" * 70)

    # Search for all combination Parquet files
    parquet_files = sorted(RESULTS_DIR.glob("ed_triage_with_topics_mcs*.parquet"))
    if not parquet_files:
        print(f"❌ Could not find grid result Parquet files in {RESULTS_DIR}/.")
        print(f"   Please run 02c_hdbscan_grid_search_v3.py first.")
        return

    print(f"\n📂 Combinations found: {len(parquet_files)}")

    total = len(parquet_files)
    for idx, pq_path in enumerate(parquet_files, start=1):
        # Extract combo_id from filename: ed_triage_with_topics_mcs500_ms30.parquet → mcs500_ms30
        combo_id = pq_path.stem.replace("ed_triage_with_topics_", "")

        log_file = RESULTS_DIR / f"03_validation_full_{combo_id}.txt"
        if log_file.exists():
            print(f"  [{idx}/{total}] {combo_id} — Already validated, skipping.")
            continue

        print(f"  [{idx}/{total}] Validating {combo_id}...")
        try:
            result = validate_single_combination(pq_path, combo_id)
            print(f"    ✅ → {result['log_file']}")
        except Exception as e:
            print(f"    ❌ ERROR: {e}")

    # ── Generate Comprehensive Ranking ────────────────────────────
    print("\n" + "=" * 70)
    print(f"  📊 Comprehensive Ranking")
    print("=" * 70)

    if SUMMARY_CSV.exists():
        df = pd.read_csv(SUMMARY_CSV)
        if len(df) > 0:
            df_sorted = df.sort_values("composite_score", ascending=False).head(15)
            print("\n[Top 15 Combinations — By Composite Score]")
            display_cols = [
                "combo_id", "min_cluster_size", "min_samples",
                "noise_ratio", "n_topics", "other_ratio",
                "n_critical_covered", "composite_score",
            ]
            available_cols = [c for c in display_cols if c in df_sorted.columns]
            print(df_sorted[available_cols].to_string(index=False))

            # Highlight the highest scoring combination
            best = df_sorted.iloc[0]
            print(f"\n🏆 Optimal Combination: {best['combo_id']}")
            print(f"   📂 Parquet: {best['parquet_path']}")
            print(f"   📝 03 Validation Log: "
                  f"{RESULTS_DIR}/03_validation_full_{best['combo_id']}.txt")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    validate_all_combinations()