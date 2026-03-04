"""
run_pipeline.py
===============
Master Pipeline Orchestrator

Executes all pipeline jobs in the correct dependency order to go from raw
listing CSV data to personalised Airbnb listing recommendations.

Execution Order & Dependencies:
  ┌──────────────────────────────────────────────────────────────────┐
  │ Step 1: listing_similarity.py   (no dependencies)               │
  │ Step 2: metadata.py             (no dependencies)               │
  │ Step 3: simulate_txns.py        (depends on step 1)             │
  │ Step 4: ones_n_zeros.py         (depends on step 3)             │
  │ Step 5: cohorts.py              (depends on step 4)             │
  │ Step 6: affinities.py           (depends on steps 2, 3, 5)     │
  │ Step 7: feature_stitching.py    (depends on steps 2, 4, 6)     │
  │ Step 8: libsvm.py               (depends on step 7)             │
  │ Step 9: FM.py                   (depends on step 8)             │
  └──────────────────────────────────────────────────────────────────┘

Usage:
    python jobs/run_pipeline.py              # run all steps
    python jobs/run_pipeline.py --start 4    # resume from step 4 onward
    python jobs/run_pipeline.py --only 5     # run only step 5

Note: Each step launches a separate Python subprocess so that Spark
sessions are fully isolated and resources are released between steps.
"""

import subprocess
import sys
import time
import argparse
import os

# ---------------------------------------------------------------------------
# Pipeline definition – ordered list of (step_number, script, description)
# ---------------------------------------------------------------------------
PIPELINE_STEPS = [
    (1, "jobs/listing_similarity.py", "Compute pairwise listing similarities"),
    (2, "jobs/metadata.py",           "Generate listing metadata (bins & categories)"),
    (3, "jobs/simulate_txns.py",      "Simulate synthetic users and transactions"),
    (4, "jobs/ones_n_zeros.py",       "Create labelled training & scoring examples"),
    (5, "jobs/cohorts.py",            "Assign RFM cohorts via K-Means clustering"),
    (6, "jobs/affinities.py",         "Compute user–metadata affinity scores"),
    (7, "jobs/feature_stitching.py",  "Assemble & stitch all feature families"),
    (8, "jobs/libsvm.py",             "Convert features to LIBSVM format"),
    (9, "jobs/FM.py",                 "Train Factorization Machine & generate recs"),
]


def run_step(step_num: int, script: str, description: str) -> bool:
    """Run a single pipeline step as a subprocess.

    Args:
        step_num:    The ordinal step number (for display purposes).
        script:      Relative path to the Python script.
        description: Human-readable description of what the step does.

    Returns:
        True if the step completed successfully, False otherwise.
    """
    separator = "=" * 70
    print(f"\n{separator}")
    print(f"  STEP {step_num}/9: {description}")
    print(f"  Script: {script}")
    print(separator)

    start_time = time.time()

    try:
        result = subprocess.run(
            [sys.executable, script],
            check=True,
            text=True,
        )
        elapsed = time.time() - start_time
        print(f"  ✓ Step {step_num} completed successfully in {elapsed:.1f}s")
        return True

    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"  ✗ Step {step_num} FAILED after {elapsed:.1f}s (exit code {e.returncode})")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Run the Airbnb Recommendation Engine pipeline."
    )
    parser.add_argument(
        "--start",
        type=int,
        default=1,
        help="Step number to start from (default: 1). Skips earlier steps.",
    )
    parser.add_argument(
        "--only",
        type=int,
        default=None,
        help="Run only this single step number.",
    )
    args = parser.parse_args()

    # Ensure we're running from the project root directory
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)
    print(f"Working directory: {os.getcwd()}")

    # Determine which steps to run
    if args.only is not None:
        steps_to_run = [(n, s, d) for n, s, d in PIPELINE_STEPS if n == args.only]
        if not steps_to_run:
            print(f"Error: step {args.only} not found. Valid steps are 1–9.")
            sys.exit(1)
    else:
        steps_to_run = [(n, s, d) for n, s, d in PIPELINE_STEPS if n >= args.start]

    print(f"\n{'#' * 70}")
    print(f"  AIRBNB RECOMMENDATION ENGINE PIPELINE")
    print(f"  Steps to run: {[n for n, _, _ in steps_to_run]}")
    print(f"{'#' * 70}")

    total_start = time.time()
    failed_steps = []

    for step_num, script, description in steps_to_run:
        success = run_step(step_num, script, description)
        if not success:
            failed_steps.append(step_num)
            print(f"\n  Pipeline halted at step {step_num}. Fix the error and re-run with:")
            print(f"    python jobs/run_pipeline.py --start {step_num}")
            sys.exit(1)

    total_elapsed = time.time() - total_start
    print(f"\n{'#' * 70}")
    print(f"  PIPELINE COMPLETE — all {len(steps_to_run)} steps finished in {total_elapsed:.1f}s")
    print(f"{'#' * 70}\n")


if __name__ == "__main__":
    main()
