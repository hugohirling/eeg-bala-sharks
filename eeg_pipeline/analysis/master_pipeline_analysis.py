"""
Master Analysis Pipeline
Automatically runs Behavioral, Time-Frequency, and Statistical testing.
"""

import sys
import subprocess
import argparse
from pathlib import Path

# Add root directory to sys.path to import paths
CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import paths

def run_script(script_path, args=[]):
    """Helper to run a python script via subprocess."""
    print(f"\n{'='*60}\nRunning: {script_path.name} {' '.join(args)}\n{'='*60}")
    cmd = [sys.executable, str(script_path)] + args
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\n[ERROR] {script_path.name} failed. Halting pipeline.")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Run full analysis pipeline.")
    parser.add_argument("--subjects", nargs="+", help="List of subjects (e.g. 01 02 03). If empty, runs all found.")
    args = parser.parse_args()

    # Find available preprocessed subjects
    preprocess_dir = paths.OUTPUT_DIR / "preprocessing"
    if not preprocess_dir.exists():
        print(f"Error: Preprocessing directory not found at {preprocess_dir}")
        sys.exit(1)

    available_subjects = set()
    for file in preprocess_dir.glob("*_epoch.fif"):
        # Extracts '01' from 'sub-01_P1_epoch.fif'
        subj = file.name.split('_')[0].replace('sub-', '')
        available_subjects.add(subj)
    
    available_subjects = sorted(list(available_subjects))

    if not available_subjects:
        print("No preprocessed subjects found.")
        sys.exit(1)

    # Determine which subjects to run
    subjects_to_run = args.subjects if args.subjects else available_subjects
    print(f"\nSubjects queued for analysis: {subjects_to_run}")

    # Define steps
    step_07 = CURRENT_DIR / "07_behavioral_analysis.py"
    step_09 = CURRENT_DIR / "09_time_frequency_analysis.py"
    step_10 = CURRENT_DIR / "10_statistical_testing.py"

    # Run Subject-Level Steps (07 and 09)
    for subj in subjects_to_run:
        print(f"\n>>> PROCESSING SUBJECT {subj} <<<")
        run_script(step_07, ["--subject", subj])
        run_script(step_09, ["--subject", subj])

    # Run Group-Level Step (10)
    print("\n>>> RUNNING GROUP STATISTICS <<<")
    run_script(step_10)

    print("\n" + "="*60)
    print("ANALYSIS PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()