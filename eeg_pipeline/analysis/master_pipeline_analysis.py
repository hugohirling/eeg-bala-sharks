"""
Master Analysis Orchestration Pipeline

This module acts as the centralized execution hub for the downstream analysis 
phase of the EEG project. Instead of manually running individual scripts for 
dozens of subjects, this orchestrator automatically detects available processed 
data and sequentially triggers Behavioral Analysis, Time-Frequency Analysis, 
and Final Group-Level Statistical Testing.

By utilizing `subprocess`, it completely isolates each script's memory environment,
ensuring that a crash in one subject's Time-Frequency calculation does not 
silently corrupt the data of the following subjects.
"""

import sys
import subprocess
import argparse
from pathlib import Path

# =============================================================================
# PATH CONFIGURATION
# =============================================================================
# Dynamically resolve the absolute path to the project root so this script 
# can be executed from any terminal directory without breaking relative imports.
CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import paths

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def run_script(script_path, args=[]):
    """
    Subprocess wrapper to execute individual Python modules.
    
    This acts as a safety boundary. If an individual script (like script 09) 
    encounters a fatal mathematical error or out-of-memory exception, this 
    wrapper catches the non-zero exit code and pauses the overarching pipeline 
    so the researcher can investigate, rather than printing corrupted results.
    
    Parameters
    ----------
    script_path : pathlib.Path
        Absolute path to the target .py script.
    args : list of str
        Command line arguments to pass to the script (e.g., ['--subject', '01']).
    """
    print(f"\n{'='*60}\nRunning: {script_path.name} {' '.join(args)}\n{'='*60}")
    
    # Construct the command: `python target_script.py --arg1 val1`
    cmd = [sys.executable, str(script_path)] + args
    
    # Execute the command and wait for it to finish
    result = subprocess.run(cmd)
    
    # A return code of 0 means absolute success. Anything else is a crash.
    if result.returncode != 0:
        print(f"\n[FATAL ERROR] {script_path.name} failed with exit code {result.returncode}.")
        print("Halting the master pipeline to prevent cascading errors.")
        sys.exit(1)


# =============================================================================
# MAIN PIPELINE EXECUTION
# =============================================================================

def main():
    """
    Main execution sequence for the master pipeline.
    Handles CLI arguments, auto-discovers preprocessed subjects, and routes 
    the execution queue through the 07 -> 09 -> 10 analysis stages.
    """
    parser = argparse.ArgumentParser(description="Run the full automated analysis pipeline.")
    parser.add_argument(
        "--subjects", 
        nargs="+", 
        help="Space-separated list of subjects (e.g. 01 02 03). If left empty, runs all found subjects."
    )
    args = parser.parse_args()

    # ---------------------------------------------------------
    # 1. Automatic Subject Discovery
    # ---------------------------------------------------------
    preprocess_dir = paths.OUTPUT_DIR / "preprocessing"
    if not preprocess_dir.exists():
        print(f"[ERROR] Preprocessing directory not found at {preprocess_dir}")
        print("You must run the preprocessing pipeline (scripts 00-03) before analysis.")
        sys.exit(1)

    available_subjects = set()
    # Hunt for valid MNE Epoch files to determine who is ready for analysis
    for file in preprocess_dir.glob("*_epoch.fif"):
        # Extract purely the numerical ID: 'sub-01_P1_epoch.fif' -> '01'
        subj = file.name.split('_')[0].replace('sub-', '')
        available_subjects.add(subj)
    
    # Sort subjects numerically for clean console output
    available_subjects = sorted(list(available_subjects))

    if not available_subjects:
        print("[ERROR] No preprocessed 'epoch.fif' subjects found. Pipeline cannot continue.")
        sys.exit(1)

    # ---------------------------------------------------------
    # 2. Queue Construction
    # ---------------------------------------------------------
    # Use explicitly requested subjects if provided, otherwise default to all found
    subjects_to_run = args.subjects if args.subjects else available_subjects
    print(f"\n[INFO] Subjects queued for analysis: {subjects_to_run}")

    # Map the absolute paths to the target scripts
    step_07 = CURRENT_DIR / "07_behavioral_analysis.py"
    step_09 = CURRENT_DIR / "09_time_frequency_analysis.py"
    step_10 = CURRENT_DIR / "10_statistical_testing.py"

    # ---------------------------------------------------------
    # 3. Subject-Level Processing
    # ---------------------------------------------------------
    # These scripts MUST run on a per-subject loop to generate individual metrics
    for subj in subjects_to_run:
        print(f"\n>>> PROCESSING SUBJECT {subj} <<<")
        run_script(step_07, ["--subject", subj])  # Behavioral Markov/Heuristics
        run_script(step_09, ["--subject", subj])  # Time-Frequency (Morlet Wavelets)

    # ---------------------------------------------------------
    # 4. Group-Level Processing
    # ---------------------------------------------------------
    # Once all individuals are processed, run the statistical scripts to aggregate 
    # the metrics into Pandas DataFrames and run overarching ANOVAs / T-Tests
    print("\n>>> RUNNING GROUP STATISTICS <<<")
    run_script(step_10)

    # Output Success Banner
    print("\n" + "="*60)
    print("🏆 ANALYSIS PIPELINE COMPLETED SUCCESSFULLY! 🏆")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()