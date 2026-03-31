"""
Master Script: Run all sanity checks in sequence for all subjects.

Usage:
    python sanity_checks/run_all_checks.py [--subjects 01,02,03] [--steps 00,01,02,03,04,05,06,07]

Examples:
    python sanity_checks/run_all_checks.py                  # All steps, all subjects
    python sanity_checks/run_all_checks.py --subjects 01    # All steps, subject 01 only
    python sanity_checks/run_all_checks.py --steps 03,05    # Steps 03 and 05 only
"""
import argparse
import subprocess
import sys
from pathlib import Path
from datetime import datetime

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sanity_checks.scripts.sc_config import SCRIPTS_DIR, STEP_CHECK_FILES


def run_sanity_check(step_id: str, subjects: str = None) -> bool:
    """Run a single sanity check script. Returns True if successful."""
    script_path = SCRIPTS_DIR / STEP_CHECK_FILES[step_id]

    if not script_path.exists():
        print(f"  ERROR: Script not found: {script_path}")
        return False

    cmd = [sys.executable, str(script_path)]
    env_vars = {}
    if subjects:
        env_vars["EEG_SUBJECTS"] = subjects

    try:
        print(f"\n{'='*80}")
        print(f"Running: Step {step_id} - {STEP_CHECK_FILES[step_id]}")
        print(f"{'='*80}")

        result = subprocess.run(
            cmd,
            cwd=str(SCRIPTS_DIR.parent.parent),
            env={**dict(subprocess.os.environ), **env_vars},
            capture_output=False,
        )

        if result.returncode == 0:
            print(f"✓ Step {step_id} completed successfully")
            return True
        else:
            print(f"⚠ Step {step_id} exited with code {result.returncode}")
            return False
    except Exception as e:
        print(f"ERROR running step {step_id}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Run all sanity checks for EEG preprocessing pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--subjects",
        type=str,
        default=None,
        help="Comma-separated subject IDs (e.g., '01,02,03'). Default: all subjects.",
    )
    parser.add_argument(
        "--steps",
        type=str,
        default=None,
        help="Comma-separated step IDs (e.g., '00,01,02,03'). Default: all steps.",
    )

    args = parser.parse_args()

    # Determine which steps to run
    if args.steps:
        steps_to_run = args.steps.split(",")
    else:
        steps_to_run = sorted(STEP_CHECK_FILES.keys())

    # Validate step IDs
    for step_id in steps_to_run:
        if step_id not in STEP_CHECK_FILES:
            print(f"ERROR: Unknown step ID '{step_id}'. Valid: {', '.join(sorted(STEP_CHECK_FILES.keys()))}")
            sys.exit(1)

    print("\n" + "=" * 80)
    print("SANITY CHECK MASTER RUNNER")
    print("=" * 80)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Steps to run: {', '.join(steps_to_run)}")
    if args.subjects:
        print(f"Subjects: {args.subjects}")
    else:
        print(f"Subjects: ALL")
    print("=" * 80)

    results = {}
    for step_id in steps_to_run:
        success = run_sanity_check(step_id, args.subjects)
        results[step_id] = success

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    passed = sum(1 for v in results.values() if v)
    failed = sum(1 for v in results.values() if not v)

    for step_id in sorted(results.keys()):
        status = "✓ PASS" if results[step_id] else "✗ FAIL"
        print(f"  Step {step_id}: {status}")

    print(f"\nTotal: {passed} passed, {failed} failed out of {len(results)}")
    print("=" * 80)

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
