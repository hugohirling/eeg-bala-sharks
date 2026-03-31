"""
Master Script: Run all sanity checks in sequence for all subjects.

This file acts as the modular entry point for our diagnostic pipeline. 
By decoupling checking logic into separate scripts and dispatching them 
via subprocesses, we guarantee isolated memory execution, mitigating the 
severe RAM constraints associated with hyperscanning EEG datasets.

Usage:
    python sanity_checks/run_all_checks.py [--subjects 01,02,03] [--steps 00,01,02,03,04,05,06,07]
"""
import argparse
import subprocess
import sys
from pathlib import Path
from datetime import datetime

from rich.progress import Progress, TaskID, SpinnerColumn, DownloadColumn, TimeElapsedColumn, TransferSpeedColumn, TextColumn, BarColumn, TimeRemainingColumn
from rich.console import Console
from rich.table import Table

# PARAMETER JUSTIFICATION:
# Using pathlib and relative paths globally guarantees cross-OS reproducibility.
# Hardcoded paths (e.g., "C:/Users/...") are strictly avoided to ensure the 
# pipeline runs flawlessly on another computer.
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sanity_checks.scripts.helpers.sc_config import SCRIPTS_DIR, STEP_CHECK_FILES

# SANITY CHECK: Ensure configuration directories exist before attempting execution
assert SCRIPTS_DIR.exists(), f"Sanity Check Failed: Missing Scripts Directory at {SCRIPTS_DIR}"
assert len(STEP_CHECK_FILES) > 0, "Sanity Check Failed: No step files configured in sc_config.py"


def run_sanity_check(step_id: str, subjects: str = None) -> bool:
    """
    Run one sanity-check step script and return True on success.

    Args:
        step_id (str): The step identifier (e.g., '00').
        subjects (str, optional): Comma-separated list of subject IDs.

    Returns:
        bool: True if the child script exited with code 0, False otherwise.
    """
    script_path = SCRIPTS_DIR / STEP_CHECK_FILES[step_id]

    if not script_path.exists():
        print(f"  ERROR: Script not found: {script_path}")
        return False

    # PARAMETER JUSTIFICATION:
    # We use sys.executable to ensure the subprocess uses the exact same Python 
    # virtual environment (and mne/mne-bids versions) as the parent script.
    cmd = [sys.executable, str(script_path)]

    env_vars = {}
    if subjects:
        env_vars["EEG_SUBJECTS"] = subjects

    try:
        # PARAMETER JUSTIFICATION:
        # capture_output=True is strictly enforced here. If child scripts dump 
        # matrices to stdout, it will completely corrupt the parent's 'rich' UI progress bars.
        result = subprocess.run(
            cmd,
            cwd=str(SCRIPTS_DIR.parent.parent),
            env={**dict(subprocess.os.environ), **env_vars},
            capture_output=True,  
            text=True,
        )

        if result.returncode == 0:
            return True
        else:
            print(f"ERROR: Step {step_id} exited with code {result.returncode}. Stderr: {result.stderr}")
            return False
    except Exception as e:
        print(f"ERROR: Exception running step {step_id}: {e}")
        return False


def main():
    console = Console()
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}", justify="right"),
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.1f}%",
        "•",
        DownloadColumn(binary_units=False),
        "•",
        TimeElapsedColumn(),
        "•",
        TimeRemainingColumn(),
        console=console,
        expand=True
    )

    parser = argparse.ArgumentParser(
        description="Run all sanity checks for EEG preprocessing pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    # PARAMETER JUSTIFICATION:
    # Giving the user the ability to filter by subject acts as a modularity safeguard.
    # Allowing tests on a single subject prevents 2-hour long bottleneck executions during debugging.
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

    if args.steps:
        steps_to_run = args.steps.split(",")
    else:
        steps_to_run = sorted(STEP_CHECK_FILES.keys())

    # SANITY CHECK: Validate user input dynamically against allowed dictionary keys
    for step_id in steps_to_run:
        assert step_id in STEP_CHECK_FILES, f"Sanity Check Failed: User requested invalid config key '{step_id}'"

    console.print("\n[bold green]" + "=" * 80 + "[/bold green]")
    console.print("[bold green]SANITY CHECK MASTER RUNNER[/bold green]")
    console.print("[bold green]" + "=" * 80 + "[/bold green]")
    console.print(f"Timestamp: {datetime.now().isoformat()}")
    console.print(f"Steps to run: {', '.join(steps_to_run)}")
    if args.subjects:
        console.print(f"Subjects: {args.subjects}")
    else:
        console.print("Subjects: ALL")
    console.print("[bold green]" + "=" * 80 + "[/bold green]")

    results = {}
    with progress:
        overall_task = progress.add_task("Running sanity checks...", total=len(steps_to_run))
        for step_id in steps_to_run:
            step_task = progress.add_task(f"Step {step_id}: {STEP_CHECK_FILES[step_id]}", total=1)
            success = run_sanity_check(step_id, args.subjects)
            results[step_id] = success

            if success:
                progress.update(step_task, completed=1, description=f"Step {step_id}: [PASS]")
            else:
                progress.update(step_task, completed=1, description=f"Step {step_id}: [FAIL]")

            progress.update(overall_task, advance=1)

    console.print("\n[bold blue]" + "=" * 80 + "[/bold blue]")
    console.print("[bold blue]SUMMARY[/bold blue]")
    console.print("[bold blue]" + "=" * 80 + "[/bold blue]")

    passed = sum(1 for v in results.values() if v)
    failed = sum(1 for v in results.values() if not v)

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Step", style="cyan", no_wrap=True)
    table.add_column("Status", style="green" if passed > failed else "red")

    for step_id in sorted(results.keys()):
        status = "✓ PASS" if results[step_id] else "✗ FAIL"
        table.add_row(step_id, status)

    console.print(table)
    console.print(f"\n[bold]Total: {passed} passed, {failed} failed out of {len(results)}[/bold]")
    console.print("[bold blue]" + "=" * 80 + "[/bold blue]")

    # System exit codes guarantee upstream CI/CD environments track reproducibility failures natively.
    sys.exit(0 if failed == 0 else 1)

if __name__ == "__main__":
    main()