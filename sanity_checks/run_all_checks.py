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

from rich.progress import Progress, TaskID, SpinnerColumn, DownloadColumn, TimeElapsedColumn, TransferSpeedColumn, TextColumn, BarColumn, TimeRemainingColumn
from rich.console import Console
from rich.table import Table

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sanity_checks.scripts.helpers.sc_config import SCRIPTS_DIR, STEP_CHECK_FILES





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
        result = subprocess.run(
            cmd,
            cwd=str(SCRIPTS_DIR.parent.parent),
            env={**dict(subprocess.os.environ), **env_vars},
            capture_output=True,  # Capture output to avoid cluttering the progress bar
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
    # Set up Rich console and progress
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
            error_msg = f"Unknown step ID '{step_id}'. Valid: {', '.join(sorted(STEP_CHECK_FILES.keys()))}"
            console.print(f"[red]ERROR:[/red] {error_msg}")
            sys.exit(1)

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

    # Summary
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
        style = "green" if results[step_id] else "red"
        table.add_row(step_id, status)

    console.print(table)
    console.print(f"\n[bold]Total: {passed} passed, {failed} failed out of {len(results)}[/bold]")
    console.print("[bold blue]" + "=" * 80 + "[/bold blue]")

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
