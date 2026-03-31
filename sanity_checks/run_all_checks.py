# This file's comments were created with the help of GitHub Copilot using GPT-5.3-Codex.
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
# Ensure the project root is importable so package-style imports work
# when this script is launched directly from the repository root.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sanity_checks.scripts.helpers.sc_config import SCRIPTS_DIR, STEP_CHECK_FILES





def run_sanity_check(step_id: str, subjects: str = None) -> bool:
    """Run one sanity-check step script and return True on success.

    The step script is executed in a subprocess so each check stays isolated.
    This prevents one script from polluting the process state of another.
    """
    # Resolve the script file for the requested step identifier.
    script_path = SCRIPTS_DIR / STEP_CHECK_FILES[step_id]

    # Fail early with a clear message if the script is missing.
    if not script_path.exists():
        print(f"  ERROR: Script not found: {script_path}")
        return False

    # Build the command using the current Python interpreter to avoid
    # environment/version mismatch between parent and child processes.
    cmd = [sys.executable, str(script_path)]

    # Optional environment overrides passed to child scripts.
    # EEG_SUBJECTS lets every step script process only selected subjects.
    env_vars = {}
    if subjects:
        env_vars["EEG_SUBJECTS"] = subjects

    try:
        # Execute from repository root context expected by the sanity scripts.
        # Output is captured to keep Rich progress rendering clean.
        result = subprocess.run(
            cmd,
            cwd=str(SCRIPTS_DIR.parent.parent),
            env={**dict(subprocess.os.environ), **env_vars},
            capture_output=True,  # Capture output to avoid cluttering the progress bar
            text=True,
        )

        # Zero return code means the step completed successfully.
        if result.returncode == 0:
            return True
        else:
            # Include stderr from the child process for quick debugging.
            print(f"ERROR: Step {step_id} exited with code {result.returncode}. Stderr: {result.stderr}")
            return False
    except Exception as e:
        # Catch process-level failures (spawn issues, invalid paths, etc.).
        print(f"ERROR: Exception running step {step_id}: {e}")
        return False


def main():
    # Initialize Rich components for structured console output and progress UI.
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

    # Parse optional subject and step filters.
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

    # Determine which steps should run in this invocation.
    # If no explicit filter is provided, run all known steps in sorted order.
    if args.steps:
        steps_to_run = args.steps.split(",")
    else:
        steps_to_run = sorted(STEP_CHECK_FILES.keys())

    # Validate requested steps before any execution starts, so users get
    # immediate feedback on typos and the run does not partially execute.
    for step_id in steps_to_run:
        if step_id not in STEP_CHECK_FILES:
            error_msg = f"Unknown step ID '{step_id}'. Valid: {', '.join(sorted(STEP_CHECK_FILES.keys()))}"
            console.print(f"[red]ERROR:[/red] {error_msg}")
            sys.exit(1)

    # Run header: helps identify scope and timing of the current run.
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

    # Execute each step and track pass/fail outcome per step id.
    results = {}
    with progress:
        # Overall task tracks full run completion.
        overall_task = progress.add_task("Running sanity checks...", total=len(steps_to_run))
        for step_id in steps_to_run:
            # Per-step task gives immediate feedback for the active script.
            step_task = progress.add_task(f"Step {step_id}: {STEP_CHECK_FILES[step_id]}", total=1)
            success = run_sanity_check(step_id, args.subjects)
            results[step_id] = success

            # Update per-step status text when the subprocess finishes.
            if success:
                progress.update(step_task, completed=1, description=f"Step {step_id}: [PASS]")
            else:
                progress.update(step_task, completed=1, description=f"Step {step_id}: [FAIL]")

            # Advance the overall progress once this step is done.
            progress.update(overall_task, advance=1)

    # Print final summary block with table and totals.
    console.print("\n[bold blue]" + "=" * 80 + "[/bold blue]")
    console.print("[bold blue]SUMMARY[/bold blue]")
    console.print("[bold blue]" + "=" * 80 + "[/bold blue]")

    # Aggregate run-level pass/fail counts for a quick health snapshot.
    passed = sum(1 for v in results.values() if v)
    failed = sum(1 for v in results.values() if not v)

    # Build a compact status table per step.
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Step", style="cyan", no_wrap=True)
    table.add_column("Status", style="green" if passed > failed else "red")

    for step_id in sorted(results.keys()):
        # Keep the table human-readable with explicit pass/fail words.
        status = "✓ PASS" if results[step_id] else "✗ FAIL"
        style = "green" if results[step_id] else "red"
        table.add_row(step_id, status)

    console.print(table)
    console.print(f"\n[bold]Total: {passed} passed, {failed} failed out of {len(results)}[/bold]")
    console.print("[bold blue]" + "=" * 80 + "[/bold blue]")

    # Use process exit code for CI/automation compatibility.
    # 0 means all checks passed, 1 means at least one failed.
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
