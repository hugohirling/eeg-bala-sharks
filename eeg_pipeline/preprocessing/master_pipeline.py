# This file has been commented using GitHub Copilot with the Grok Code Fast 1 model.

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path
import mne
from rich.console import Console
from rich.live import Live
from rich.logging import RichHandler
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn, TimeRemainingColumn

CURRENT_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = CURRENT_DIR.parent
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from preprocessing import config
from helper.general.helper_functions import get_previous_step_file

PIPELINE_STEPS = config.PIPELINE_STEPS
PERSON_SPECIFIC_STEPS = config.PERSON_SPECIFIC_STEPS


def _normalize_step_alias(value: str) -> str:
    token = value.strip().lower().replace("\\", "/")
    token = token.replace(".py", "")
    token = token.replace("preprocessing/", "")
    if "_" in token and token.split("_", 1)[0].isdigit():
        token = token.split("_", 1)[1]
    alias_map = {
        "downsampling": "downsample",
        "filtering": "filter",
        "epoching": "epoch",
        "interpolate": "interpolate_bad_channels",
        "badchannels": "bad_channels_detect",
    }
    return alias_map.get(token, token)


def _build_step_lookup(available_steps: list[str]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for step in available_steps:
        path_token = step.replace("\\", "/")
        basename = Path(step).name
        stem = Path(step).stem
        short = stem.split("_", 1)[1] if "_" in stem and stem.split("_", 1)[0].isdigit() else stem

        aliases = {
            path_token,
            path_token.lower(),
            basename,
            basename.lower(),
            stem,
            stem.lower(),
            short,
            short.lower(),
            _normalize_step_alias(path_token),
            _normalize_step_alias(basename),
            _normalize_step_alias(stem),
            _normalize_step_alias(short),
        }
        for alias in aliases:
            lookup[alias] = step
    return lookup


def _resolve_step_tokens(step_tokens: list[str], available_steps: list[str], *, arg_name: str) -> list[str]:
    lookup = _build_step_lookup(available_steps)
    resolved: list[str] = []
    for token in step_tokens:
        key = _normalize_step_alias(token)
        if key in lookup:
            resolved.append(lookup[key])
            continue
        key = token.replace("\\", "/")
        if key in lookup:
            resolved.append(lookup[key])
            continue
        valid = ", ".join(available_steps)
        raise ValueError(f"Unknown value in {arg_name}: '{token}'. Valid step files: {valid}")

    seen: set[str] = set()
    unique = [step for step in resolved if not (step in seen or seen.add(step))]
    return unique


def _apply_skip_steps(selected_steps: list[str], skip_steps: list[str]) -> list[str]:
    skip_set = set(skip_steps)
    return [step for step in selected_steps if step not in skip_set]


def prompt_subject_selection(all_subjects: list[str]) -> list[str]:
    """
    Interactively prompts the user to select a subset of subjects from the available list.

    This function displays all available subjects with their indices, then allows the user
    to input selections in various formats: 'all' for all subjects, ranges like '1-5',
    or comma-separated indices like '1,3,5'. It validates the input and returns a
    deduplicated list of selected subject IDs.

    Args:
        all_subjects (list[str]): List of all available subject IDs (e.g., ['01', '02']).

    Returns:
        list[str]: List of selected subject IDs in the order they appear in all_subjects,
                   without duplicates.
    """
    """Show available subjects and let the user pick a subset interactively."""
    # Display available subjects with indices
    print(f"\nAvailable subjects ({len(all_subjects)} total):")
    for i, s in enumerate(all_subjects, start=1):
        print(f"  {i:>3}: sub-{s}")
    print()
    # Show selection instructions
    print("Select subjects to process:")
    print("  all        → process all subjects")
    print("  1-5        → subjects 1 through 5 (by position above)")
    print("  1,3,5      → specific positions")
    print()

    # Input loop for user selection
    while True:
        raw = input("Your selection: ").strip()
        if not raw:
            continue

        if raw.lower() == "all":
            return all_subjects

        selected: list[str] = []
        try:
            # Parse comma-separated parts
            for part in raw.split(","):
                part = part.strip()
                if "-" in part:
                    # Handle range selection
                    start_s, end_s = part.split("-", 1)
                    start, end = int(start_s), int(end_s)
                    if not (1 <= start <= end <= len(all_subjects)):
                        raise ValueError
                    selected.extend(all_subjects[start - 1 : end])
                else:
                    # Handle single index
                    idx = int(part)
                    if not (1 <= idx <= len(all_subjects)):
                        raise ValueError
                    selected.append(all_subjects[idx - 1])
        except (ValueError, IndexError):
            print(f"  Invalid input. Please use positions between 1 and {len(all_subjects)}.")
            continue

        # deduplicate while preserving order
        seen: set[str] = set()
        unique = [s for s in selected if not (s in seen or seen.add(s))]  # type: ignore[func-returns-value]
        print(f"  → Selected {len(unique)} subject(s): {', '.join('sub-' + s for s in unique)}")
        return unique


# Shared Rich console for consistent terminal output across progress bars and logging
CONSOLE = Console()


def setup_logger() -> logging.Logger:
    """
    Sets up and returns a shared logger for the EEG pipeline.

    This function configures a logger with RichHandler for enhanced console output
    in the main process, and falls back to a standard StreamHandler if Rich is not
    available. It detects if running in a subprocess via environment variable and
    adjusts the handler accordingly to avoid conflicts.

    Returns:
        logging.Logger: Configured logger instance for the pipeline.
    """
    logger = logging.getLogger("eeg_pipeline")
    logger.setLevel(logging.INFO)

    if logger.handlers:
        return logger

    try:
        # Use RichHandler for main process with enhanced formatting
        handler = RichHandler(
            console=CONSOLE,
            rich_tracebacks=True,
            markup=True,
            show_time=True,
            show_level=True,
            show_path=False,
        )
        formatter = logging.Formatter("%(message)s")
        handler.setFormatter(formatter)
    except ImportError:
        # Fallback to standard handler if Rich not available
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(formatter)

    logger.addHandler(handler)
    return logger

# Expose a shared module-level logger for step scripts.
LOGGER = setup_logger()

def setup_pipeline_progress() -> Progress:
    """
    Creates and configures a Rich Progress instance for pipeline execution tracking.

    This function sets up a progress bar with columns for description, progress bar,
    completion count, elapsed time, and remaining time. It uses the shared console
    for consistent output.

    Returns:
        Progress: Configured Progress instance for tracking pipeline steps.
    """
    progress_bar = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
        console=CONSOLE,
        transient=False,
        refresh_per_second=10,
    )
    return progress_bar

PIPELINE_PROGRESS = setup_pipeline_progress()


def _input_requirements_for_step(step_file: str, subject_id: str) -> list[Path]:
    """
    Determines the required input files for a given pipeline step and subject.

    This helper function checks if the step is person-specific (e.g., for P1 and P2)
    and constructs the expected input file paths based on the previous step's output.
    It uses the helper function to find the correct file paths.

    Args:
        step_file (str): The filename of the current pipeline step.
        subject_id (str): The subject ID (e.g., '01').

    Returns:
        list[Path]: List of required input file paths for the step.
    """
    if step_file in PERSON_SPECIFIC_STEPS:
        persons = ["P1", "P2"]
    else:
        persons = [None]

    required_inputs: list[Path] = []
    for person in persons:
        input_path = get_previous_step_file(
            subject_id=subject_id,
            current_step=step_file,
            person=person,
            pipeline_steps=config.PIPELINE_STEPS,
            step_output_suffixes=config.STEP_OUTPUT_SUFFIXES,
        )
        if input_path is None:
            continue
        required_inputs.append(input_path)
    return required_inputs


def preflight_check_for_step(step_file: str) -> list[str]:
    """
    Performs a preflight check to ensure all required input files exist for a pipeline step.

    This function iterates over all subjects and checks if the necessary input files
    for the given step are present. It logs errors for missing files and returns
    a list of error messages.

    Args:
        step_file (str): The filename of the pipeline step to check.

    Returns:
        list[str]: List of error messages for missing input files. Empty if all inputs exist.
    """
    missing_messages: list[str] = []

    for subject_id in config.SUBJECTS:
        required_inputs = _input_requirements_for_step(step_file, subject_id)
        for input_path in required_inputs:
            if input_path.exists():
                continue
            missing_messages.append(
                f"Missing input for step {step_file}, subject {subject_id}: {input_path}"
            )

    if missing_messages:
        LOGGER.error(f"Preflight failed for step {step_file}: required input files are missing.")
        for message in missing_messages:
            LOGGER.error(message)

    return missing_messages


def run_pipeline(steps: list[str], skip_preflight: bool = False) -> int:
    """
    Executes the EEG preprocessing pipeline with progress tracking and logging.

    This function runs each pipeline step in sequence, displaying progress bars for
    both pipeline steps and subjects processed within each step. It captures real-time
    output from subprocesses, parses PROGRESS messages for subject updates, and logs
    other output. Preflight checks are performed unless skipped.

    Args:
        steps (list[str]): List of pipeline step filenames to execute in order.
        skip_preflight (bool): If True, skips input existence checks before each step.

    Returns:
        int: Exit code (0 for success, non-zero for failure).
    """
    progress = setup_pipeline_progress()
    with Live(progress, console=CONSOLE, refresh_per_second=1) as live:
        # Initialize progress tasks for pipeline and subjects
        progress_task_id = progress.add_task("Pipeline progress", total=len(steps))
        progress_task_id_subjects = progress.add_task("Subjects", total=len(config.SUBJECTS))

        for index, step_file in enumerate(steps, start=1):
            step_path = PIPELINE_DIR / step_file
            if not step_path.exists():
                LOGGER.error(f"Step file not found: {step_path}")
                return 1

            # Reset subject progress for this step
            progress.update(progress_task_id_subjects, completed=0, time_elapsed=0)
            progress.update(progress_task_id, description=f"{index}/{len(steps)} {step_file}")
            live.refresh()

            if not skip_preflight:
                missing = preflight_check_for_step(step_file)
                if missing:
                    LOGGER.error(f"Stopping pipeline before {step_file} due to missing inputs.")
                    return 1

            LOGGER.info(f"[{index}/{len(steps)}] Running {step_file}")
            # Run the step as a subprocess to capture output
            proc = subprocess.Popen(
                [sys.executable, str(step_path)],
                cwd=str(PIPELINE_DIR),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=dict(os.environ, EEG_SUBPROCESS="1"),
            )
            # Read subprocess output line by line
            while True:
                line = proc.stdout.readline()
                if not line:
                    break
                line = line.rstrip()
                if line.startswith("PROGRESS:"):
                    # Parse subject progress updates
                    try:
                        idx = int(line.split(":")[1])
                        progress.update(progress_task_id_subjects, completed=idx)
                    except ValueError:
                        pass
                else:
                    # Log other output
                    LOGGER.info(line)
            proc.wait()
            if proc.returncode != 0:
                LOGGER.error(f"Step failed with exit code {proc.returncode}: {step_file}")
                return proc.returncode

            # Ensure subject progress is complete
            progress.update(progress_task_id_subjects, completed=len(config.SUBJECTS))

            progress.advance(progress_task_id)
            live.refresh()

            LOGGER.info(f"Completed {step_file}")

    LOGGER.info("Pipeline completed successfully.")
    return 0


def parse_args() -> argparse.Namespace:
    """
    Parses command-line arguments for the pipeline runner.

    This function sets up an ArgumentParser with options for specifying steps,
    skipping preflight checks, and selecting subjects.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Run EEG preprocessing pipeline with preflight checks.")
    parser.add_argument(
        "--steps",
        nargs="+",
        default=PIPELINE_STEPS,
        help="Optional subset of step filenames to run in order.",
    )
    parser.add_argument(
        "--skip-steps",
        nargs="+",
        default=[],
        help=(
            "Steps to skip. Accepts full paths or short names like downsample, "
            "bad_channels_detect, filter, ica, epoch."
        ),
    )
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Skip input existence checks before running steps.",
    )
    parser.add_argument(
        "--subjects",
        nargs="+",
        default=None,
        metavar="ID",
        help="Subject IDs to process (e.g. 01 02 03). Skips interactive prompt.",
    )
    return parser.parse_args()


def main() -> int:
    """
    Main entry point for the EEG preprocessing pipeline.

    This function handles argument parsing, subject selection (interactive or via args),
    sets up environment variables for subprocesses, and runs the pipeline.

    Returns:
        int: Exit code (0 for success, non-zero for failure).
    """

    mne.set_config("MNE_LOGGING_LEVEL", "ERROR")  # suppress verbose MNE info logs

    args = parse_args()

    try:
        selected_steps = _resolve_step_tokens(args.steps, PIPELINE_STEPS, arg_name="--steps")
        skip_steps = _resolve_step_tokens(args.skip_steps, PIPELINE_STEPS, arg_name="--skip-steps") if args.skip_steps else []
    except ValueError as exc:
        LOGGER.error(str(exc))
        return 1

    steps = _apply_skip_steps(selected_steps, skip_steps)
    if not steps:
        LOGGER.error("No steps left to run after applying --skip-steps.")
        return 1

    all_subjects = config.SUBJECTS
    if not all_subjects:
        LOGGER.error("No subjects found. Check BIDS_ROOT in preprocessing/config.py or set EEG_SUBJECTS env var.")
        return 1

    if args.subjects:
        chosen = args.subjects
        print(f"Using subjects from --subjects flag: {', '.join('sub-' + s for s in chosen)}")
    else:
        chosen = prompt_subject_selection(all_subjects)

    # Propagate selection to all subprocess steps via the existing env-var mechanism
    os.environ["EEG_SUBJECTS"] = ",".join(chosen)
    # Also update in-process config so preflight checks use the same set
    config.SUBJECTS = chosen

    LOGGER.info("Starting EEG pipeline runner")
    LOGGER.info(f"Subjects ({len(chosen)}): {config.SUBJECTS}")
    LOGGER.info(f"Steps: {steps}")
    if skip_steps:
        LOGGER.info(f"Skipped steps: {skip_steps}")

    return run_pipeline(steps, skip_preflight=args.skip_preflight)


if __name__ == "__main__":
    # Standard Python entry point guard to run main() when script is executed directly
    raise SystemExit(main())
