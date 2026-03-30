from __future__ import annotations

import argparse
from concurrent.futures import process
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


def prompt_subject_selection(all_subjects: list[str]) -> list[str]:
    """Show available subjects and let the user pick a subset interactively."""
    print(f"\nAvailable subjects ({len(all_subjects)} total):")
    for i, s in enumerate(all_subjects, start=1):
        print(f"  {i:>3}: sub-{s}")
    print()
    print("Select subjects to process:")
    print("  all        → process all subjects")
    print("  1-5        → subjects 1 through 5 (by position above)")
    print("  1,3,5      → specific positions")
    print()

    while True:
        raw = input("Your selection: ").strip()
        if not raw:
            continue

        if raw.lower() == "all":
            return all_subjects

        selected: list[str] = []
        try:
            for part in raw.split(","):
                part = part.strip()
                if "-" in part:
                    start_s, end_s = part.split("-", 1)
                    start, end = int(start_s), int(end_s)
                    if not (1 <= start <= end <= len(all_subjects)):
                        raise ValueError
                    selected.extend(all_subjects[start - 1 : end])
                else:
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


# Shared Rich console here so progress and logging share the same terminal object.
CONSOLE = Console()


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("eeg_pipeline")
    logger.setLevel(logging.INFO)

    if logger.handlers:
        return logger

    try:
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
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(formatter)

    logger.addHandler(handler)
    return logger

# Expose a shared module-level logger for step scripts.
LOGGER = setup_logger()

def setup_pipeline_progress() -> Progress:
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
    progress = setup_pipeline_progress()
    with Live(progress, console=CONSOLE, refresh_per_second=1) as live:
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
            proc = subprocess.Popen(
                [sys.executable, str(step_path)],
                cwd=str(PIPELINE_DIR),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=dict(os.environ, EEG_SUBPROCESS="1"),
            )
            while True:
                line = proc.stdout.readline()
                if not line:
                    break
                line = line.rstrip()
                if line.startswith("PROGRESS:"):
                    try:
                        idx = int(line.split(":")[1])
                        progress.update(progress_task_id_subjects, completed=idx)
                    except ValueError:
                        pass
                else:
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
    parser = argparse.ArgumentParser(description="Run EEG preprocessing pipeline with preflight checks.")
    parser.add_argument(
        "--steps",
        nargs="+",
        default=PIPELINE_STEPS,
        help="Optional subset of step filenames to run in order.",
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

    mne.set_config("MNE_LOGGING_LEVEL", "ERROR")  # suppress verbose MNE info logs

    args = parse_args()

    steps = args.steps

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

    return run_pipeline(steps, skip_preflight=args.skip_preflight)


if __name__ == "__main__":
    raise SystemExit(main())
