from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

import config
from helper.helper_functions import get_previous_step_file

PIPELINE_STEPS = [
    "preprocessing/00_downsample.py",
    "preprocessing/01_split_players.py",
    "preprocessing/02_rename_set_montage.py",
    "preprocessing/03_bad_channels_detect.py",
    "preprocessing/04_interpolate_bad_channels.py",
    "preprocessing/05_filter.py",
    "preprocessing/06_ica.py",
    "preprocessing/07_epoch.py",
]

PERSON_SPECIFIC_STEPS = {
    "preprocessing/02_rename_set_montage.py",
    "preprocessing/03_bad_channels_detect.py",
    "preprocessing/04_interpolate_bad_channels.py",
    "preprocessing/05_filter.py",
    "preprocessing/06_ica.py",
    "preprocessing/07_epoch.py",
}


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


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("eeg_pipeline")
    logger.setLevel(logging.INFO)

    if logger.handlers:
        return logger

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def _input_requirements_for_step(step_file: str, subject_id: str) -> list[Path]:
    if step_file in PERSON_SPECIFIC_STEPS:
        persons = ["P1", "P2"]
    else:
        persons = [None]

    required_inputs: list[Path] = []
    for person in persons:
        input_path = get_previous_step_file(subject_id=subject_id, current_step=step_file, person=person)
        if input_path is None:
            continue
        required_inputs.append(input_path)
    return required_inputs


def preflight_check_for_step(step_file: str, logger: logging.Logger) -> list[str]:
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
        logger.error(f"Preflight failed for step {step_file}: required input files are missing.")
        for message in missing_messages:
            logger.error(message)

    return missing_messages


def run_pipeline(steps: list[str], logger: logging.Logger, skip_preflight: bool = False) -> int:
    for index, step_file in enumerate(steps, start=1):
        step_path = CURRENT_DIR / step_file
        if not step_path.exists():
            logger.error(f"Step file not found: {step_path}")
            return 1

        if not skip_preflight:
            missing = preflight_check_for_step(step_file, logger)
            if missing:
                logger.error(f"Stopping pipeline before {step_file} due to missing inputs.")
                return 1

        logger.info(f"[{index}/{len(steps)}] Running {step_file}")
        result = subprocess.run([sys.executable, str(step_path)], cwd=str(CURRENT_DIR))
        if result.returncode != 0:
            logger.error(f"Step failed with exit code {result.returncode}: {step_file}")
            return result.returncode

        logger.info(f"Completed {step_file}")

    logger.info("Pipeline completed successfully.")
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
    args = parse_args()
    logger = setup_logger()

    steps = args.steps

    all_subjects = config.SUBJECTS
    if not all_subjects:
        logger.error("No subjects found. Check BIDS_ROOT in config.py or set EEG_SUBJECTS env var.")
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

    logger.info("Starting EEG pipeline runner")
    logger.info(f"Subjects ({len(chosen)}): {config.SUBJECTS}")
    logger.info(f"Steps: {steps}")

    return run_pipeline(steps, logger, skip_preflight=args.skip_preflight)


if __name__ == "__main__":
    raise SystemExit(main())
