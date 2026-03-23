from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

import config
from helper.helper_functions import get_previous_step_file

PIPELINE_STEPS = [
    "00_downsample.py",
    "01_split_players.py",
    "02_rename_set_montage.py",
    "03_bad_channels_detect.py",
    "03_interpolate_bad_channels.py",
    "01_filter.py",
    "04_ica.py",
    "05_epoch.py",
]

PERSON_SPECIFIC_STEPS = {
    "02_rename_set_montage.py",
    "03_bad_channels_detect.py",
    "03_interpolate_bad_channels.py",
    "01_filter.py",
    "04_ica.py",
    "05_epoch.py",
}


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
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logger = setup_logger()

    steps = args.steps

    logger.info("Starting EEG pipeline runner")
    logger.info(f"Subjects: {config.SUBJECTS}")
    logger.info(f"Steps: {steps}")

    return run_pipeline(steps, logger, skip_preflight=args.skip_preflight)


if __name__ == "__main__":
    raise SystemExit(main())
