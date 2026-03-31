from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = CURRENT_DIR.parent
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from preprocessing import config
from preprocessing_authors import config_authors as authors_cfg
from helper.general.helper_functions import get_previous_step_file
from helper.authors.authors_helpers import resolve_initial_input, subject_tag

STEP_FILE_MAP = authors_cfg.STEP_FILE_MAP
PIPELINE_STEPS = authors_cfg.PIPELINE_STEPS
RAW_FLOW_STEPS = authors_cfg.RAW_FLOW_STEPS
STEP_OUTPUT_SUFFIXES = authors_cfg.STEP_OUTPUT_SUFFIXES


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("authors_preprocessing")
    logger.setLevel(logging.INFO)

    if logger.handlers:
        return logger

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    log_dir = CURRENT_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_dir / "master_pipeline_authors.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run authors preprocessing pipeline with flexible step order.")
    parser.add_argument(
        "--steps",
        nargs="+",
        default=PIPELINE_STEPS,
        help="Step names to run in order.",
    )
    parser.add_argument(
        "--subjects",
        nargs="+",
        default=None,
        metavar="ID",
        help="Subject IDs (without sub- prefix), e.g. 01 02 03",
    )
    parser.add_argument(
        "--output-dir",
        default=str(authors_cfg.DEFAULT_OUTPUT_DIR),
        help="Output directory for authors pipeline.",
    )
    parser.add_argument(
        "--target-sfreq",
        type=int,
        default=int(authors_cfg.DOWNSAMPLE_TARGET_SFREQ),
        help="Target sampling frequency for downsample step.",
    )
    parser.add_argument(
        "--bin-duration",
        type=float,
        default=float(authors_cfg.TIME_BINNING.get("bin_duration", 0.25)),
        help="Time bin duration in seconds for binning step.",
    )
    parser.add_argument(
        "--interactive-noisy",
        action=argparse.BooleanOptionalAction,
        default=bool(authors_cfg.NOISY_CHANNEL_DETECTION.get("manual_review", False)),
        help="Enable interactive raw plot during noisy-channel detection.",
    )
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Skip input checks before executing steps.",
    )
    return parser.parse_args()


def _output_dir(path_like: str | Path) -> Path:
    out = Path(path_like)
    out.mkdir(parents=True, exist_ok=True)
    (out / "qc").mkdir(parents=True, exist_ok=True)
    return out


def _resolve_subjects(subjects_arg: list[str] | None) -> list[str]:
    if subjects_arg:
        return [s.replace("sub-", "") for s in subjects_arg]
    return list(config.SUBJECTS)


def _resolve_step_input(subject_id: str, step_name: str, output_dir: Path) -> Path | None:
    return get_previous_step_file(
        subject_id=subject_id,
        current_step=step_name,
        output_dir=output_dir,
        pipeline_steps=PIPELINE_STEPS,
        step_output_suffixes=STEP_OUTPUT_SUFFIXES,
    )


def preflight_check_for_subject(subject_id: str, steps: list[str], output_dir: Path, logger: logging.Logger) -> list[str]:
    messages: list[str] = []
    sub = subject_tag(subject_id)

    for step in steps:
        if step in RAW_FLOW_STEPS:
            input_path = _resolve_step_input(subject_id, step, output_dir)
            if input_path is None:
                try:
                    resolve_initial_input(subject_id, logger)
                except Exception as exc:
                    messages.append(f"{sub} | {step}: no initial input ({exc})")
                continue

            if not input_path.exists():
                messages.append(f"{sub} | {step}: missing previous-step input {input_path}")
            continue

        if step == "epoch":
            input_path = _resolve_step_input(subject_id, step, output_dir)
            if input_path is None or not input_path.exists():
                messages.append(f"{sub} | epoch: missing input {input_path}")
            continue

        if step == "baseline_correction_binning":
            for phase in ["decision", "response", "feedback"]:
                phase_file = output_dir / f"{sub}_{phase}_authors-epo.fif"
                if not phase_file.exists():
                    messages.append(f"{sub} | baseline_correction_binning: missing {phase_file}")

    return messages


def _build_step_command(step_name: str) -> list[str]:
    step_rel_path = STEP_FILE_MAP[step_name]
    step_path = PIPELINE_DIR / step_rel_path
    return [sys.executable, str(step_path)]


def run_pipeline(steps: list[str], logger: logging.Logger) -> int:
    for index, step_name in enumerate(steps, start=1):
        step_rel = STEP_FILE_MAP.get(step_name)
        if step_rel is None:
            logger.error(f"Unknown step name: {step_name}")
            return 1

        step_path = PIPELINE_DIR / step_rel
        if not step_path.exists():
            logger.error(f"Step file not found: {step_path}")
            return 1

        logger.info(f"[{index}/{len(steps)}] Running {step_name} ({step_rel})")
        result = subprocess.run(_build_step_command(step_name), cwd=str(PIPELINE_DIR))
        if result.returncode != 0:
            logger.error(f"Step failed with exit code {result.returncode}: {step_name}")
            return result.returncode

        logger.info(f"Completed {step_name}")

    logger.info("Authors master pipeline finished successfully.")
    return 0


def main() -> int:
    args = parse_args()
    logger = setup_logger()
    output_dir = _output_dir(args.output_dir)

    subjects = _resolve_subjects(args.subjects)
    if not subjects:
        logger.error("No subjects found. Check BIDS_ROOT in preprocessing/config.py or set EEG_SUBJECTS.")
        return 1

    invalid = [s for s in args.steps if s not in PIPELINE_STEPS]
    if invalid:
        logger.error(f"Invalid steps: {invalid}. Allowed: {PIPELINE_STEPS}")
        return 1

    # Propagate selected subjects and output dir to step scripts.
    os.environ["EEG_SUBJECTS"] = ",".join(subjects)
    os.environ["EEG_AUTHORS_OUTPUT_DIR"] = str(output_dir)
    os.environ["EEG_AUTHORS_TARGET_SFREQ"] = str(args.target_sfreq)
    os.environ["EEG_AUTHORS_BIN_DURATION"] = str(args.bin_duration)
    os.environ["EEG_AUTHORS_INTERACTIVE_NOISY"] = "1" if args.interactive_noisy else "0"

    if not args.skip_preflight:
        preflight_errors: list[str] = []
        for subject_id in subjects:
            preflight_errors.extend(preflight_check_for_subject(subject_id, args.steps, output_dir, logger))
        if preflight_errors:
            logger.error("Preflight failed for authors pipeline:")
            for msg in preflight_errors:
                logger.error(msg)
            logger.error("Use --skip-preflight if you want to run anyway.")
            return 1

    logger.info("Starting authors pipeline runner")
    logger.info(f"Subjects ({len(subjects)}): {[subject_tag(s) for s in subjects]}")
    logger.info(f"Steps: {args.steps}")

    return run_pipeline(args.steps, logger)


if __name__ == "__main__":
    raise SystemExit(main())
