"""
CLI and subject-selection helpers for sanity-check scripts.

REASONING:
- Purpose: keep argument parsing and subject normalization reusable without overloading the configuration module.
- Reproducibility: all step scripts interpret `--subjects` and visualization defaults in the same way.
- Parameter notes: visualization entrypoints default to a small subset for speed, while textual checks default to all configured subjects.
"""

from __future__ import annotations

import argparse
from typing import Iterable, List, Sequence

from helpers.sc_config import DEFAULT_VIS_SUBJECT_LIMIT


def normalize_subject_id(subject_id: str) -> str:
    value = str(subject_id).strip()
    if not value:
        return value
    return value.zfill(2) if value.isdigit() else value


def parse_subjects_csv(subject_str: str | None) -> List[str]:
    if not subject_str:
        return []
    return [normalize_subject_id(part) for part in subject_str.split(",") if part.strip()]


def resolve_subjects(
    subject_str: str | None,
    configured_subjects: Sequence[str] | Iterable[str],
    *,
    mode: str = "check",
    viz_default_limit: int = DEFAULT_VIS_SUBJECT_LIMIT,
) -> List[str]:
    parsed = parse_subjects_csv(subject_str)
    if parsed:
        return parsed

    configured = [normalize_subject_id(subject_id) for subject_id in configured_subjects]
    if mode == "check":
        return configured
    return configured[:viz_default_limit]


def add_subjects_argument(parser: argparse.ArgumentParser, *, viz_default_limit: int = DEFAULT_VIS_SUBJECT_LIMIT) -> None:
    parser.add_argument(
        "--subjects",
        type=str,
        default=None,
        help=f"Comma-separated subject IDs. Default: all subjects for checks, first {viz_default_limit} for visualizations.",
    )


def add_mode_argument(parser: argparse.ArgumentParser, *, default: str = "check") -> None:
    parser.add_argument(
        "--mode",
        type=str,
        default=default,
        choices=["check", "viz", "both"],
        help="Run the textual check, the visualization, or both.",
    )


def add_duration_argument(parser: argparse.ArgumentParser, *, default: int) -> None:
    parser.add_argument(
        "--duration",
        type=int,
        default=default,
        help=f"Duration in seconds used for plots (default: {default})",
    )
