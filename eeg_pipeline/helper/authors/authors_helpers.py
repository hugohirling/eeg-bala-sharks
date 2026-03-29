from __future__ import annotations

import logging
import os
from pathlib import Path

import mne
import numpy as np

from preprocessing import config
from preprocessing_authors import config_authors as authors_cfg

PIPELINE_STEPS = authors_cfg.PIPELINE_STEPS
STEP_OUTPUT_SUFFIXES = authors_cfg.STEP_OUTPUT_SUFFIXES


def resolve_output_dir() -> Path:
    configured = os.environ.get("EEG_AUTHORS_OUTPUT_DIR")
    if configured:
        path = Path(configured)
    else:
        path = Path(authors_cfg.DEFAULT_OUTPUT_DIR)
    path.mkdir(parents=True, exist_ok=True)
    (path / "qc").mkdir(parents=True, exist_ok=True)
    return path


def subject_tag(subject_id: str) -> str:
    return subject_id if subject_id.startswith("sub-") else f"sub-{subject_id}"


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def resolve_target_sfreq() -> int:
    raw = os.environ.get("EEG_AUTHORS_TARGET_SFREQ")
    if raw is not None:
        return int(raw)
    return int(authors_cfg.DOWNSAMPLE_TARGET_SFREQ)


def resolve_bin_duration() -> float:
    raw = os.environ.get("EEG_AUTHORS_BIN_DURATION")
    if raw is not None:
        return float(raw)
    return float(authors_cfg.TIME_BINNING.get("bin_duration", 0.25))


def resolve_interactive_noisy() -> bool:
    default = bool(authors_cfg.NOISY_CHANNEL_DETECTION.get("manual_review", False))
    return _env_bool("EEG_AUTHORS_INTERACTIVE_NOISY", default)


def resolve_initial_input(subject_id: str, logger: logging.Logger | None = None) -> Path:
    sub_tag = subject_tag(subject_id)

    eeg_dir = Path(config.BIDS_ROOT) / sub_tag / "eeg"
    if eeg_dir.exists():
        for pattern in ("*.fif", "*.bdf", "*.edf", "*.set"):
            matches = sorted(eeg_dir.glob(pattern))
            if matches:
                if logger is not None:
                    logger.info(f"Initial input for {sub_tag}: {matches[0]}")
                return matches[0]

    fallback = Path(config.OUTPUT_DIR) / f"{sub_tag}_downsampled.fif"
    if fallback.exists():
        if logger is not None:
            logger.info(f"Fallback initial input for {sub_tag}: {fallback}")
        return fallback

    raise FileNotFoundError(
        f"No input file found for {sub_tag}. Looked in {eeg_dir} and fallback {fallback}."
    )


def events_from_raw_or_synthetic(raw: mne.io.BaseRaw) -> np.ndarray:
    stim_channels = mne.pick_types(raw.info, stim=True)
    if len(stim_channels) > 0:
        return mne.find_events(raw, stim_channel=raw.ch_names[stim_channels[0]])
    if len(raw.annotations) > 0:
        events, _ = mne.events_from_annotations(raw)
        return events

    sfreq = raw.info["sfreq"]
    trial_duration_samples = int(5 * sfreq)
    samples = np.arange(0, raw.n_times, trial_duration_samples)
    return np.column_stack([samples, np.zeros_like(samples), np.ones_like(samples)])
