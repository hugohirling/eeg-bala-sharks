# This file has been commented using GitHub Copilot with the Grok Code Fast 1 model.

from pathlib import Path
import sys

from mne_bids import BIDSPath, read_raw_bids

CURRENT_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = CURRENT_DIR.parent
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s"
)

LOGGER = logging.getLogger(__name__)

from preprocessing import config
from helper.general.helper_functions import save_current_step_file


def load_data(subject_id):
    """
    Loads raw EEG data for a given subject from BIDS format.

    This function constructs a BIDSPath for the subject and task, reads the raw
    BIDS data, and loads it into memory.

    Args:
        subject_id (str): The subject identifier (e.g., '01').

    Returns:
        mne.io.Raw: The loaded raw EEG data.
    """
    LOGGER.info(f"Loading subject {subject_id}")

    bids_path = BIDSPath(
        subject=subject_id,
        task="RPS",
        datatype="eeg",
        suffix="eeg",
        root=config.BIDS_ROOT,
    )

    raw = read_raw_bids(bids_path)
    raw.load_data()
    return raw


def downsample_data(raw, target_sfreq=config.DOWNSAMPLE_SFREQ):
    """
    Downsamples the raw EEG data to a target sampling frequency.

    This function resamples the data to reduce the sampling rate, which can
    help with processing efficiency while preserving essential information.

    Args:
        raw (mne.io.Raw): The raw EEG data to downsample.
        target_sfreq (float): The target sampling frequency in Hz.

    Returns:
        mne.io.Raw: The downsampled raw data.
    """
    LOGGER.info(f"Original sampling frequency: {raw.info['sfreq']} Hz")

    raw.resample(target_sfreq)
    LOGGER.info(f"Downsampled to {target_sfreq} Hz")
    return raw


def process_subject(subject_id):
    """
    Processes a single subject by loading, downsampling, and saving the data.

    This function orchestrates the downsampling pipeline for one subject,
    from loading the raw data to saving the processed file.

    Args:
        subject_id (str): The subject identifier.

    Returns:
        Path: The path to the saved downsampled file.
    """
    raw = load_data(subject_id)
    raw = downsample_data(raw)
    out_path = save_current_step_file(
        raw,
        subject_id,
        __file__,
        step_output_suffixes=config.STEP_OUTPUT_SUFFIXES,
    )
    LOGGER.info(f"Saved downsampled file to: {out_path}")
    return out_path

if __name__ == "__main__":
    # Process all subjects and report progress
    for i, subj in enumerate(config.SUBJECTS):
        process_subject(subj)
        LOGGER.info(f"PROGRESS:{i + 1}")
