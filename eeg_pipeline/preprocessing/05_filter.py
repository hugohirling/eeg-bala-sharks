# This file has been commented using GitHub Copilot with the Grok Code Fast 1 model.

from pathlib import Path
import sys

import mne

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
from helper.general.helper_functions import get_step_io_files, save_current_step_file


def load_previous_step_data(subject_id):
    """
    Loads the previous step's data for a subject.

    Args:
        subject_id (str): The subject identifier.

    Returns:
        tuple: (raw, path_in)
    """
    path_in, _ = get_step_io_files(
        subject_id,
        __file__,
        pipeline_steps=config.PIPELINE_STEPS,
        step_output_suffixes=config.STEP_OUTPUT_SUFFIXES,
    )
    if path_in is None:
        raise ValueError("Filter step requires an input file from the previous pipeline step")

    LOGGER.info(f"Loading previous step file: {path_in}")
    return mne.io.read_raw_fif(path_in, preload=True), path_in


def filter_data(raw, l_freq=config.FREQ_LOWER, h_freq=config.FREQ_UPPER):
    """
    Applies bandpass filtering to the raw data.

    Args:
        raw (mne.io.Raw): The raw EEG data.
        l_freq (float): Low cutoff frequency.
        h_freq (float): High cutoff frequency.

    Returns:
        mne.io.Raw: The filtered raw data.
    """
    LOGGER.info(f"Applying bandpass filter: {l_freq} - {h_freq} Hz")
    raw.filter(l_freq=l_freq, h_freq=h_freq)
    return raw


def process_subject(subject_id):
    """
    Processes filtering for a subject, handling both P1 and P2 data.

    Args:
        subject_id (str): The subject identifier.

    Returns:
        list: List of tuples (person, path_in, out_path).
    """
    outputs = []
    for person in ["P1", "P2"]:
        path_in, _ = get_step_io_files(
            subject_id,
            __file__,
            person=person,
            pipeline_steps=config.PIPELINE_STEPS,
            step_output_suffixes=config.STEP_OUTPUT_SUFFIXES,
        )
        if path_in is None:
            raise ValueError("Filter step requires an input file from the previous pipeline step")

        LOGGER.info(f"Loading previous step file: {path_in}")
        raw = mne.io.read_raw_fif(path_in, preload=True)
        # Apply bandpass filtering
        raw = filter_data(raw)
        out_path = save_current_step_file(
            raw,
            subject_id,
            __file__,
            person=person,
            step_output_suffixes=config.STEP_OUTPUT_SUFFIXES,
        )
        LOGGER.info(f"Saved filtered file ({person}) to: {out_path}")
        outputs.append((person, path_in, out_path))
    return outputs


if __name__ == "__main__":
    for i, subj in enumerate(config.SUBJECTS):
        process_subject(subj)
        LOGGER.info(f"PROGRESS:{i + 1}")
