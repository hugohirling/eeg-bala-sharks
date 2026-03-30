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
    LOGGER.info(f"Original sampling frequency: {raw.info['sfreq']} Hz")

    raw.resample(target_sfreq)
    LOGGER.info(f"Downsampled to {target_sfreq} Hz")
    return raw


def process_subject(subject_id):
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
    for i, subj in enumerate(config.SUBJECTS):
        process_subject(subj)
        LOGGER.info(f"PROGRESS:{i + 1}")
