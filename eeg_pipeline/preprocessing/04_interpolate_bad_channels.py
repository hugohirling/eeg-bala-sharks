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


def interpolate_bad_channels(subject_id):
    """
    Interpolates bad channels for a subject.

    This function loads the raw data, identifies bad channels (only A and B channels),
    interpolates them, and saves the updated data.

    Args:
        subject_id (str): The subject identifier.

    Returns:
        list: List of tuples (person, out_path).
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
            raise ValueError("Interpolation step requires renamed+montaged input files")

        LOGGER.info(f"Loading previous step file ({person}): {path_in}")
        raw = mne.io.read_raw_fif(path_in, preload=True)

        # Get bad channels and filter to only EEG channels (A and B), exclude C and D
        bad_channels = list(raw.info.get("bads", []))
        bad_channels = [ch for ch in bad_channels if ch in raw.ch_names]
        
        # Only interpolate A and B channels (actual EEG), skip C and D
        valid_bad_channels = [ch for ch in bad_channels if ch.startswith(("A", "B"))]
        
        if valid_bad_channels:
            raw.info["bads"] = valid_bad_channels
            raw.interpolate_bads(reset_bads=True)
            LOGGER.info(f"Interpolated bad channels for {subject_id} {person}: {', '.join(valid_bad_channels)}")
        else:
            raw.info["bads"] = []
            LOGGER.info(f"No bad A/B channels to interpolate for {subject_id} {person}.")

        out_path = save_current_step_file(
            raw,
            subject_id,
            __file__,
            person=person,
            step_output_suffixes=config.STEP_OUTPUT_SUFFIXES,
        )
        LOGGER.info(f"Saved interpolated file ({person}) to: {out_path}")
        outputs.append((person, out_path))

    return outputs


if __name__ == "__main__":
    for i, subj in enumerate(config.SUBJECTS):
        interpolate_bad_channels(subj)
        LOGGER.info(f"PROGRESS:{i + 1}")
