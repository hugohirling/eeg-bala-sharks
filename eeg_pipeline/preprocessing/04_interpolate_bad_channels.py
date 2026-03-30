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

        bad_channels = list(raw.info.get("bads", []))
        bad_channels = [ch for ch in bad_channels if ch in raw.ch_names]

        if bad_channels:
            raw.info["bads"] = bad_channels
            raw.interpolate_bads(reset_bads=True)
            LOGGER.info(f"Interpolated bad channels for {subject_id} {person}: {', '.join(bad_channels)}")
        else:
            LOGGER.info(f"No bad channels marked in Step 03 for {subject_id} {person}.")

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
