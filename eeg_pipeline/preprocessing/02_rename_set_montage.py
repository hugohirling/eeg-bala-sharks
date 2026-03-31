# This file has been commented using GitHub Copilot with the Grok Code Fast 1 model.

from pathlib import Path
import sys

import mne
import numpy as np
from scipy.io import loadmat

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


def _make_biosemi64_montage(raw):
    """
    Creates a BioSemi64 montage from a template MAT file.

    This function loads the BioSemi64 electrode positions from a MAT file,
    adjusts the coordinates, and creates a digitization montage.

    Args:
        raw (mne.io.Raw): The raw EEG data to set the montage for.

    Returns:
        mne.channels.DigMontage: The created montage.
    """
    mat_path = Path(config.BIOSEMI64_MAT_PATH)
    if not mat_path.exists():
        raise FileNotFoundError(f"Missing BioSemi template: {mat_path}")

    # Load electrode positions from MAT file
    mat_data = loadmat(str(mat_path))
    if "biosemi64" not in mat_data:
        raise KeyError("Variable 'biosemi64' not found in biosemi64.mat")

    positions = np.asarray(mat_data["biosemi64"], dtype=float)
    if positions.shape != (64, 3):
        raise ValueError(
            f"Expected biosemi64 shape (64, 3), got {positions.shape}"
        )

    # Scale positions to realistic head radius in meters
    positions = positions * 0.1

    ordered_labels = list(config.channel_labels.values())
    # Create position dictionary for channels present in raw data
    ch_pos = {
        label: tuple(positions[idx])
        for idx, label in enumerate(ordered_labels)
        if label in raw.ch_names
    }
    if not ch_pos:
        raise ValueError("No BioSemi labels matched the current raw channel names")

    LOGGER.info(f"Setting montage for {len(ch_pos)} channels based on biosemi64.mat template")
    return mne.channels.make_dig_montage(ch_pos=ch_pos, coord_frame="head")


def _rename_eeg_channels(raw, person):
    """
    Renames EEG channels based on the person prefix and maps to 10-20 system.

    This function strips the player prefix, removes non-EEG channels,
    and renames channels to standard 10-20 labels.

    Args:
        raw (mne.io.Raw): The raw EEG data.
        person (str): The person ('P1' or 'P2').
    """
    source_prefix = config.PLAYER_PREFIX_MAP[person]

    eeg_picks = mne.pick_types(raw.info, eeg=True)
    eeg_channels = [raw.ch_names[idx] for idx in eeg_picks]

    # Strip the player-specific prefix from channel names
    strip_prefix_map = {
        ch: ch[len(source_prefix):]
        for ch in eeg_channels
        if ch.startswith(source_prefix)
    }
    if strip_prefix_map:
        raw.rename_channels(strip_prefix_map)

    # Remove non-EEG channels starting with 'C' or 'D'
    channels_to_pick = [ch for ch in raw.ch_names if ch.startswith("A") or ch.startswith("B")]
    if channels_to_pick:
        raw.pick_channels(channels_to_pick)

    # Map remaining channels to 10-20 system labels
    mapping_1020 = {
        ch: config.channel_labels[ch]
        for ch in raw.ch_names
        if ch in config.channel_labels
    }
    if mapping_1020:
        raw.rename_channels(mapping_1020)


def rename_and_set_montage(subject_id):
    """
    Renames channels and sets the montage for a subject.

    This function processes both P1 and P2 data, renames channels,
    sets the BioSemi64 montage, and saves the updated data.

    Args:
        subject_id (str): The subject identifier.

    Returns:
        list: List of tuples (person, out_path).
    """
    outputs = []
    for person in ["P1", "P2"]:
        # Get file paths for the current person
        path_in, _ = get_step_io_files(
            subject_id,
            __file__,
            person=person,
            pipeline_steps=config.PIPELINE_STEPS,
            step_output_suffixes=config.STEP_OUTPUT_SUFFIXES,
        )
        if path_in is None:
            raise ValueError("Rename/montage step requires split input files")

        LOGGER.info(f"Processing subject {subject_id}, person {person}")
        LOGGER.info(f"Loading previous step file ({person}): {path_in}")
        raw = mne.io.read_raw_fif(path_in, preload=True)

        # Rename EEG channels
        _rename_eeg_channels(raw, person)

        # Create and set the BioSemi64 montage
        montage = _make_biosemi64_montage(raw)
        raw.set_montage(montage, match_case=False, on_missing="warn")

        # Save the processed data
        out_path = save_current_step_file(
            raw,
            subject_id,
            __file__,
            person=person,
            step_output_suffixes=config.STEP_OUTPUT_SUFFIXES,
        )
        LOGGER.info(f"Saved renamed+montaged file ({person}) to: {out_path}")
        outputs.append((person, out_path))

    return outputs


if __name__ == "__main__":
    for i, subj in enumerate(config.SUBJECTS):
        rename_and_set_montage(subj)
        LOGGER.info(f"PROGRESS:{i + 1}")
