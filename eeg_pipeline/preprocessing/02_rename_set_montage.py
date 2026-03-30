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
    Creates a BioSemi64 montage from the template file.

    This function loads the BioSemi64 electrode positions from a .mat file,
    converts units if necessary, and creates a digitization montage for the
    channels present in the raw data.

    Args:
        raw (mne.io.Raw): The raw data to create the montage for.

    Returns:
        mne.channels.DigMontage: The created montage object.

    Raises:
        FileNotFoundError: If the BioSemi template file is missing.
        KeyError: If the expected variable is not in the .mat file.
        ValueError: If the positions array has unexpected shape or no matching channels.
    """
    mat_path = Path(config.BIOSEMI64_MAT_PATH)
    if not mat_path.exists():
        raise FileNotFoundError(f"Missing BioSemi template: {mat_path}")

    mat_data = loadmat(str(mat_path))
    if "biosemi64" not in mat_data:
        raise KeyError("Variable 'biosemi64' not found in biosemi64.mat")

    positions = np.asarray(mat_data["biosemi64"], dtype=float)
    if positions.shape != (64, 3):
        raise ValueError(
            f"Expected biosemi64 shape (64, 3), got {positions.shape}"
        )

    # Convert from mm to meters if necessary (biosemi64.mat is typically in mm)
    # Check if the head radius would be reasonable in meters
    mean_distance = np.linalg.norm(positions, axis=1).mean()
    if mean_distance > 1:  # If mean distance > 1m, likely in millimeters
        LOGGER.info(f"BioSemi positions appear to be in mm (mean distance: {mean_distance:.2f}). Converting to meters.")
        positions = positions / 1000.0
    
    # Keep original 3D coordinates from biosemi64.mat to stay aligned with
    # the FieldTrip preprocessing reference (elec.pnt = biosemi64).
    ordered_labels = list(config.channel_labels.values())
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
    Renames EEG channels according to the BioSemi64 standard.

    This function strips player prefixes, removes unwanted channels (starting with 'C' or 'D'),
    and maps channels to 10-20 system labels.

    Args:
        raw (mne.io.Raw): The raw data to modify.
        person (str): The player identifier ('P1' or 'P2').
    """
    source_prefix = config.PLAYER_PREFIX_MAP[person]

    eeg_picks = mne.pick_types(raw.info, eeg=True)
    eeg_channels = [raw.ch_names[idx] for idx in eeg_picks]

    # Strip the player prefix from EEG channels
    strip_prefix_map = {
        ch: ch[len(source_prefix):]
        for ch in eeg_channels
        if ch.startswith(source_prefix)
    }
    if strip_prefix_map:
        raw.rename_channels(strip_prefix_map)

    # Remove channels starting with 'C' or 'D'
    channels_to_drop = [ch for ch in raw.ch_names if ch.startswith('C') or ch.startswith('D')]
    if channels_to_drop:
        raw.drop_channels(channels_to_drop)

    # Map to 10-20 system labels
    mapping_1020 = {
        ch: config.channel_labels[ch]
        for ch in raw.ch_names
        if ch in config.channel_labels
    }
    if mapping_1020:
        raw.rename_channels(mapping_1020)


def rename_and_set_montage(subject_id):
    """
    Renames channels and sets the BioSemi64 montage for both players.

    This function processes each player's data file, renames channels,
    applies the BioSemi64 montage, and saves the updated files.

    Args:
        subject_id (str): The subject identifier.

    Returns:
        list[tuple[str, Path]]: List of tuples with player ID and output file path.
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
            raise ValueError("Rename/montage step requires split input files")

        LOGGER.info(f"Processing subject {subject_id}, person {person}")
        LOGGER.info(f"Loading previous step file ({person}): {path_in}")
        raw = mne.io.read_raw_fif(path_in, preload=True)

        _rename_eeg_channels(raw, person)

        montage = _make_biosemi64_montage(raw)
        raw.set_montage(montage, match_case=False, on_missing="warn")

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
    # Process all subjects
    for subj in config.SUBJECTS:
        rename_and_set_montage(subj)
