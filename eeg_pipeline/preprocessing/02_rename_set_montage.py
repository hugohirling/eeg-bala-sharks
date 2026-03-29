from pathlib import Path
import sys

import mne
import numpy as np
from scipy.io import loadmat

CURRENT_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = CURRENT_DIR.parent
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from preprocessing import config
from helper.general.helper_functions import get_step_io_files, save_current_step_file


def _make_biosemi64_montage(raw):
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

    return mne.channels.make_dig_montage(ch_pos=ch_pos, coord_frame="head")


def _rename_eeg_channels(raw, person):
    source_prefix = config.PLAYER_PREFIX_MAP[person]

    eeg_picks = mne.pick_types(raw.info, eeg=True)
    eeg_channels = [raw.ch_names[idx] for idx in eeg_picks]

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

    mapping_1020 = {
        ch: config.channel_labels[ch]
        for ch in raw.ch_names
        if ch in config.channel_labels
    }
    if mapping_1020:
        raw.rename_channels(mapping_1020)


def rename_and_set_montage(subject_id):
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

        print(f"Loading previous step file ({person}): {path_in}")
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
        print(f"Saved renamed+montaged file ({person}) to: {out_path}")
        outputs.append((person, out_path))

    return outputs


if __name__ == "__main__":
    for subj in config.SUBJECTS:
        rename_and_set_montage(subj)
