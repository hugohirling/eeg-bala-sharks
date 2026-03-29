from pathlib import Path
import sys

import mne

CURRENT_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = CURRENT_DIR.parent
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from preprocessing import config
from helper.general.helper_functions import get_step_io_files, save_current_step_file


def _pick_existing_channels(raw, channels):
    return [ch for ch in channels if ch in raw.ch_names]


def _set_channel_types(raw, eeg_channels, player_prefix):
    type_map = {ch: "eeg" for ch in eeg_channels}

    for channel in raw.ch_names:
        if channel in eeg_channels:
            continue
        if channel == "Status":
            type_map[channel] = "stim"
            continue
        if not channel.startswith(player_prefix):
            continue

        suffix = channel.split("-", 1)[1] if "-" in channel else channel
        channel_type = config.channel_types.get(suffix)
        if channel_type is not None:
            type_map[channel] = channel_type

    if type_map:
        raw.set_channel_types(type_map)


def split_players(subject_id):
    path_in, _ = get_step_io_files(
        subject_id,
        __file__,
        pipeline_steps=config.PIPELINE_STEPS,
        step_output_suffixes=config.STEP_OUTPUT_SUFFIXES,
    )
    if path_in is None:
        raise ValueError("Split step requires an input file from the previous pipeline step")

    print(f"Loading previous step file: {path_in}")
    raw = mne.io.read_raw_fif(path_in, preload=True)

    outputs = []
    for person, prefix in config.PLAYER_PREFIX_MAP.items():
        eeg_channels = [
            ch for ch in raw.ch_names if ch.startswith(f"{prefix}A") or ch.startswith(f"{prefix}B")
        ]
        if not eeg_channels:
            raise RuntimeError(f"No EEG channels found for {person} with prefix '{prefix}'")

        aux_channels = [ch for ch in raw.ch_names if ch.startswith(prefix) and ch not in eeg_channels]
        channels_to_keep = _pick_existing_channels(raw, eeg_channels + aux_channels + ["Status"])

        split_raw = raw.copy().pick(channels_to_keep)
        _set_channel_types(split_raw, eeg_channels=eeg_channels, player_prefix=prefix)

        out_path = save_current_step_file(
            split_raw,
            subject_id,
            __file__,
            person=person,
            step_output_suffixes=config.STEP_OUTPUT_SUFFIXES,
        )
        print(f"Saved split file ({person}) to: {out_path}")
        outputs.append((person, out_path))

    return outputs


if __name__ == "__main__":
    for subj in config.SUBJECTS:
        split_players(subj)
