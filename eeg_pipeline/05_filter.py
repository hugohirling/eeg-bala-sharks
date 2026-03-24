from pathlib import Path
import sys

import mne

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

import config
from helper.helper_functions import get_step_io_files, save_current_step_file


def load_previous_step_data(subject_id):
    path_in, _ = get_step_io_files(subject_id, __file__)
    if path_in is None:
        raise ValueError("Filter step requires an input file from the previous pipeline step")

    print(f"Loading previous step file: {path_in}")
    return mne.io.read_raw_fif(path_in, preload=True), path_in


def filter_data(raw, l_freq=config.FREQ_LOWER, h_freq=config.FREQ_UPPER):
    print(f"Applying bandpass filter: {l_freq} - {h_freq} Hz")
    raw.filter(l_freq=l_freq, h_freq=h_freq)
    return raw


def process_subject(subject_id):
    outputs = []
    for person in ["P1", "P2"]:
        path_in, _ = get_step_io_files(subject_id, __file__, person=person)
        if path_in is None:
            raise ValueError("Filter step requires an input file from the previous pipeline step")

        print(f"Loading previous step file: {path_in}")
        raw = mne.io.read_raw_fif(path_in, preload=True)
        raw = filter_data(raw)
        out_path = save_current_step_file(raw, subject_id, __file__, person=person)
        print(f"Saved filtered file ({person}) to: {out_path}")
        outputs.append((person, path_in, out_path))
    return outputs


if __name__ == "__main__":
    for subj in config.SUBJECTS:
        process_subject(subj)