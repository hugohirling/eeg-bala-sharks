from pathlib import Path
import sys

import mne

CURRENT_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = CURRENT_DIR.parent
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

import config
from helper.helper_functions import get_step_io_files, save_current_step_file


def _extract_events(raw):
    events, event_id = mne.events_from_annotations(raw, verbose=False)
    if len(events) > 0 and len(event_id) > 0:
        return events, event_id

    if "Status" in raw.ch_names:
        events = mne.find_events(raw, stim_channel="Status", shortest_event=1, verbose=False)
        if len(events) > 0:
            unique_codes = sorted(set(events[:, 2]))
            event_id = {f"event_{code}": int(code) for code in unique_codes}
            return events, event_id

    raise RuntimeError("No events found in annotations or Status channel")


def make_epochs(raw):
    events, event_id = _extract_events(raw)
    baseline = (config.EPOCH_BASELINE_MIN, config.EPOCH_BASELINE_MAX)

    epochs = mne.Epochs(
        raw,
        events=events,
        event_id=event_id,
        tmin=config.EPOCH_TMIN,
        tmax=config.EPOCH_TMAX,
        baseline=baseline,
        preload=True,
        reject_by_annotation=True,
    )
    return epochs


def process_subject(subject_id):
    outputs = []
    for person in ["P1", "P2"]:
        path_in, _ = get_step_io_files(subject_id, __file__, person=person)
        if path_in is None:
            raise ValueError("Epoching step requires ICA-cleaned input files")

        print(f"Loading previous step file ({person}): {path_in}")
        raw = mne.io.read_raw_fif(path_in, preload=True)
        epochs = make_epochs(raw)

        out_path = save_current_step_file(epochs, subject_id, __file__, person=person)
        print(f"Saved epochs file ({person}) to: {out_path}")

        outputs.append((person, out_path, len(epochs)))

    return outputs


if __name__ == "__main__":
    for subj in config.SUBJECTS:
        process_subject(subj)
