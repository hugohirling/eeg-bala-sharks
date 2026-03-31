from pathlib import Path
import sys

import mne
import numpy as np

CURRENT_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = CURRENT_DIR.parent
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from preprocessing import config
from helper.general.helper_functions import get_step_io_files
from helper.general.helper_functions import load_raw_fif, save_epochs_fif

from helper.authors.authors_helpers import PIPELINE_STEPS, STEP_OUTPUT_SUFFIXES, events_from_raw_or_synthetic, resolve_output_dir


def process_subject(subject_id):
    output_dir = resolve_output_dir()

    path_in, _ = get_step_io_files(
        subject_id=subject_id,
        current_step=__file__,
        output_dir=output_dir,
        pipeline_steps=PIPELINE_STEPS,
        step_output_suffixes=STEP_OUTPUT_SUFFIXES,
    )
    if path_in is None or not Path(path_in).exists():
        raise FileNotFoundError(f"Input not found for sub-{subject_id}: {path_in}")

    print(f"[authors] Loading input: {path_in}")
    raw = load_raw_fif(path_in, preload=True)

    events = events_from_raw_or_synthetic(raw)
    event_id = {"trial_start": 1}
    sfreq = raw.info["sfreq"]

    decision = mne.Epochs(raw, events, event_id=event_id, tmin=-0.2, tmax=2.0, baseline=(-0.2, 0), preload=True)

    response_events = events.copy()
    response_events[:, 0] = response_events[:, 0] + int(2.0 * sfreq)
    response = mne.Epochs(raw, response_events, event_id=event_id, tmin=-0.2, tmax=2.0, baseline=(-0.2, 0), preload=True)

    feedback_events = events.copy()
    feedback_events[:, 0] = feedback_events[:, 0] + int(4.0 * sfreq)
    feedback = mne.Epochs(raw, feedback_events, event_id=event_id, tmin=-0.2, tmax=1.0, baseline=(-0.2, 0), preload=True)

    save_epochs_fif(decision, output_dir / f"sub-{subject_id}_decision_authors-epo.fif")
    save_epochs_fif(response, output_dir / f"sub-{subject_id}_response_authors-epo.fif")
    save_epochs_fif(feedback, output_dir / f"sub-{subject_id}_feedback_authors-epo.fif")

    print(f"[authors] Saved epoch outputs for sub-{subject_id}")


if __name__ == "__main__":
    for subj in config.SUBJECTS:
        process_subject(subj)
