from pathlib import Path
import sys

import mne
import numpy as np

CURRENT_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = CURRENT_DIR.parent
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from preprocessing import config

from helper.general.helper_functions import load_epochs_fif, save_epochs_fif, save_json
from helper.authors.authors_helpers import resolve_bin_duration, resolve_output_dir


def create_time_bins(epochs, bin_duration=0.25):
    data = epochs.get_data()
    sfreq = epochs.info["sfreq"]
    times = epochs.times
    bin_size_samples = int(bin_duration * sfreq)

    binned_data = []
    bin_times = []
    start_idx = 0
    while start_idx < data.shape[2]:
        end_idx = min(start_idx + bin_size_samples, data.shape[2])
        binned_data.append(np.mean(data[:, :, start_idx:end_idx], axis=2))
        bin_times.append(np.mean(times[start_idx:end_idx]))
        start_idx = end_idx

    return np.stack(binned_data, axis=2), np.array(bin_times)


def process_subject(subject_id, bin_duration=0.25):
    output_dir = resolve_output_dir()

    for phase in ["decision", "response", "feedback"]:
        epochs_file = output_dir / f"sub-{subject_id}_{phase}_authors-epo.fif"
        if not epochs_file.exists():
            print(f"[authors] Missing epoch file for {phase}: {epochs_file}")
            continue

        epochs = load_epochs_fif(epochs_file, preload=True)
        binned_data, binned_times = create_time_bins(epochs, bin_duration=bin_duration)

        binned_info = mne.create_info(
            ch_names=epochs.ch_names,
            sfreq=1.0 / float(bin_duration),
            ch_types=epochs.get_channel_types(),
        )
        binned_info["bads"] = list(epochs.info.get("bads", []))
        montage = epochs.get_montage()
        if montage is not None:
            binned_info.set_montage(montage, on_missing="ignore")

        binned_epochs = mne.EpochsArray(
            binned_data,
            binned_info,
            events=epochs.events,
            event_id=epochs.event_id,
            tmin=float(binned_times[0]) if len(binned_times) else float(epochs.times[0]),
            baseline=None,
            metadata=epochs.metadata,
        )

        binned_file = output_dir / f"sub-{subject_id}_{phase}_binned_authors-epo.fif"
        save_epochs_fif(binned_epochs, binned_file)

        meta_file = output_dir / f"sub-{subject_id}_{phase}_binned_authors_meta.json"
        save_json(
            {
                "subject": f"sub-{subject_id}",
                "phase": phase,
                "n_epochs": int(binned_data.shape[0]),
                "n_channels": int(binned_data.shape[1]),
                "n_bins": int(binned_data.shape[2]),
                "bin_duration_ms": int(bin_duration * 1000),
            },
            meta_file,
        )

        print(f"[authors] Saved binned output: {binned_file}")


if __name__ == "__main__":
    bin_duration = resolve_bin_duration()
    for subj in config.SUBJECTS:
        process_subject(subj, bin_duration=bin_duration)
