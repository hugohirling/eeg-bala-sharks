from pathlib import Path
import mne
from utils import load_raw, save_raw
import numpy as np


def run_ica(raw, n_components=0.99, random_state=97):
    """
    Run ICA to remove ocular (and optionally muscular) artifacts.
    """
    print("Running ICA...")

    # High-pass filter copy for ICA fitting (recommended)
    raw_for_ica = raw.copy().filter(l_freq=1.0, h_freq=None)

    ica = mne.preprocessing.ICA(
        n_components=n_components,
        method="fastica",
        random_state=random_state,
        max_iter="auto"
    )

    ica.fit(raw_for_ica)

    # ---- EOG detection ----
    eog_inds, eog_scores = ica.find_bads_eog(raw)
    ica.exclude.extend(eog_inds)

    print(f"Marked EOG components for exclusion: {eog_inds}")

    # ---- OPTIONAL: EMG / muscle heuristic ----
    # (simple variance-based heuristic; conservative)
    try:
        muscle_inds, muscle_scores = ica.find_bads_muscle(raw, threshold=1.5)
        ica.exclude.extend(muscle_inds)
        print(f"Marked muscle components for exclusion: {muscle_inds}")
    except Exception:
        print("Muscle component detection skipped.")

    # Apply ICA to raw data
    raw_clean = raw.copy()
    ica.apply(raw_clean)

    return raw_clean, ica


def process_subject(path_in, path_out):
    """
    Load raw EEG, run ICA artifact removal, and save cleaned raw file.
    """
    raw = load_raw(path_in, preload=True)

    raw_clean, ica = run_ica(raw)

    save_raw(raw_clean, path_out)

    print(f"Saved ICA-cleaned file to: {path_out}")


if __name__ == "__main__":
    from preprocessing import config

    for subj in config.SUBJECTS:
        p1 = Path(config.OUTPUT_DIR) / f"sub-{subj}_P1_filtered.fif"
        p2 = Path(config.OUTPUT_DIR) / f"sub-{subj}_P2_filtered.fif"

        out_p1 = Path(config.OUTPUT_DIR) / f"sub-{subj}_P1_ica_cleaned.fif"
        out_p2 = Path(config.OUTPUT_DIR) / f"sub-{subj}_P2_ica_cleaned.fif"

        process_subject(p1, out_p1)
        process_subject(p2, out_p2)
