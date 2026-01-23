# 03_artifact_detection.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import mne
from autoreject import AutoReject
from utils import save_object
import config

def detect_artifacts(subject_id):
    """
    Load pre-processed epochs, detect artifacts using AutoReject, and save cleaned epochs.
    """
    # Load epoched and re-referenced data
    epochs_file = config.OUTPUT_DIR / f"epochs_ref_{subject_id}-epo.fif"
    if not epochs_file.exists():
        raise FileNotFoundError(f"{epochs_file} not found. Make sure downsample + re-reference steps ran successfully.")

    print(f"[03_artifact_detection] Loading epochs from {epochs_file}...")
    epochs = mne.read_epochs(epochs_file, preload=True)

    # AutoReject
    print("[03_artifact_detection] Running AutoReject...")
    ar = AutoReject(n_jobs=1, verbose='t')  # You can adjust n_jobs if needed
    epochs_clean = ar.fit_transform(epochs)
    bad_channels = getattr(ar, 'bad_chs_', [])
    print(f"[03_artifact_detection] Detected bad channels: {bad_channels}")

    # Save bad channels list
    bads_file = config.OUTPUT_DIR / f"bads_{subject_id}.pkl"
    save_object(bad_channels, bads_file)
    print(f"[03_artifact_detection] Saved bad channels to {bads_file}")

    # Save cleaned epochs
    out_file = config.OUTPUT_DIR / f"epochs_clean_{subject_id}-epo.fif"
    print(f"[03_artifact_detection] Saving cleaned epochs to {out_file}...")
    epochs_clean.save(out_file, overwrite=True)

    return epochs_clean, bad_channels


if __name__ == "__main__":
    for subj in config.SUBJECTS:
        detect_artifacts(subj)
