# 03_autoreject.py

import mne
import numpy as np
from pathlib import Path
from autoreject import AutoReject
from preprocessing import config

def apply_autoreject(epochs):
    """
    Apply AutoReject to detect and interpolate bad channels/epochs.
    
    WHY:
    - AutoReject erkennt statistische Ausreißer in Epochs (Muskelartefakte, Spitzen, etc.)
    - Interpoliert schlechte Kanäle in guten Epochs
    - Markiert vollständig fehlerhafte Epochs zur Entfernung
    - Besser als manuelle Varianz-Threshold, da es adaptive Schwellenwerte verwendet

    Parameters:
    -----------
    epochs : mne.Epochs
        The input epochs object
        
    Returns:
    --------
    epochs_ar : mne.Epochs
        Cleaned epochs with bad channels interpolated
    ar : AutoReject object
        The fitted AutoReject object (contains rejection thresholds)
    """
    print("\n=== Applying AutoReject ===")
    print(f"Input: {len(epochs)} epochs, {len(epochs.ch_names)} channels")
    
    # --- Initialize AutoReject ---
    ar = AutoReject(
        n_jobs=config.AR_N_JOBS,
        verbose=config.AR_VERBOSE
    )
    
    # --- Fit and apply ---
    epochs_ar = ar.fit_transform(epochs)
    
    # --- Report ---
    n_rejected = len(epochs) - len(epochs_ar)
    print(f"Epochs rejected: {n_rejected} / {len(epochs)} ({100*n_rejected/len(epochs):.1f}%)")
    print(f"Output: {len(epochs_ar)} epochs")
    
    return epochs_ar, ar


def process_subject(path_in, path_out):
    """
    Load epochs from previous step, apply AutoReject, and save cleaned epochs.
    """
    epochs = mne.read_epochs(path_in, preload=True, verbose=False)
    
    # Run AutoReject
    epochs_ar, ar = apply_autoreject(epochs)
    
    # Save annotated epochs file
    epochs_ar.save(path_out, overwrite=True)
    
    print(f"Saved autoreject-cleaned epochs to: {path_out}")


if __name__ == "__main__":
    from preprocessing import config
    for subj in config.SUBJECTS:
        p1 = Path(config.OUTPUT_DIR) / f"sub-{subj}_P1_epoch.fif"
        p2 = Path(config.OUTPUT_DIR) / f"sub-{subj}_P2_epoch.fif"

        out_p1 = Path(config.OUTPUT_DIR) / f"sub-{subj}_P1_autoreject.fif"
        out_p2 = Path(config.OUTPUT_DIR) / f"sub-{subj}_P2_autoreject.fif"

        process_subject(p1, out_p1)
        process_subject(p2, out_p2)
