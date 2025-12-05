"""
sanity_check_02_re_reference.py / Jupyter Notebook Version

Standalone sanity check for re-referenced EEG data (hyperscanning).

Purpose:
- Quick QA to ensure re-referencing worked correctly
- Visual inspection of EEG signals (interactive)
- Power spectral density (PSD) check
- Channel mean baseline check
- Optionally compare with original (pre-reference) data
- Confirms participant splitting and mastoid referencing

Usage in Notebook:
    %matplotlib notebook
    sanity_check_reref('01')  # or loop through subjects
"""

import mne
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import config  # assumes your config.py with OUTPUT_DIR and SUBJECTS

def sanity_check_reref(subject_id, compare_original=False):
    print(f"\n=== Sanity check for re-referenced data: Subject {subject_id} ===")

    # -------------------------------
    # Paths to re-referenced files
    # -------------------------------
    file_p1 = Path(config.OUTPUT_DIR) / f"sub-{subject_id}_task-RPS_p1_ref_raw.fif"
    file_p2 = Path(config.OUTPUT_DIR) / f"sub-{subject_id}_task-RPS_p2_ref_raw.fif"

    # Optional: paths to original raw files (before re-reference)
    if compare_original:
        file_orig = Path(config.OUTPUT_DIR) / f"sub-{subject_id}_task-RPS_raw.fif"

    # -------------------------------
    # Load raw data
    # -------------------------------
    raw_p1 = mne.io.read_raw_fif(file_p1, preload=True)
    raw_p2 = mne.io.read_raw_fif(file_p2, preload=True)
    if compare_original:
        raw_orig = mne.io.read_raw_fif(file_orig, preload=True)

    # -------------------------------
    # 1) Print info
    # -------------------------------
    print("\n--- Participant 1 Info ---")
    print(raw_p1.info)
    print("\n--- Participant 2 Info ---")
    print(raw_p2.info)

    # -------------------------------
    # 2) Plot EEG signals (interactive)
    # -------------------------------
    print("Plotting EEG signals for Participant 1...")
    raw_p1.plot(n_channels=32, scalings='auto', title=f"Subject {subject_id} - P1 EEG")

    print("Plotting EEG signals for Participant 2...")
    raw_p2.plot(n_channels=32, scalings='auto', title=f"Subject {subject_id} - P2 EEG")

    # -------------------------------
    # 3) Plot Power Spectral Density (PSD)
    # -------------------------------
    print("Plotting PSD for Participant 1...")
    fig1 = raw_p1.plot_psd(fmax=50, average=True)
    fig1.suptitle(f"Subject {subject_id} - P1 PSD")

    print("Plotting PSD for Participant 2...")
    fig2 = raw_p2.plot_psd(fmax=50, average=True)
    fig2.suptitle(f"Subject {subject_id} - P2 PSD")

    # -------------------------------
    # 4) Check channel means (baseline)
    # -------------------------------
    mean_p1 = raw_p1.get_data().mean(axis=1)
    mean_p2 = raw_p2.get_data().mean(axis=1)

    plt.figure(figsize=(12,4))
    plt.plot(mean_p1, label="P1 channel means")
    plt.plot(mean_p2, label="P2 channel means")
    plt.xlabel("Channels")
    plt.ylabel("Mean amplitude (µV)")
    plt.title(f"Subject {subject_id} - Channel Means After Re-reference")
    plt.legend()
    plt.show()

    # -------------------------------
    # 5) Optional: Compare with original raw
    # -------------------------------
    if compare_original:
        print("Comparing re-referenced vs original signals...")
        raw_orig_data = raw_orig.get_data()
        raw_p1_data = raw_p1.get_data(picks=[ch for ch in raw_p1.ch_names if ch.startswith("1-")])
        raw_p2_data = raw_p2.get_data(picks=[ch for ch in raw_p2.ch_names if ch.startswith("2-")])

        plt.figure(figsize=(12,4))
        plt.plot(raw_orig_data.mean(axis=0), label="Original average across channels", alpha=0.7)
        plt.plot(raw_p1_data.mean(axis=0), label="P1 re-referenced average", alpha=0.7)
        plt.plot(raw_p2_data.mean(axis=0), label="P2 re-referenced average", alpha=0.7)
        plt.xlabel("Time samples")
        plt.ylabel("Amplitude (µV)")
        plt.title(f"Subject {subject_id} - Signal comparison before/after re-reference")
        plt.legend()
        plt.show()

    print("Sanity check complete.\n")


# -------------------------------
# Script entry (optional for .py)
# -------------------------------
if __name__ == "__main__":
    for subj in config.SUBJECTS:
        try:
            sanity_check_reref(subj, compare_original=True)
        except Exception as e:
            print(f"Subject {subj}: FAILED — {e}")
