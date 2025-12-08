# 03_detect_noisy_channels.py

import mne
import numpy as np
from pathlib import Path
from utils import load_raw, save_raw

def detect_noisy_channels(raw):
    """
    Detect noisy EEG channels using a custom method:
    - High variance channels (likely loose electrodes or cable noise)
    - Flat or near-flat channels (likely disconnected electrodes)

    WHY:
    EEG channels with extreme or flat signals distort ICA, re-referencing,
    epoch rejection, and downstream analyses. Detecting them before
    preprocessing ensures a robust pipeline.
    """

    print("\n=== Detecting noisy channels ===")

    # -------------------------------
    # 1) Pick EEG channels only
    # -------------------------------
    picks = mne.pick_types(raw.info, eeg=True, exclude='bads')
    data = raw.get_data(picks=picks)
    ch_names = [raw.ch_names[i] for i in picks]

    # -------------------------------
    # 2) Compute variance per channel
    # -------------------------------
    variances = data.var(axis=1)
    median_var = np.median(variances)

    # -------------------------------
    # 3) High variance channels (>10x median)
    # -------------------------------
    high_variance = [
        ch_names[i] for i, v in enumerate(variances) if v > 10 * median_var
    ]

    # -------------------------------
    # 4) Flat / near-flat channels (variance ~ 0)
    # -------------------------------
    flat_channels = [
        ch_names[i] for i, v in enumerate(variances) if v < 1e-13
    ]

    # -------------------------------
    # 5) Near-flat channels using std
    # -------------------------------
    stds = data.std(axis=1)
    near_flat = [
        ch_names[i] for i, s in enumerate(stds) if s < 1e-6
    ]

    # -------------------------------
    # 6) Combine all detected bad channels
    # -------------------------------
    all_bad_channels = list(set(high_variance + flat_channels + near_flat))
    raw.info['bads'] = all_bad_channels

    # -------------------------------
    # 7) Reporting
    # -------------------------------
    print(f"High variance channels: {high_variance}")
    print(f"Flat channels: {flat_channels}")
    print(f"Near-flat channels: {near_flat}")
    print(f"FINAL noisy channels: {all_bad_channels}")

    return raw


def process_subject(path_in, path_out):
    """
    Load raw EEG, detect noisy channels, and save annotated raw file.
    """
    raw = load_raw(path_in, preload=True)

    # Run custom noisy channel detection
    raw = detect_noisy_channels(raw)

    # Save annotated raw file (bad channels marked, not interpolated)
    save_raw(raw, path_out)

    print(f"Saved noisy-channel-cleaned file to: {path_out}")


if __name__ == "__main__":
    import config
    for subj in config.SUBJECTS:
        p1 = Path(config.OUTPUT_DIR) / f"sub-{subj}_P1_raw.fif"
        p2 = Path(config.OUTPUT_DIR) / f"sub-{subj}_P2_raw.fif"

        out_p1 = Path(config.OUTPUT_DIR) / f"sub-{subj}_P1_raw_noisy_cleaned.fif"
        out_p2 = Path(config.OUTPUT_DIR) / f"sub-{subj}_P2_raw_noisy_cleaned.fif"

        process_subject(p1, out_p1)
        process_subject(p2, out_p2)
