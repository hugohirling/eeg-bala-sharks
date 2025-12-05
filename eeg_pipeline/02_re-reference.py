# 02_re-reference.py
"""
Re-reference EEG data for hyperscanning (two participants) using linked mastoids.

- Participant 1: 1-A29 (TP9, left), 1-B29 (TP10, right)
- Participant 2: 2-A29 (TP9, left), 2-B29 (TP10, right)

MNE requires raw data to be loaded (preload=True) before setting reference.
Output filenames follow MNE/BIDS convention: *_raw.fif
"""

import config
import mne
from pathlib import Path
from utils import load_raw, save_raw

# ----------------------------------------------------------------------
# Helper function to detect mastoid channels
# ----------------------------------------------------------------------
def find_mastoids(ch_names, prefix):
    """
    Detects the mastoid channels for a participant.

    In the BioSemi ActiveTwo system:
    - Each participant has 64 EEG channels.
    - Mastoid channels correspond to TP9 (left) and TP10 (right).
    - In our dataset, these map to channels ending with '29':
        * Participant 1: 1-A29 (left), 1-B29 (right)
        * Participant 2: 2-A29 (left), 2-B29 (right)
    - Returns first two channels matching this pattern.

    Args:
        ch_names (list of str): channel names in raw data
        prefix (str): '1-' for participant 1, '2-' for participant 2

    Returns:
        list of str: left and right mastoid channels
    """
    candidates = [ch for ch in ch_names if ch.startswith(prefix) and ch.endswith("29")]
    return candidates[:2]  # take up to two channels

# ----------------------------------------------------------------------
# Main re-referencing function
# ----------------------------------------------------------------------
def rereference_subject(subject_id):
    print(f"\n=== Re-referencing subject {subject_id} ===")

    # Load raw data with preload=True (required for referencing)
    in_file = Path(config.OUTPUT_DIR) / f"sub-{subject_id}_task-RPS_raw.fif"
    if not in_file.exists():
        raise FileNotFoundError(f"Input file missing: {in_file}")

    raw = load_raw(in_file, preload=True)
    print(f"Loaded {in_file}")

    # ------------------------------------------------------------------
    # Split into Participant 1 and Participant 2
    # Each participant has independent mastoids; referencing must be done separately
    # ------------------------------------------------------------------
    p1_channels = [ch for ch in raw.ch_names if ch.startswith("1-")]
    p2_channels = [ch for ch in raw.ch_names if ch.startswith("2-")]

    if len(p1_channels) == 0 or len(p2_channels) == 0:
        raise RuntimeError("Could not find channels for P1 or P2.")

    raw_p1 = raw.copy().pick(p1_channels)
    raw_p2 = raw.copy().pick(p2_channels)

    print(f"P1 channels: {len(p1_channels)} | P2 channels: {len(p2_channels)}")

    # ------------------------------------------------------------------
    # Detect mastoids for each participant
    # ------------------------------------------------------------------
    mastoids_p1 = find_mastoids(raw_p1.ch_names, prefix="1-")
    mastoids_p2 = find_mastoids(raw_p2.ch_names, prefix="2-")

    print(f"Mastoids P1: {mastoids_p1}")
    print(f"Mastoids P2: {mastoids_p2}")

    if len(mastoids_p1) == 0 or len(mastoids_p2) == 0:
        raise RuntimeError("Could not detect mastoids for both participants.")

    # ------------------------------------------------------------------
    # Apply linked-mastoid reference independently
    # ------------------------------------------------------------------
    print("Applying mastoid reference to P1…")
    raw_p1.set_eeg_reference(ref_channels=mastoids_p1)

    print("Applying mastoid reference to P2…")
    raw_p2.set_eeg_reference(ref_channels=mastoids_p2)

    # Save re-referenced files
    out_p1 = Path(config.OUTPUT_DIR) / f"sub-{subject_id}_task-RPS_p1_ref_raw.fif"
    out_p2 = Path(config.OUTPUT_DIR) / f"sub-{subject_id}_task-RPS_p2_ref_raw.fif"

    save_raw(raw_p1, out_p1)
    save_raw(raw_p2, out_p2)

    print(f"[OK] Saved P1 → {out_p1}")
    print(f"[OK] Saved P2 → {out_p2}")

    return out_p1, out_p2


# Script entry point
if __name__ == "__main__":
    for subj in config.SUBJECTS:
        try:
            rereference_subject(subj)
        except Exception as e:
            print(f"Subject {subj}: FAILED — {e}")
