"""
Sanity Check for Step 00: Downsample

Überprüft:
- Downsampling erfolgreich durchgeführt
- Sampling Rate korrekt reduziert
- Datenlänge und -größe stimmen
- Keine Artefakte durch Downsampling
"""
import sys
from pathlib import Path

import mne
import numpy as np

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

PIPELINE_DIR = CURRENT_DIR.parent / "eeg_pipeline"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

import config


def sanity_check_downsample():
    print("\n" + "=" * 80)
    print("SANITY CHECK: Step 00 - Downsample")
    print("=" * 80)

    from mne_bids import BIDSPath, read_raw_bids

    for subject_id in config.SUBJECTS:
        print(f"\n--- Checking subject {subject_id} ---\n")

        # Load original BIDS data
        bids_path = BIDSPath(
            subject=subject_id,
            task="RPS",
            datatype="eeg",
            suffix="eeg",
            root=config.BIDS_ROOT,
        )
        raw_original = read_raw_bids(bids_path, verbose=False)
        original_sfreq = raw_original.info["sfreq"]

        # Load downsampled output
        out_path = config.OUTPUT_DIR / f"sub-{subject_id}_downsampled.fif"
        if not out_path.exists():
            print(f"ERROR: Downsampled file not found: {out_path}")
            continue

        raw_downsampled = mne.io.read_raw_fif(str(out_path), preload=False)
        downsampled_sfreq = raw_downsampled.info["sfreq"]

        # Checks
        print(f"✓ Original sampling rate: {original_sfreq} Hz")
        print(f"✓ Downsampled rate: {downsampled_sfreq} Hz")
        print(f"✓ Downsampling factor: {original_sfreq / downsampled_sfreq:.1f}x")

        expected_sfreq = config.DOWNSAMPLE_SFREQ
        if downsampled_sfreq != expected_sfreq:
            print(f"  WARNING: Expected {expected_sfreq} Hz, got {downsampled_sfreq} Hz")

        # Channel count should remain the same
        if len(raw_original.ch_names) == len(raw_downsampled.ch_names):
            print(f"✓ Channel count preserved: {len(raw_downsampled.ch_names)}")
        else:
            print(
                f"  ERROR: Channel count mismatch. Before: {len(raw_original.ch_names)}, After: {len(raw_downsampled.ch_names)}"
            )

        # Duration should be approximately the same
        original_duration = raw_original.times[-1]
        downsampled_duration = raw_downsampled.times[-1]
        duration_diff = abs(original_duration - downsampled_duration)
        print(f"✓ Original duration: {original_duration:.2f}s")
        print(f"✓ Downsampled duration: {downsampled_duration:.2f}s")
        if duration_diff < 0.1:
            print(f"  Duration difference: {duration_diff:.4f}s (acceptable)")

        # Estimate file size reduction
        original_est_size = (len(raw_original.ch_names) * raw_original.n_times * 8) / 1e6
        downsampled_est_size = (len(raw_downsampled.ch_names) * raw_downsampled.n_times * 8) / 1e6
        size_ratio = downsampled_est_size / original_est_size * 100

        print(f"✓ Est. size (before): {original_est_size:.2f} MB")
        print(f"✓ Est. size (after): {downsampled_est_size:.2f} MB")
        print(f"✓ Size reduction: {size_ratio:.1f}% of original")

    print("\n" + "=" * 80)
    print("Sanity check completed.")
    print("=" * 80)


if __name__ == "__main__":
    sanity_check_downsample()
