"""
Sanity Check for Step 07: Epoching

Überprüft:
- Epochs erfolgreich erstellt
- Event-Anzahl und -Typen
- Epoch-Größe (Anzahl und Dimensionen)
- Baseline-Korrektur
"""
import sys
from pathlib import Path

import mne

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

import config


def sanity_check_epoch():
    print("\n" + "=" * 80)
    print("SANITY CHECK: Step 07 - Epoching")
    print("=" * 80)

    for subject_id in config.SUBJECTS:
        print(f"\n--- Checking subject {subject_id} ---")

        for person in ["P1", "P2"]:
            raw_path = config.OUTPUT_DIR / f"sub-{subject_id}_{person}_ica_cleaned.fif"
            epoch_path = config.OUTPUT_DIR / f"sub-{subject_id}_{person}_epoch.fif"

            if not raw_path.exists():
                print(f"\n  {person}: Input file (ica_cleaned) not found")
                continue

            if not epoch_path.exists():
                print(f"\n  {person}: Output file (epoch) not found")
                continue

            raw = mne.io.read_raw_fif(str(raw_path), preload=False)
            epochs = mne.read_epochs(str(epoch_path), preload=False)

            print(f"\n{person}:")
            print(f"  ✓ Files exist")

            # Check epoch info
            print(f"  ✓ Number of epochs: {len(epochs)}")
            print(f"  ✓ Event types in epochs: {list(epochs.event_id.keys())}")

            # Check dimensions
            print(f"  ✓ Epoch dimensions: ({len(epochs)}, {len(epochs.ch_names)}, {epochs.get_data().shape[2]})")
            print(f"    Channels: {len(epochs.ch_names)}")
            print(f"    Samples per epoch: {epochs.get_data().shape[2]}")

            # Check if time window makes sense
            tmin_actual = epochs.times[0]
            tmax_actual = epochs.times[-1]
            print(f"  ✓ Time window: [{tmin_actual:.3f}, {tmax_actual:.3f}] s")

            expected_duration = config.EPOCH_TMAX - config.EPOCH_TMIN
            actual_duration = tmax_actual - tmin_actual
            if abs(actual_duration - expected_duration) < 0.01:
                print(f"    Expected duration: {expected_duration:.3f}s - ✓ matches")
            else:
                print(f"    WARNING: Expected {expected_duration:.3f}s, got {actual_duration:.3f}s")

            # Check sampling rate
            sfreq = epochs.info["sfreq"]
            print(f"  ✓ Sampling rate: {sfreq} Hz")

            # Check for baseline
            if epochs.baseline is not None:
                print(f"  ✓ Baseline period: {epochs.baseline}")
            else:
                print(f"  WARNING: No baseline correction applied")

            # Check bad channels
            bads = epochs.info.get("bads", [])
            if len(bads) == 0:
                print(f"  ✓ No bad channels marked")
            else:
                print(f"  ⚠ Bad channels marked: {len(bads)} ({', '.join(bads[:3])}...)")

            # Sanity check: ensure event count reasonable
            if len(epochs) > 0:
                print(f"  ✓ Epoch count reasonable: {len(epochs)} epochs")
            else:
                print(f"  ERROR: No epochs created!")

            # Check for NaN or inf
            data = epochs.get_data()
            nan_count = int(np.isnan(data).sum())
            inf_count = int(np.isinf(data).sum())
            if nan_count == 0 and inf_count == 0:
                print(f"  ✓ No NaN or Inf values detected")
            else:
                print(f"  ERROR: Found {nan_count} NaN and {inf_count} Inf values")

    print("\n" + "=" * 80)
    print("Sanity check completed.")
    print("=" * 80)


if __name__ == "__main__":
    import numpy as np

    sanity_check_epoch()
