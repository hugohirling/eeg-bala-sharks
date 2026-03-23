"""
Sanity Check for Step 06: ICA Artifact Removal

Überprüft:
- ICA Komponenten extrahiert
- EOG Artefakte erkannt
- Amplituden-Reduktion plausibel
- ICA-Zerlegung gespeichert
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

import config


def sanity_check_ica():
    print("\n" + "=" * 80)
    print("SANITY CHECK: Step 06 - ICA Artifact Removal")
    print("=" * 80)

    for subject_id in config.SUBJECTS:
        print(f"\n--- Checking subject {subject_id} ---")

        for person in ["P1", "P2"]:
            before_path = config.OUTPUT_DIR / f"sub-{subject_id}_{person}_filtered.fif"
            after_path = config.OUTPUT_DIR / f"sub-{subject_id}_{person}_ica_cleaned.fif"
            ica_path = config.QC_DIR / f"sub-{subject_id}_{person}_ica.fif"

            if not before_path.exists():
                print(f"\n  {person}: Input file (filtered) not found")
                continue

            if not after_path.exists():
                print(f"\n  {person}: Output file (ica_cleaned) not found")
                continue

            raw_before = mne.io.read_raw_fif(str(before_path), preload=False)
            raw_after = mne.io.read_raw_fif(str(after_path), preload=False)

            print(f"\n{person}:")
            print(f"  ✓ Files exist")

            # Check ICA decomposition file
            if ica_path.exists():
                try:
                    ica = mne.preprocessing.read_ica(str(ica_path))
                    print(f"  ✓ ICA decomposition loaded: {ica_path.name}")
                    print(f"    Number of components: {ica.n_components_}")
                    print(f"    Components marked for exclusion: {len(ica.exclude)}")
                except Exception as e:
                    print(f"  ERROR loading ICA: {e}")
            else:
                print(f"  WARNING: ICA file not found at {ica_path}")

            # Check metadata preserved
            if len(raw_before.ch_names) == len(raw_after.ch_names):
                print(f"  ✓ Channel count same: {len(raw_after.ch_names)}")
            else:
                print(f"  ERROR: Channel count changed")

            if raw_before.n_times == raw_after.n_times:
                print(f"  ✓ Sample count same: {raw_after.n_times}")
            else:
                print(f"  ERROR: Sample count changed")

            # Compare amplitudes
            eeg_picks = mne.pick_types(raw_before.info, eeg=True)
            if len(eeg_picks) > 0:
                t_end = min(120, raw_before.times[-1])
                t_idx_end = int(t_end * raw_before.info["sfreq"])

                data_before = raw_before.get_data(picks=eeg_picks, start=0, stop=t_idx_end)
                data_after = raw_after.get_data(picks=eeg_picks, start=0, stop=t_idx_end)

                std_before = np.std(data_before)
                std_after = np.std(data_after)

                print(f"  ✓ EEG amplitude (first 120s):")
                print(f"    Before ICA - Std: {std_before:.6f} µV")
                print(f"    After ICA - Std: {std_after:.6f} µV")

                reduction_pct = (1 - std_after / std_before) * 100
                print(f"    Change: {reduction_pct:+.1f}%")

                if std_after > std_before * 1.5:
                    print(f"    WARNING: Amplitude increased significantly (possible ICA error)")

            # Check EOG
            eog_picks = mne.pick_types(raw_before.info, eog=True)
            if len(eog_picks) > 0:
                eog_data_before = raw_before.get_data(picks=eog_picks, start=0, stop=t_idx_end)
                eog_data_after = raw_after.get_data(picks=eog_picks, start=0, stop=t_idx_end)

                eog_std_before = np.std(eog_data_before)
                eog_std_after = np.std(eog_data_after)
                eog_reduction = (1 - eog_std_after / eog_std_before) * 100

                print(f"  ✓ EOG amplitude:")
                print(f"    Before ICA - Std: {eog_std_before:.6f} µV")
                print(f"    After ICA - Std: {eog_std_after:.6f} µV")
                print(f"    Reduction: {eog_reduction:.1f}%")

    print("\n" + "=" * 80)
    print("Sanity check completed.")
    print("=" * 80)


if __name__ == "__main__":
    sanity_check_ica()
