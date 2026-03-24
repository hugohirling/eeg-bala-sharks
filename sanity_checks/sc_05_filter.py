"""
Sanity Check for Step 05: Filter (Bandpass 1-40 Hz)

Überprüft:
- Filter erfolgreich angewendet
- Frequenzband korrekt (1-40 Hz)
- Power Spectral Density vor/nach Vergleich
- Amplituden reduziert
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

PIPELINE_DIR = CURRENT_DIR.parent / "eeg_pipeline"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

import config


def sanity_check_filter():
    print("\n" + "=" * 80)
    print("SANITY CHECK: Step 05 - Bandpass Filter (1-40 Hz)")
    print("=" * 80)

    for subject_id in config.SUBJECTS:
        print(f"\n--- Checking subject {subject_id} ---")

        for person in ["P1", "P2"]:
            before_path = config.OUTPUT_DIR / f"sub-{subject_id}_{person}_interpolated.fif"
            after_path = config.OUTPUT_DIR / f"sub-{subject_id}_{person}_filtered.fif"

            if not before_path.exists():
                print(f"\n  {person}: Input file (interpolated) not found")
                continue

            if not after_path.exists():
                print(f"\n  {person}: Output file (filtered) not found")
                continue

            raw_before = mne.io.read_raw_fif(str(before_path), preload=False)
            raw_after = mne.io.read_raw_fif(str(after_path), preload=False)

            print(f"\n{person}:")
            print(f"  ✓ Files exist")

            # Check metadata
            if len(raw_before.ch_names) == len(raw_after.ch_names):
                print(f"  ✓ Channel count same: {len(raw_after.ch_names)}")
            else:
                print(f"  ERROR: Channel count changed")

            if raw_before.info["sfreq"] == raw_after.info["sfreq"]:
                print(f"  ✓ Sampling rate same: {raw_after.info['sfreq']} Hz")
            else:
                print(f"  ERROR: Sampling rate changed")

            if raw_before.n_times == raw_after.n_times:
                print(f"  ✓ Sample count same: {raw_after.n_times}")
            else:
                print(f"  ERROR: Sample count changed")

            # Compare amplitudes for sanity
            eeg_picks = mne.pick_types(raw_before.info, eeg=True)
            if len(eeg_picks) > 0:
                # Get a small sample for amplitude check
                t_end = min(60, raw_before.times[-1])
                t_idx_end = int(t_end * raw_before.info["sfreq"])

                data_before = raw_before.get_data(picks=eeg_picks[0:1], start=0, stop=t_idx_end)
                data_after = raw_after.get_data(picks=eeg_picks[0:1], start=0, stop=t_idx_end)

                std_before = np.std(data_before)
                std_after = np.std(data_after)

                print(f"  ✓ Sample amplitude (first channel, first 60s):")
                print(f"    Before filter - Std: {std_before:.6f} µV")
                print(f"    After filter - Std: {std_after:.6f} µV")

                if std_after < std_before:
                    reduction_pct = (1 - std_after / std_before) * 100
                    print(f"    Reduction: {reduction_pct:.1f}% (expected for high-frequency noise removal)")
                else:
                    print(f"    WARNING: Amplitude increased after filtering")

    # Create comparison plot for visualization
    print(f"\n  Creating PSD comparison plot...")
    try:
        subject_id = config.SUBJECTS[0]
        person = "P1"
        before_path = config.OUTPUT_DIR / f"sub-{subject_id}_{person}_interpolated.fif"
        after_path = config.OUTPUT_DIR / f"sub-{subject_id}_{person}_filtered.fif"

        if before_path.exists() and after_path.exists():
            raw_before = mne.io.read_raw_fif(str(before_path), preload=False)
            raw_after = mne.io.read_raw_fif(str(after_path), preload=False)

            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            # VORHER
            raw_before_eeg = raw_before.copy().pick_types(eeg=True)
            raw_before_eeg.plot_psd(fmax=60, ax=axes[0], show=False)
            axes[0].axvline(
                x=config.FREQ_LOWER, color="red", linestyle="--", label=f"Filter: {config.FREQ_LOWER} Hz"
            )
            axes[0].axvline(
                x=config.FREQ_UPPER, color="red", linestyle="--", label=f"Filter: {config.FREQ_UPPER} Hz"
            )
            axes[0].set_title("BEFORE: Power Spectral Density")
            axes[0].legend()

            # NACHHER
            raw_after_eeg = raw_after.copy().pick_types(eeg=True)
            raw_after_eeg.plot_psd(fmax=60, ax=axes[1], show=False)
            axes[1].axvline(
                x=config.FREQ_LOWER, color="red", linestyle="--", label=f"Filter: {config.FREQ_LOWER} Hz"
            )
            axes[1].axvline(
                x=config.FREQ_UPPER, color="red", linestyle="--", label=f"Filter: {config.FREQ_UPPER} Hz"
            )
            axes[1].set_title(f"AFTER: PSD (Filtered {config.FREQ_LOWER}-{config.FREQ_UPPER} Hz)")
            axes[1].legend()

            plt.tight_layout()
            plot_path = config.QC_DIR / f"sub-{subject_id}_{person}_filter_psd_comparison.png"
            plt.savefig(plot_path, dpi=100, bbox_inches="tight")
            print(f"  ✓ Plot saved: {plot_path.name}")
            plt.close()
    except Exception as e:
        print(f"  Could not save plot: {e}")

    print("\n" + "=" * 80)
    print("Sanity check completed.")
    print("=" * 80)


if __name__ == "__main__":
    sanity_check_filter()
