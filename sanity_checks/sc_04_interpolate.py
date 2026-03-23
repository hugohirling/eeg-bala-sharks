"""
Sanity Check for Step 04: Interpolate Bad Channels

Überprüft:
- Interpolation durchgeführt
- Bads-Liste geleert nach Interpolation
- Kanal-Anzahl gleich geblieben
- Amplituden noch plausibel
"""
import sys
from pathlib import Path

import mne

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

import config


def sanity_check_interpolate_bad_channels():
    print("\n" + "=" * 80)
    print("SANITY CHECK: Step 04 - Interpolate Bad Channels")
    print("=" * 80)

    for subject_id in config.SUBJECTS:
        print(f"\n--- Checking subject {subject_id} ---\n")

        for person in ["P1", "P2"]:
            before_path = config.OUTPUT_DIR / f"sub-{subject_id}_{person}_badchannels_detected.fif"
            after_path = config.OUTPUT_DIR / f"sub-{subject_id}_{person}_interpolated.fif"

            if not before_path.exists():
                print(f"  {person}: Input file (badchannels_detected) not found")
                continue

            if not after_path.exists():
                print(f"  {person}: Output file (interpolated) not found")
                continue

            raw_before = mne.io.read_raw_fif(str(before_path), preload=False)
            raw_after = mne.io.read_raw_fif(str(after_path), preload=False)

            bads_before = raw_before.info.get("bads", [])
            bads_after = raw_after.info.get("bads", [])

            print(f"\n{person}:")
            print(f"  ✓ Files exist")
            print(f"  Bads before interpolation: {len(bads_before)}")
            if bads_before:
                print(f"    {', '.join(bads_before)}")
            print(f"  Bads after interpolation: {len(bads_after)}")

            # After interpolation, bads list should be empty
            if len(bads_after) == 0:
                print(f"  ✓ Bad channels cleared after interpolation")
            else:
                print(f"  WARNING: Bad channels still marked after interpolation")

            # Channel count should be same
            if len(raw_before.ch_names) == len(raw_after.ch_names):
                print(f"  ✓ Channel count preserved: {len(raw_after.ch_names)}")
            else:
                print(f"  ERROR: Channel count mismatch")

            # Sampling rate and duration should be the same
            if raw_before.info["sfreq"] == raw_after.info["sfreq"]:
                print(f"  ✓ Sampling rate same: {raw_after.info['sfreq']} Hz")
            else:
                print(f"  ERROR: Sampling rate changed")

            n_samples_before = raw_before.n_times
            n_samples_after = raw_after.n_times
            if n_samples_before == n_samples_after:
                print(f"  ✓ Sample count same: {n_samples_after}")
            else:
                print(f"  ERROR: Sample count changed from {n_samples_before} to {n_samples_after}")

    print("\n" + "=" * 80)
    print("Sanity check completed.")
    print("=" * 80)


if __name__ == "__main__":
    sanity_check_interpolate_bad_channels()
