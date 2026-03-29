"""
Sanity Check for Step 01: Split Players

Überprüft:
- Spieler korrekt aufgeteilt
- Kanäle pro Person korrekt zugeordnet
- Kanal-Typen gesetzt
- Status-Kanal vorhanden für beide
"""
import sys
from pathlib import Path

import mne

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

PIPELINE_DIR = CURRENT_DIR.parent.parent / "eeg_pipeline"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from preprocessing import config


def sanity_check_split_players():
    print("\n" + "=" * 80)
    print("SANITY CHECK: Step 01 - Split Players")
    print("=" * 80)

    for subject_id in config.SUBJECTS:
        print(f"\n--- Checking subject {subject_id} ---\n")

        for person in ["P1", "P2"]:
            path = config.OUTPUT_DIR / f"sub-{subject_id}_{person}_split.fif"

            if not path.exists():
                print(f"  ERROR: Split file not found for {person}: {path}")
                continue

            raw = mne.io.read_raw_fif(str(path), preload=False)

            print(f"\n{person}:")
            print(f"  ✓ File exists: {path.name}")
            print(f"  ✓ Total channels: {len(raw.ch_names)}")

            # Count by type
            eeg_picks = mne.pick_types(raw.info, eeg=True, exclude=[])
            eog_picks = mne.pick_types(raw.info, eog=True, exclude=[])
            resp_picks = mne.pick_types(raw.info, resp=True, exclude=[])
            misc_picks = mne.pick_types(raw.info, misc=True, exclude=[])
            stim_picks = mne.pick_types(raw.info, stim=True, exclude=[])

            print(f"    - EEG channels: {len(eeg_picks)}")
            print(f"    - EOG channels: {len(eog_picks)}")
            print(f"    - Respiration channels: {len(resp_picks)}")
            print(f"    - Misc channels: {len(misc_picks)}")
            print(f"    - Stim channels: {len(stim_picks)}")

            if len(stim_picks) == 0:
                print(f"    WARNING: No stim channel found for {person}")

            prefix = config.PLAYER_PREFIX_MAP[person]
            ch_count_with_prefix = sum(1 for ch in raw.ch_names if ch.startswith(prefix))
            if ch_count_with_prefix > 0:
                print(f"    WARNING: {ch_count_with_prefix} channels still have prefix '{prefix}'")
            else:
                print(f"    ✓ Prefix '{prefix}' successfully removed from all EEG channels")

            print(f"  ✓ Sampling rate: {raw.info['sfreq']} Hz")
            print(f"  ✓ Duration: {raw.times[-1]:.2f}s")

    print("\n" + "=" * 80)
    print("Sanity check completed.")
    print("=" * 80)


if __name__ == "__main__":
    sanity_check_split_players()

