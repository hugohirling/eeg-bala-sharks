"""
Sanity Check for Step 03: Bad Channels Detect

Überprüft:
- Bad channels identifiziert
- QC-Reports erstellt
- Markierung in Raw-Objekten
- Statistiken plausibel
"""
import sys
from pathlib import Path

import mne
import numpy as np

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

PIPELINE_DIR = CURRENT_DIR.parent.parent / "eeg_pipeline"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

import config


def sanity_check_bad_channels_detect():
    print("\n" + "=" * 80)
    print("SANITY CHECK: Step 03 - Bad Channels Detect")
    print("=" * 80)

    for subject_id in config.SUBJECTS:
        print(f"\n--- Checking subject {subject_id} ---\n")

        for person in ["P1", "P2"]:
            path = config.OUTPUT_DIR / f"sub-{subject_id}_{person}_badchannels_detected.fif"

            if not path.exists():
                print(f"  ERROR: Bad-channels detected file not found for {person}")
                continue

            raw = mne.io.read_raw_fif(str(path), preload=False)

            print(f"\n{person}:")
            print(f"  ✓ File exists: {path.name}")

            bads = raw.info.get("bads", [])
            print(f"  ✓ Bad channels marked: {len(bads)}")
            if bads:
                print(f"    Channels: {', '.join(bads)}")

            # Check QC report
            qc_report_path = config.QC_DIR / f"sub-{subject_id}_{person}_bad_channels_detect.tsv"
            if qc_report_path.exists():
                print(f"  ✓ QC report generated: {qc_report_path.name}")
                with open(qc_report_path, "r") as f:
                    lines = f.readlines()
                    print(f"    {len(lines) - 1} channels analyzed (excluding header)")
            else:
                print(f"  WARNING: QC report not found")

            # Sanity check: EEG channel count
            eeg_picks = mne.pick_types(raw.info, eeg=True, exclude="bads")
            print(f"  ✓ EEG channels (excluding bads): {len(eeg_picks)}")

            if len(bads) > len(mne.pick_types(raw.info, eeg=True)):
                print(
                    f"  WARNING: More bad channels than total EEG channels (possible error)"
                )

            print(f"  ✓ Sampling rate: {raw.info['sfreq']} Hz")

    print("\n" + "=" * 80)
    print("Sanity check completed.")
    print("=" * 80)


if __name__ == "__main__":
    sanity_check_bad_channels_detect()

