"""
Sanity Check for Step 02: Rename & Set Montage

Überprüft:
- Kanäle korrekt umbenannt (BioSemi → 10-20)
- Montage gesetzt
- Elektroden-Positionen vorhanden
- Kanal-Metadaten intakt
"""
import sys
from pathlib import Path

import mne
import matplotlib.pyplot as plt

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

PIPELINE_DIR = CURRENT_DIR.parent / "eeg_pipeline"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from eeg_pipeline import config


def _save_montage_visualizations(raw, subject_id, person):
    config.QC_DIR.mkdir(parents=True, exist_ok=True)

    raw_eeg = raw.copy().pick("eeg")

    fig_2d = raw_eeg.plot_sensors(kind="topomap", show_names=True, show=False)
    fig_2d.suptitle(f"sub-{subject_id} {person} EEG sensor layout (2D)")
    out_2d = config.QC_DIR / f"sub-{subject_id}_{person}_montage_positions_2d.png"
    fig_2d.savefig(out_2d, dpi=200, bbox_inches="tight")
    plt.close(fig_2d)
    print(f"  ✓ Saved montage plot (2D): {out_2d.name}")

    try:
        fig_3d = raw_eeg.plot_sensors(kind="3d", show_names=False, show=False)
        fig_3d.suptitle(f"sub-{subject_id} {person} EEG sensor layout (3D)")
        out_3d = config.QC_DIR / f"sub-{subject_id}_{person}_montage_positions_3d.png"
        fig_3d.savefig(out_3d, dpi=200, bbox_inches="tight")
        plt.close(fig_3d)
        print(f"  ✓ Saved montage plot (3D): {out_3d.name}")
    except Exception as exc:
        print(f"  WARNING: Could not save 3D montage plot: {exc}")


def sanity_check_rename_montage():
    print("\n" + "=" * 80)
    print("SANITY CHECK: Step 02 - Rename & Set Montage")
    print("=" * 80)

    for subject_id in config.SUBJECTS:
        print(f"\n--- Checking subject {subject_id} ---\n")

        for person in ["P1", "P2"]:
            path = config.OUTPUT_DIR / f"sub-{subject_id}_{person}_renamed_montaged.fif"

            if not path.exists():
                print(f"  ERROR: Renamed+Montaged file not found for {person}: {path}")
                continue

            raw = mne.io.read_raw_fif(str(path), preload=False)

            print(f"\n{person}:")
            print(f"  ✓ File exists: {path.name}")

            eeg_picks = mne.pick_types(raw.info, eeg=True)
            eeg_names = [raw.ch_names[i] for i in eeg_picks]

            print(f"  ✓ EEG channels: {len(eeg_names)}")

            # Check if standard 10-20 names are present
            standard_10_20_names = {
                "Fp1",
                "Fp2",
                "Fz",
                "F3",
                "F4",
                "Cz",
                "C3",
                "C4",
                "Pz",
                "P3",
                "P4",
                "Oz",
            }
            found_standard = sum(1 for ch in eeg_names if ch in standard_10_20_names)
            print(f"    Standard 10-20 channels found: {found_standard}/{len(eeg_names)}")

            # Check montage
            if raw.info.get("dig") is not None and len(raw.info["dig"]) > 0:
                print(f"  ✓ Montage/electrode positions set: {len(raw.info['dig'])} digitization points")
                _save_montage_visualizations(raw, subject_id, person)
            else:
                print(f"  WARNING: No electrode positions found in montage")

            # Check for old prefixes
            prefix = config.PLAYER_PREFIX_MAP[person]
            old_prefix_count = sum(1 for ch in raw.ch_names if ch.startswith(prefix))
            if old_prefix_count > 0:
                print(f"  WARNING: {old_prefix_count} channels still have old prefix '{prefix}'")
            else:
                print(f"  ✓ Old prefixes successfully removed")

            print(f"  ✓ Sampling rate: {raw.info['sfreq']} Hz")
            print(f"  ✓ Duration: {raw.times[-1]:.2f}s")
            print(f"  ✓ First 5 EEG channels: {eeg_names[:5]}")

    print("\n" + "=" * 80)
    print("Sanity check completed.")
    print("=" * 80)


if __name__ == "__main__":
    sanity_check_rename_montage()
