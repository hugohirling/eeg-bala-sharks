"""
Sanity Check for Step 06: ICA Artifact Removal

Überprüft:
- ICA Komponenten extrahiert
- EOG Artefakte erkannt
- Amplituden-Reduktion plausibel
- ICA-Zerlegung gespeichert
- Gezielt: ausgeschlossene Komponenten, Blink-Effekt auf EEG, Frontal-Overlay
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
            t_end = min(120, raw_before.times[-1])
            t_idx_end = int(t_end * raw_before.info["sfreq"])
            if len(eeg_picks) > 0:
                data_before = raw_before.get_data(picks=eeg_picks, start=0, stop=t_idx_end)
                data_after = raw_after.get_data(picks=eeg_picks, start=0, stop=t_idx_end)

                std_before = np.std(data_before)
                std_after = np.std(data_after)
                std_before_uv = std_before * 1e6
                std_after_uv = std_after * 1e6

                print(f"  ✓ EEG amplitude (first 120s):")
                print(f"    Before ICA - Std: {std_before_uv:.3f} µV")
                print(f"    After ICA - Std: {std_after_uv:.3f} µV")

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
                eog_std_before_uv = eog_std_before * 1e6
                eog_std_after_uv = eog_std_after * 1e6
                eog_reduction = (1 - eog_std_after / eog_std_before) * 100

                print(f"  ✓ EOG amplitude:")
                print(f"    Before ICA - Std: {eog_std_before_uv:.3f} µV")
                print(f"    After ICA - Std: {eog_std_after_uv:.3f} µV")
                print(f"    Reduction: {eog_reduction:.1f}%")

    # Create targeted ICA QC plots
    print(f"\n  Creating targeted ICA QC plots...")
    try:
        subject_id = config.SUBJECTS[0]
        person = "P1"
        before_path = config.OUTPUT_DIR / f"sub-{subject_id}_{person}_filtered.fif"
        after_path = config.OUTPUT_DIR / f"sub-{subject_id}_{person}_ica_cleaned.fif"
        ica_path = config.QC_DIR / f"sub-{subject_id}_{person}_ica.fif"

        if before_path.exists() and after_path.exists() and ica_path.exists():
            raw_before = mne.io.read_raw_fif(str(before_path), preload=False)
            raw_after = mne.io.read_raw_fif(str(after_path), preload=False)
            ica = mne.preprocessing.read_ica(str(ica_path))

            # 1) Topomaps of excluded ICA components
            if len(ica.exclude) > 0:
                figs = ica.plot_components(picks=ica.exclude, show=False)
                if not isinstance(figs, list):
                    figs = [figs]
                for idx, fig in enumerate(figs, start=1):
                    comp_path = config.QC_DIR / f"sub-{subject_id}_{person}_ica_excluded_components_{idx}.png"
                    fig.savefig(comp_path, dpi=100, bbox_inches="tight")
                    plt.close(fig)
                print(f"  ✓ ICA excluded-component topomaps saved ({len(figs)} file(s))")
            else:
                print("  INFO: No ICA components marked as excluded")

            # 2) Blink-locked EEG GFP before/after (same blink events, robust extraction)
            eog_picks = mne.pick_types(raw_before.info, eog=True)
            if len(eog_picks) > 0:
                eog_name = raw_before.ch_names[eog_picks[0]]
                eog_events = mne.preprocessing.find_eog_events(raw_before, ch_name=eog_name, verbose=False)
                eeg_picks = mne.pick_types(raw_before.info, eeg=True)

                if len(eog_events) > 0 and len(eeg_picks) > 0:
                    sfreq = raw_before.info["sfreq"]
                    tmin = -0.5
                    tmax = 0.5
                    n_pre = int(abs(tmin) * sfreq)
                    n_post = int(tmax * sfreq)
                    win_len = n_pre + n_post + 1

                    eeg_before_all = raw_before.get_data(picks=eeg_picks)
                    eeg_after_all = raw_after.get_data(picks=eeg_picks)

                    segments_before = []
                    segments_after = []
                    for event_sample in eog_events[:, 0]:
                        start = int(event_sample) - n_pre
                        stop = int(event_sample) + n_post + 1
                        if start < 0 or stop > eeg_before_all.shape[1]:
                            continue
                        segments_before.append(eeg_before_all[:, start:stop])
                        segments_after.append(eeg_after_all[:, start:stop])

                    if len(segments_before) > 0:
                        arr_before = np.stack(segments_before, axis=0)
                        arr_after = np.stack(segments_after, axis=0)

                        evoked_before = arr_before.mean(axis=0)
                        evoked_after = arr_after.mean(axis=0)
                        gfp_before_uv = evoked_before.std(axis=0) * 1e6
                        gfp_after_uv = evoked_after.std(axis=0) * 1e6
                        times = np.linspace(tmin, tmax, win_len)

                        fig, ax = plt.subplots(1, 1, figsize=(10, 4))
                        ax.plot(times, gfp_before_uv, label="Before ICA", color="tab:red", linewidth=2)
                        ax.plot(times, gfp_after_uv, label="After ICA", color="tab:green", linewidth=2)
                        ax.axvline(0, color="k", linestyle="--", linewidth=1)
                        ax.set_title(f"Blink-locked EEG GFP ({eog_name})")
                        ax.set_xlabel("Time (s)")
                        ax.set_ylabel("GFP (µV)")
                        ax.legend()
                        ax.grid(alpha=0.3)
                        plt.tight_layout()
                        blink_path = config.QC_DIR / f"sub-{subject_id}_{person}_ica_blink_locked_gfp.png"
                        fig.savefig(blink_path, dpi=120, bbox_inches="tight")
                        plt.close(fig)
                        print(f"  ✓ Blink-locked EEG GFP saved: {blink_path.name}")
                    else:
                        print("  INFO: Blink events were too close to edges, skipping blink-locked QC")
                else:
                    print("  INFO: No blink events or EEG picks found, skipping blink-locked QC")
            else:
                print("  INFO: No EOG channel found, skipping blink-locked QC")

            # 3) Frontal EEG channel QC: separate subplots + delta
            eeg_names = [raw_before.ch_names[i] for i in mne.pick_types(raw_before.info, eeg=True)]
            preferred = ["Fp1", "Fpz", "Fp2", "AFz", "Fz"]
            frontal_ch = next((ch for ch in preferred if ch in eeg_names), eeg_names[0] if eeg_names else None)

            if frontal_ch is not None:
                sfreq = raw_before.info["sfreq"]
                duration_sec = min(20, raw_before.times[-1])
                stop_idx = int(duration_sec * sfreq)
                times = np.arange(stop_idx) / sfreq

                before_uv = raw_before.get_data(picks=[frontal_ch], start=0, stop=stop_idx)[0] * 1e6
                after_uv = raw_after.get_data(picks=[frontal_ch], start=0, stop=stop_idx)[0] * 1e6
                delta_uv = before_uv - after_uv

                fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

                axes[0].plot(times, before_uv, color="tab:red", linewidth=1)
                axes[0].set_title(f"Before ICA ({frontal_ch})")
                axes[0].set_ylabel("Amplitude (µV)")
                axes[0].grid(alpha=0.3)

                axes[1].plot(times, after_uv, color="tab:green", linewidth=1)
                axes[1].set_title(f"After ICA ({frontal_ch})")
                axes[1].set_ylabel("Amplitude (µV)")
                axes[1].grid(alpha=0.3)

                axes[2].plot(times, delta_uv, color="tab:blue", linewidth=1)
                axes[2].axhline(0, color="k", linestyle="--", linewidth=0.8)
                axes[2].set_title(f"Delta: Before - After ({frontal_ch})")
                axes[2].set_xlabel("Time (s)")
                axes[2].set_ylabel("Delta (µV)")
                axes[2].grid(alpha=0.3)

                plt.tight_layout()
                frontal_qc_path = config.QC_DIR / f"sub-{subject_id}_{person}_ica_frontal_qc_{frontal_ch}.png"
                fig.savefig(frontal_qc_path, dpi=120, bbox_inches="tight")
                plt.close(fig)
                print(f"  ✓ Frontal EEG QC (before/after/delta) saved: {frontal_qc_path.name}")
            else:
                print("  INFO: No EEG channels found, skipping frontal QC")
        else:
            print("  INFO: Missing filtered/ica_cleaned/ica files for targeted plot generation")
    except Exception as e:
        print(f"  Could not save ICA plots: {e}")

    print("\n" + "=" * 80)
    print("Sanity check completed.")
    print("=" * 80)


if __name__ == "__main__":
    sanity_check_ica()
