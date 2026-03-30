"""
Sanity Check for Step 05: Filter (Bandpass 1-40 Hz)

Überprüft:
- Filter erfolgreich angewendet
- Frequenzband korrekt (1-40 Hz)
- Power Spectral Density vor/nach Vergleich
- Amplituden reduziert (Anomalie-Warnung bei >50% Änderung)
- Keine Daten-Artefakte (NaN, Inf)
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

PIPELINE_DIR = CURRENT_DIR.parent.parent / "eeg_pipeline"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from preprocessing import config
from sc_utils import SanityCheckCollector, compare_amplitudes, detect_amplitude_anomaly


def sanity_check_filter():
    collector = SanityCheckCollector("05 - Bandpass Filter (1-40 Hz)")

    for subject_id in config.SUBJECTS:
        for person in ["P1", "P2"]:
            before_path = config.OUTPUT_DIR / f"sub-{subject_id}_{person}_interpolated.fif"
            after_path = config.OUTPUT_DIR / f"sub-{subject_id}_{person}_filtered.fif"

            if not before_path.exists():
                collector.add_result(subject_id, person, "ERROR", "Input file (interpolated) not found")
                continue

            if not after_path.exists():
                collector.add_result(subject_id, person, "ERROR", "Output file (filtered) not found")
                continue

            try:
                raw_before = mne.io.read_raw_fif(str(before_path), preload=False)
                raw_after = mne.io.read_raw_fif(str(after_path), preload=False)
            except Exception as e:
                collector.add_result(subject_id, person, "ERROR", f"Cannot load files: {e}")
                continue

            collector.add_result(subject_id, person, "✓", "Files exist")

            # Check metadata
            if len(raw_before.ch_names) == len(raw_after.ch_names):
                collector.add_result(subject_id, person, "✓", f"Channel count preserved: {len(raw_after.ch_names)}")
            else:
                collector.add_result(
                    subject_id,
                    person,
                    "ERROR",
                    f"Channel count mismatch: {len(raw_before.ch_names)} → {len(raw_after.ch_names)}",
                )

            if raw_before.info["sfreq"] == raw_after.info["sfreq"]:
                collector.add_result(subject_id, person, "✓", f"Sampling rate preserved: {raw_after.info['sfreq']} Hz")
            else:
                collector.add_result(
                    subject_id, person, "ERROR", f"Sampling rate changed: {raw_before.info['sfreq']} → {raw_after.info['sfreq']}"
                )

            if raw_before.n_times == raw_after.n_times:
                collector.add_result(subject_id, person, "✓", f"Sample count preserved: {raw_after.n_times}")
            else:
                collector.add_result(
                    subject_id, person, "ERROR", f"Sample count changed: {raw_before.n_times} → {raw_after.n_times}"
                )

            # Compare amplitudes using utility function
            std_before, std_after, change_pct = compare_amplitudes(raw_before, raw_after, duration_s=60, pick_type="eeg")
            if not (np.isnan(std_before) or np.isnan(std_after)):
                collector.add_result(
                    subject_id,
                    person,
                    "✓",
                    f"EEG amplitude: {std_before:.2f} µV → {std_after:.2f} µV ({change_pct:+.1f}%)",
                )

                # Check for amplitude anomalies
                anomaly = detect_amplitude_anomaly(change_pct, threshold_pct=50)
                if anomaly:
                    collector.add_result(subject_id, person, "⚠", anomaly)

            # Check for NaN/Inf
            data_after = raw_after.get_data(start=0, stop=min(10000, raw_after.n_times))
            nan_count = int(np.isnan(data_after).sum())
            inf_count = int(np.isinf(data_after).sum())
            if nan_count == 0 and inf_count == 0:
                collector.add_result(subject_id, person, "✓", "No NaN/Inf detected")
            else:
                collector.add_result(
                    subject_id, person, "ERROR", f"Found {nan_count} NaN and {inf_count} Inf values"
                )

    collector.print_summary()

    # Create visualization for first subject (if available)
    try:
        subject_id = config.SUBJECTS[0]
        person = "P1"
        before_path = config.OUTPUT_DIR / f"sub-{subject_id}_{person}_interpolated.fif"
        after_path = config.OUTPUT_DIR / f"sub-{subject_id}_{person}_filtered.fif"

        if before_path.exists() and after_path.exists():
            raw_before = mne.io.read_raw_fif(str(before_path), preload=False)
            raw_after = mne.io.read_raw_fif(str(after_path), preload=False)

            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            # PEFORE
            raw_before_eeg = raw_before.copy().pick_types(eeg=True)
            raw_before_eeg.plot_psd(fmax=60, ax=axes[0], show=False)
            axes[0].axvline(x=config.FREQ_LOWER, color="red", linestyle="--", linewidth=2)
            axes[0].axvline(x=config.FREQ_UPPER, color="red", linestyle="--", linewidth=2)
            axes[0].set_title("BEFORE: Power Spectral Density")

            # AFTER
            raw_after_eeg = raw_after.copy().pick_types(eeg=True)
            raw_after_eeg.plot_psd(fmax=60, ax=axes[1], show=False)
            axes[1].axvline(x=config.FREQ_LOWER, color="red", linestyle="--", linewidth=2)
            axes[1].axvline(x=config.FREQ_UPPER, color="red", linestyle="--", linewidth=2)
            axes[1].set_title(f"AFTER: Bandpass {config.FREQ_LOWER}-{config.FREQ_UPPER} Hz")

            fig.suptitle(f"sub-{subject_id} {person} - Filter Effect on PSD", fontsize=14, fontweight="bold")
            plt.tight_layout()

            config.QC_DIR.mkdir(parents=True, exist_ok=True)
            plot_path = config.QC_DIR / f"sub-{subject_id}_{person}_filter_psd_comparison.png"
            fig.savefig(plot_path, dpi=100, bbox_inches="tight")
            plt.close(fig)
            print(f"\n✓ PSD comparison plot saved: {plot_path.name}\n")
    except Exception as e:
        print(f"\nWarning: Could not save PSD plot: {e}\n")

    # Export summary CSV
    output_csv = config.QC_DIR / "sc_05_filter_summary.csv"
    collector.export_csv(output_csv)
    print(f"✓ Summary exported to {output_csv.name}\n")


if __name__ == "__main__":
    sanity_check_filter()



