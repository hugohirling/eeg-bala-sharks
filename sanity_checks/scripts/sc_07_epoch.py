"""
Sanity Check for Step 07: Epoching

Überprüft:
- Epochs erfolgreich erstellt
- Event-Anzahl und -Typen plausibel
- Epoch-Größe und Dimensionen
- Baseline-Korrektur vorhanden
- Keine NaN/Inf-Werte
- Anomalie-Detektion: zu wenige/viele Epochs
"""
import sys
from pathlib import Path

import numpy as np
import mne

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

PIPELINE_DIR = CURRENT_DIR.parent.parent / "eeg_pipeline"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from preprocessing import config
from sc_utils import SanityCheckCollector


def sanity_check_epoch():
    collector = SanityCheckCollector("07 - Epoching")

    for subject_id in config.SUBJECTS:
        for person in ["P1", "P2"]:
            raw_path = config.OUTPUT_DIR / f"sub-{subject_id}_{person}_ica_cleaned.fif"
            epoch_path = config.OUTPUT_DIR / f"sub-{subject_id}_{person}_epoch.fif"

            if not raw_path.exists():
                collector.add_result(subject_id, person, "ERROR", "Input file (ica_cleaned) not found")
                continue

            if not epoch_path.exists():
                collector.add_result(subject_id, person, "ERROR", "Output file (epoch) not found")
                continue

            try:
                raw = mne.io.read_raw_fif(str(raw_path), preload=False)
                epochs = mne.read_epochs(str(epoch_path), preload=False)
            except Exception as e:
                collector.add_result(subject_id, person, "ERROR", f"Cannot load files: {e}")
                continue

            # Check basic structure
            collector.add_result(subject_id, person, "✓", f"Files exist")
            collector.add_result(subject_id, person, "✓", f"Number of epochs: {len(epochs)}")
            
            # Check if epoch count is reasonable
            if len(epochs) == 0:
                collector.add_result(subject_id, person, "ERROR", "No epochs created (count = 0)")
            elif len(epochs) < 10:
                collector.add_result(subject_id, person, "⚠", f"Very few epochs ({len(epochs)} < 10) - may be incomplete data")
            elif len(epochs) > config.MAX_EPOCHS:
                collector.add_result(subject_id, person, "⚠", f"Epoch count exceeds MAX_EPOCHS ({len(epochs)} > {config.MAX_EPOCHS})")
            else:
                collector.add_result(subject_id, person, "✓", f"Epoch count within expected range")

            # Check event types
            event_types = list(epochs.event_id.keys())
            if event_types:
                collector.add_result(subject_id, person, "✓", f"Event types found: {', '.join(event_types)}")
            else:
                collector.add_result(subject_id, person, "ERROR", "No event types defined")

            # Check time window
            tmin_actual = epochs.times[0]
            tmax_actual = epochs.times[-1]
            expected_duration = config.EPOCH_TMAX - config.EPOCH_TMIN
            actual_duration = tmax_actual - tmin_actual

            collector.add_result(
                subject_id,
                person,
                "✓",
                f"Time window: [{tmin_actual:.3f}, {tmax_actual:.3f}] s (expected {expected_duration:.3f} s)",
            )

            if abs(actual_duration - expected_duration) > 0.01:
                collector.add_result(
                    subject_id,
                    person,
                    "⚠",
                    f"Duration mismatch: {actual_duration:.3f} vs {expected_duration:.3f} s",
                )

            # Check sampling rate
            sfreq = epochs.info["sfreq"]
            collector.add_result(subject_id, person, "✓", f"Sampling rate: {sfreq} Hz")

            # Check dimensions
            n_channels = len(epochs.ch_names)
            n_samples = epochs.get_data().shape[2] if len(epochs) > 0 else 0
            collector.add_result(subject_id, person, "✓", f"Dimensions: ({len(epochs)} epochs, {n_channels} channels, {n_samples} samples)")

            # Check for baseline correction
            if epochs.baseline is not None:
                collector.add_result(subject_id, person, "✓", f"Baseline period: {epochs.baseline}")
            else:
                collector.add_result(subject_id, person, "⚠", "No baseline correction applied")

            # Check bad channels
            bads = epochs.info.get("bads", [])
            if len(bads) == 0:
                collector.add_result(subject_id, person, "✓", "No bad channels marked")
            else:
                bad_pct = len(bads) / n_channels * 100 if n_channels > 0 else 0
                collector.add_result(subject_id, person, "⚠", f"Bad channels marked: {len(bads)}/{n_channels} ({bad_pct:.1f}%)")

            # Check for data integrity
            if len(epochs) > 0:
                data = epochs.get_data()
                nan_count = int(np.isnan(data).sum())
                inf_count = int(np.isinf(data).sum())
                
                if nan_count == 0 and inf_count == 0:
                    collector.add_result(subject_id, person, "✓", "No NaN/Inf values")
                else:
                    collector.add_result(
                        subject_id,
                        person,
                        "ERROR",
                        f"Data integrity issue: {nan_count} NaN and {inf_count} Inf values",
                    )

                # Check for extreme values
                data_abs = np.abs(data)
                max_val = np.nanmax(data_abs)
                if max_val > 1e4:  # > 10 mV in V units
                    collector.add_result(
                        subject_id,
                        person,
                        "⚠",
                        f"Large amplitude detected: {max_val*1e6:.0f} µV (possible artifact)",
                    )

    collector.print_summary()

    # Export summary CSV
    output_csv = config.QC_DIR / "sc_07_epoch_summary.csv"
    collector.export_csv(output_csv)
    print(f"\n✓ Summary exported to {output_csv.name}\n")


if __name__ == "__main__":
    sanity_check_epoch()

