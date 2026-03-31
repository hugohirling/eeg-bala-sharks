"""
Sanity Check Visualization for Step 05: Filter (Bandpass 1-40 Hz)

Creates verification plots for the filtering step:
- Before/after PSD comparison with passband and stopband emphasis
- Before/after time series of one representative EEG channel
- Bandpower summary showing attenuation below 1 Hz and above 40 Hz

Usage:
    python sanity_checks/scripts/sc_05_filter.py [--mode check|viz|both] [--subjects 01,02] [--duration 30]

Options:
    --mode: Run textual checks, visualizations, or both
    --subjects: Comma-separated subject IDs (default: first 2)
    --duration: Duration in seconds used for PSD and time series (default: 30)

REASONING:
- Purpose: document why the chosen passband keeps task-relevant EEG content while attenuating slow drift and high-frequency noise.
- Reproducibility: filter cutoffs are read from preprocessing/config.py, so the same config should reproduce the same passband/stopband checks.
- Parameter notes: the QC figures emphasize <1 Hz, 1-40 Hz, and >40 Hz because those bands directly reflect the chosen bandpass settings.
"""

import argparse
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

from preprocessing import config
from plots.sc_05_filter_plots import (
    HIGH_BAND,
    LOW_BAND,
    PASS_BAND,
    band_mean as _band_mean,
    plot_bandpower_summary,
    plot_psd_comparison,
    plot_timeseries_comparison,
)
from helpers.sc_cli import add_duration_argument, add_mode_argument, add_subjects_argument, resolve_subjects
from helpers.sc_config import DEFAULT_PERSONS
from helpers.sc_signal import compute_psd as _compute_psd
from helpers.sc_utils import SanityCheckCollector, compare_amplitudes, detect_amplitude_anomaly

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Filter sanity check (step 05): checks and optional visualizations.",
    )
    add_subjects_argument(parser)
    add_mode_argument(parser)
    add_duration_argument(parser, default=30)
    return parser.parse_args(argv)

def sanity_check_filter(subjects, duration, run_visualizations=True):
    collector = SanityCheckCollector("05 - Bandpass Filter (1-40 Hz)")
    collector.set_step_context(
        purpose="Filtering should suppress slow drift and high-frequency noise while preserving the interpretable EEG band used later in the project.",
        reproducibility="The passband is controlled by config.FREQ_LOWER and config.FREQ_UPPER, so the same inputs and config should yield the same PSD changes.",
        parameter_notes=[
            f"Low cutoff = {config.FREQ_LOWER} Hz to reduce slow drifts before ICA and decoding.",
            f"High cutoff = {config.FREQ_UPPER} Hz to keep conventional EEG content while attenuating high-frequency noise.",
        ],
    )

    print("\n" + "=" * 80)
    print("VISUALIZATION: Step 05 - Filter Verification")
    print("=" * 80)
    print(f"Subjects: {', '.join(subjects)}")
    print(f"Duration: {duration}s")
    print(f"Output: {config.QC_DIR}")
    print("=" * 80 + "\n")

    for subject_id in subjects:
        print(f"\n--- Subject {subject_id} ---")
        for person in DEFAULT_PERSONS:
            before_path = config.OUTPUT_DIR / f"sub-{subject_id}_{person}_interpolated.fif"
            after_path = config.OUTPUT_DIR / f"sub-{subject_id}_{person}_filtered.fif"

            if not before_path.exists():
                collector.add_result(subject_id, person, "ERROR", "Input file (interpolated) not found")
                continue
            if not after_path.exists():
                collector.add_result(subject_id, person, "ERROR", "Output file (filtered) not found")
                continue

            try:
                raw_before = mne.io.read_raw_fif(str(before_path), preload=False, verbose=False)
                raw_after = mne.io.read_raw_fif(str(after_path), preload=False, verbose=False)
            except Exception as exc:
                collector.add_result(subject_id, person, "ERROR", f"Cannot load files: {exc}")
                continue

            collector.add_result(subject_id, person, "âœ“", "Files exist")
            if len(raw_before.ch_names) == len(raw_after.ch_names):
                collector.add_result(subject_id, person, "âœ“", f"Channel count preserved: {len(raw_after.ch_names)}")
            else:
                collector.add_result(subject_id, person, "ERROR", f"Channel count mismatch: {len(raw_before.ch_names)} -> {len(raw_after.ch_names)}")
            if raw_before.info["sfreq"] == raw_after.info["sfreq"]:
                collector.add_result(subject_id, person, "âœ“", f"Sampling rate preserved: {raw_after.info['sfreq']} Hz")
            else:
                collector.add_result(subject_id, person, "ERROR", f"Sampling rate changed: {raw_before.info['sfreq']} -> {raw_after.info['sfreq']}")
            if raw_before.n_times == raw_after.n_times:
                collector.add_result(subject_id, person, "âœ“", f"Sample count preserved: {raw_after.n_times}")
            else:
                collector.add_result(subject_id, person, "ERROR", f"Sample count changed: {raw_before.n_times} -> {raw_after.n_times}")

            std_before, std_after, change_pct = compare_amplitudes(raw_before, raw_after, duration_s=60, pick_type="eeg")
            if not (np.isnan(std_before) or np.isnan(std_after)):
                collector.add_result(subject_id, person, "âœ“", f"EEG amplitude: {std_before:.2f} ÂµV -> {std_after:.2f} ÂµV ({change_pct:+.1f}%)")
                anomaly = detect_amplitude_anomaly(change_pct, threshold_pct=50)
                if anomaly:
                    collector.add_result(subject_id, person, "âš ", anomaly)

            freqs_before, mean_before, _, _ = _compute_psd(raw_before, duration)
            freqs_after, mean_after, _, _ = _compute_psd(raw_after, duration)
            if freqs_before is not None and freqs_after is not None:
                low_before = _band_mean(mean_before, freqs_before, LOW_BAND)
                low_after = _band_mean(mean_after, freqs_after, LOW_BAND)
                pass_before = _band_mean(mean_before, freqs_before, PASS_BAND)
                pass_after = _band_mean(mean_after, freqs_after, PASS_BAND)
                high_before = _band_mean(mean_before, freqs_before, HIGH_BAND)
                high_after = _band_mean(mean_after, freqs_after, HIGH_BAND)
                if low_before > 0 and pass_before > 0 and high_before > 0:
                    low_change = (low_after - low_before) / low_before * 100.0
                    pass_change = (pass_after - pass_before) / pass_before * 100.0
                    high_change = (high_after - high_before) / high_before * 100.0
                    collector.add_result(subject_id, person, "âœ“", f"Bandpower change low/pass/high: {low_change:+.1f}% / {pass_change:+.1f}% / {high_change:+.1f}%")
                    if low_change > -20.0:
                        collector.add_result(subject_id, person, "âš ", "Weak attenuation below 1 Hz")
                    if high_change > -20.0:
                        collector.add_result(subject_id, person, "âš ", "Weak attenuation above 40 Hz")

            data_after = raw_after.get_data(start=0, stop=min(10000, raw_after.n_times))
            nan_count = int(np.isnan(data_after).sum())
            inf_count = int(np.isinf(data_after).sum())
            if nan_count == 0 and inf_count == 0:
                collector.add_result(subject_id, person, "âœ“", "No NaN/Inf detected")
            else:
                collector.add_result(subject_id, person, "ERROR", f"Found {nan_count} NaN and {inf_count} Inf values")

            if run_visualizations:
                print(f"\n  {person}:")
                plot_psd_comparison(raw_before, raw_after, subject_id, person, duration, config.QC_DIR)
                plot_timeseries_comparison(raw_before, raw_after, subject_id, person, duration, config.QC_DIR)
                plot_bandpower_summary(raw_before, raw_after, subject_id, person, duration, config.QC_DIR)

    collector.print_summary()
    output_csv = config.QC_DIR / "sc_05_filter_summary.csv"
    collector.export_csv(output_csv)
    print(f"\nâœ“ Summary exported to {output_csv.name}\n")


def main(argv=None):
    args = parse_args(argv)
    mode = args.mode
    subjects = resolve_subjects(args.subjects, config.SUBJECTS, mode=mode)
    run_visualizations = mode in ("viz", "both")
    sanity_check_filter(subjects, args.duration, run_visualizations=run_visualizations)


if __name__ == "__main__":
    main()




