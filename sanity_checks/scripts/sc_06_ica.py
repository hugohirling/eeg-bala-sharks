"""
Sanity Check for Step 06: ICA Component Removal

Creates verification plots for the ICA decomposition and bad component removal:
- Component topomaps showing spatial patterns of all ICA components
- Variance explained by each component (with bad components highlighted)
- Time-domain signatures of bad components (2-second samples)
- Before/after EEG PSD comparison to verify artifact removal
- Summary of removed components and their characteristics

Usage:
    python sanity_checks/scripts/sc_06_ica.py [--subjects 01,02] [--duration 30]

Options:
    --subjects: Comma-separated subject IDs (default: first 2)
    --duration: Duration in seconds used for PSD comparison (default: 30)

REASONING:
- Purpose: show why certain components are removed and how that choice changes the data quality.
- Reproducibility: ICLabel thresholds and artifact classes are defined in preprocessing/config.py, making the decision rule explicit.
- Parameter notes: the QC summary focuses on removed-component count, ICLabel probabilities, and before/after amplitude changes because these justify the ICA choice.
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
from plots.sc_06_ica_plots import (
    collect_component_metadata,
    component_summary_text,
    plot_bad_component_timeseries,
    plot_component_topomaps,
    plot_psd_comparison_ica,
    plot_variance_explained,
)
from helpers.sc_cli import add_duration_argument, add_mode_argument, add_subjects_argument, resolve_subjects
from helpers.sc_config import DEFAULT_PERSONS
from helpers.sc_utils import SanityCheckCollector, compare_amplitudes, detect_amplitude_anomaly


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Visualize ICA decomposition and component removal",
    )
    add_subjects_argument(parser)
    add_mode_argument(parser, default="check")
    add_duration_argument(parser, default=30)
    return parser.parse_args(argv)


def sanity_check_ica(subjects, duration, *, run_visualizations=True):
    collector = SanityCheckCollector("06 - ICA Component Removal")
    collector.set_step_context(
        purpose="ICA should isolate ocular or other stereotyped artifacts so they can be removed with an explicit and reviewable decision rule.",
        reproducibility="Component labeling uses config.ICA_LABEL_METHOD, config.ICA_ARTIFACT_LABELS, and config.ICA_LABEL_MIN_PROBA, so the exclusion logic is transparent and repeatable.",
        parameter_notes=[
            f"ICLabel minimum probability = {float(config.ICA_LABEL_MIN_PROBA):.2f} for artifact rejection.",
            f"Artifact labels considered removable: {', '.join(config.ICA_ARTIFACT_LABELS)}.",
        ],
    )

    print("\n" + "=" * 80)
    print("VISUALIZATION: Step 06 - ICA Decomposition Verification")
    print("=" * 80)
    print(f"Subjects: {', '.join(subjects)}")
    print(f"Duration: {duration}s")
    print(f"Output: {config.QC_DIR}")
    print("=" * 80 + "\n")

    for subject_id in subjects:
        print(f"\n--- Subject {subject_id} ---")
        for person in DEFAULT_PERSONS:
            before_path = config.OUTPUT_DIR / f"sub-{subject_id}_{person}_filtered.fif"
            after_path = config.OUTPUT_DIR / f"sub-{subject_id}_{person}_ica_cleaned.fif"
            ica_path = config.ICA_DIR / f"sub-{subject_id}_{person}_ica.fif"

            if not before_path.exists():
                collector.add_result(subject_id, person, "ERROR", "Input file (filtered) not found")
                continue
            if not after_path.exists():
                collector.add_result(subject_id, person, "ERROR", "Output file (ICA) not found")
                continue
            if not ica_path.exists():
                collector.add_result(subject_id, person, "ERROR", "ICA decomposition file not found")
                continue

            try:
                raw_before = mne.io.read_raw_fif(str(before_path), preload=False, verbose=False)
                raw_after = mne.io.read_raw_fif(str(after_path), preload=False, verbose=False)
                ica = mne.preprocessing.read_ica(str(ica_path))
            except Exception as exc:
                collector.add_result(subject_id, person, "ERROR", f"Cannot load files: {exc}")
                continue

            component_meta = collect_component_metadata(raw_before, ica)

            collector.add_result(subject_id, person, "âœ“", "Files exist")
            
            # Verify channel count
            if len(raw_before.ch_names) == len(raw_after.ch_names):
                collector.add_result(subject_id, person, "âœ“", f"Channel count preserved: {len(raw_after.ch_names)}")
            else:
                collector.add_result(subject_id, person, "ERROR", f"Channel count mismatch: {len(raw_before.ch_names)} -> {len(raw_after.ch_names)}")
            
            # Verify sampling rate
            if raw_before.info["sfreq"] == raw_after.info["sfreq"]:
                collector.add_result(subject_id, person, "âœ“", f"Sampling rate preserved: {raw_after.info['sfreq']} Hz")
            else:
                collector.add_result(subject_id, person, "ERROR", f"Sampling rate changed: {raw_before.info['sfreq']} -> {raw_after.info['sfreq']}")
            
            # Verify sample count
            if raw_before.n_times == raw_after.n_times:
                collector.add_result(subject_id, person, "âœ“", f"Sample count preserved: {raw_after.n_times}")
            else:
                collector.add_result(subject_id, person, "ERROR", f"Sample count changed: {raw_before.n_times} -> {raw_after.n_times}")
            
            # Check ICA components
            if ica.n_components > 0:
                collector.add_result(subject_id, person, "âœ“", f"ICA fitted with {ica.n_components} components")
            else:
                collector.add_result(subject_id, person, "ERROR", "ICA has zero components")
            
            # Check for removed components
            if len(ica.exclude) > 0:
                collector.add_result(
                    subject_id,
                    person,
                    "âœ“",
                    f"Removed {len(ica.exclude)}/{ica.n_components} components: {component_summary_text(component_meta, limit=20)}",
                )
            else:
                collector.add_result(subject_id, person, "âš ", "No components marked for removal")
            
            # Compare amplitudes
            std_before, std_after, change_pct = compare_amplitudes(raw_before, raw_after, duration_s=60, pick_type="eeg")
            if not (np.isnan(std_before) or np.isnan(std_after)):
                collector.add_result(subject_id, person, "âœ“", f"EEG amplitude: {std_before:.2f} ÂµV -> {std_after:.2f} ÂµV ({change_pct:+.1f}%)")
                if change_pct > 20.0:
                    collector.add_result(subject_id, person, "âš ", "Larger than expected amplitude increase after ICA")
            
            # Check for NaN/Inf
            data_after = raw_after.get_data(start=0, stop=min(10000, raw_after.n_times))
            nan_count = int(np.isnan(data_after).sum())
            inf_count = int(np.isinf(data_after).sum())
            if nan_count == 0 and inf_count == 0:
                collector.add_result(subject_id, person, "âœ“", "No NaN/Inf detected")
            else:
                collector.add_result(subject_id, person, "ERROR", f"Found {nan_count} NaN and {inf_count} Inf values")

            if run_visualizations:
                print(f"\n  {person}:")
                plot_component_topomaps(ica, raw_before, component_meta, subject_id, person, config.QC_DIR)
                plot_variance_explained(ica, raw_before, component_meta, subject_id, person, config.QC_DIR)
                plot_bad_component_timeseries(ica, raw_before, component_meta, subject_id, person, config.QC_DIR)
                plot_psd_comparison_ica(raw_before, raw_after, component_meta, subject_id, person, duration, config.QC_DIR)

    collector.print_summary()
    output_csv = config.QC_DIR / "sc_06_ica_summary.csv"
    collector.export_csv(output_csv)
    print(f"\n[OK] Summary exported to {output_csv.name}\n")


def main(argv=None):
    args = parse_args(argv)
    check_subjects = resolve_subjects(args.subjects, config.SUBJECTS, mode="check")
    viz_subjects = resolve_subjects(args.subjects, config.SUBJECTS, mode="viz")

    if args.mode == "check":
        sanity_check_ica(check_subjects, args.duration, run_visualizations=False)
    elif args.mode == "viz":
        sanity_check_ica(viz_subjects, args.duration, run_visualizations=True)
    else:
        sanity_check_ica(check_subjects, args.duration, run_visualizations=True)


if __name__ == "__main__":
    main()


