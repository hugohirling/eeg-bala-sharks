"""
Sanity Check for Step 07: Epoching

Ueberprueft:
- Epochs erfolgreich erstellt
- Event-Anzahl und -Typen plausibel
- Epoch-Groesse und Dimensionen
- Baseline-Korrektur vorhanden
- Keine NaN/Inf-Werte
- Anomalie-Detektion: zu wenige/viele Epochs

REASONING:
- Purpose: ensure that continuous recordings were segmented into analysis-ready trials with the expected timing and event structure.
- Reproducibility: epoch timing is fully determined by config.EPOCH_TMIN, config.EPOCH_TMAX, and the event definitions stored in the data.
- Parameter notes: unusually low or high epoch counts are warnings because they often indicate missing triggers, duplicate events, or overly permissive event selection.
"""
import sys
import argparse
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
from helpers.sc_cli import add_mode_argument, add_subjects_argument, resolve_subjects
from helpers.sc_utils import SanityCheckCollector, seems_correct_because, strange_because


def sanity_check_epoch(subjects):
    collector = SanityCheckCollector("07 - Epoching")
    collector.set_step_context(
        purpose="Epoching converts continuous data into event-locked trials, so timing and event definitions need to be explicitly justified and checked.",
        reproducibility="The epoch window is controlled by preprocessing/config.py, which means the same triggers and config should create the same trial lengths on another machine.",
        parameter_notes=[
            f"Expected epoch duration is config.EPOCH_TMAX - config.EPOCH_TMIN = {config.EPOCH_TMAX - config.EPOCH_TMIN:.3f} s.",
            f"Epoch counts above config.MAX_EPOCHS = {config.MAX_EPOCHS} are flagged because duplicated or unexpected triggers become plausible.",
        ],
    )

    for subject_id in subjects:
        for person in ["P1", "P2"]:
            raw_path = config.OUTPUT_DIR / f"sub-{subject_id}_{person}_ica_cleaned.fif"
            epoch_path = config.OUTPUT_DIR / f"sub-{subject_id}_{person}_epoch.fif"

            if not raw_path.exists():
                collector.add_result(
                    subject_id,
                    person,
                    "ERROR",
                    "Input file (ica_cleaned) not found",
                    category="file_io",
                    rationale=strange_because("epoching depends on the ICA-cleaned continuous recording and cannot be validated without it"),
                )
                continue

            if not epoch_path.exists():
                collector.add_result(
                    subject_id,
                    person,
                    "ERROR",
                    "Output file (epoch) not found",
                    category="file_io",
                    rationale=strange_because("the final preprocessing stage should produce an epoch file for downstream decoding"),
                )
                continue

            try:
                raw = mne.io.read_raw_fif(str(raw_path), preload=False)
                epochs = mne.read_epochs(str(epoch_path), preload=False)
            except Exception as e:
                collector.add_result(
                    subject_id,
                    person,
                    "ERROR",
                    f"Cannot load files: {e}",
                    category="file_io",
                    rationale=strange_because("sanity checks must be able to reopen both the continuous and epoched files to be reproducible"),
                )
                continue

            # Check basic structure
            collector.add_result(
                subject_id,
                person,
                "âœ“",
                "Files exist",
                category="file_io",
                rationale=seems_correct_because("the epoch file and its source continuous file are both available for direct comparison"),
            )
            collector.add_result(
                subject_id,
                person,
                "âœ“",
                f"Number of epochs: {len(epochs)}",
                category="events",
                rationale=seems_correct_because("epoch count is the first quick proxy for whether event extraction behaved sensibly"),
            )
            
            # Check if epoch count is reasonable
            if len(epochs) == 0:
                collector.add_result(
                    subject_id,
                    person,
                    "ERROR",
                    "No epochs created (count = 0)",
                    category="events",
                    rationale=strange_because("a valid recording with events should yield at least some trials for the task"),
                )
            elif len(epochs) < 10:
                collector.add_result(
                    subject_id,
                    person,
                    "âš ",
                    f"Very few epochs ({len(epochs)} < 10)",
                    category="events",
                    rationale=strange_because("such a low count often reflects missing triggers or an incomplete recording rather than a normal participant"),
                )
            elif len(epochs) > config.MAX_EPOCHS:
                collector.add_result(
                    subject_id,
                    person,
                    "âš ",
                    f"Epoch count exceeds MAX_EPOCHS ({len(epochs)} > {config.MAX_EPOCHS})",
                    category="events",
                    rationale=strange_because("too many epochs can indicate duplicate triggers or overly broad event inclusion criteria"),
                    parameter_note=f"Upper reference comes from config.MAX_EPOCHS = {config.MAX_EPOCHS}.",
                )
            else:
                collector.add_result(
                    subject_id,
                    person,
                    "âœ“",
                    "Epoch count within expected range",
                    category="events",
                    rationale=seems_correct_because("the number of extracted trials is plausible for the task and recording length"),
                )

            # Check event types
            event_types = list(epochs.event_id.keys())
            if event_types:
                collector.add_result(
                    subject_id,
                    person,
                    "âœ“",
                    f"Event types found: {', '.join(event_types)}",
                    category="events",
                    rationale=seems_correct_because("a non-empty event dictionary is required for interpretable condition-wise analyses"),
                )
            else:
                collector.add_result(
                    subject_id,
                    person,
                    "ERROR",
                    "No event types defined",
                    category="events",
                    rationale=strange_because("epochs without named events are hard to interpret and nearly impossible to reproduce in later analyses"),
                )

            # Check time window
            tmin_actual = epochs.times[0]
            tmax_actual = epochs.times[-1]
            expected_duration = config.EPOCH_TMAX - config.EPOCH_TMIN
            actual_duration = tmax_actual - tmin_actual

            collector.add_result(
                subject_id,
                person,
                "âœ“",
                f"Time window: [{tmin_actual:.3f}, {tmax_actual:.3f}] s (expected {expected_duration:.3f} s)",
                category="temporal_integrity",
                rationale=seems_correct_because("the epoch span should match the analysis window defined in the preprocessing config"),
                parameter_note=f"Expected duration derived from config.EPOCH_TMIN={config.EPOCH_TMIN} and config.EPOCH_TMAX={config.EPOCH_TMAX}.",
            )

            if abs(actual_duration - expected_duration) > 0.01:
                collector.add_result(
                    subject_id,
                    person,
                    "âš ",
                    f"Duration mismatch: {actual_duration:.3f} vs {expected_duration:.3f} s",
                    category="temporal_integrity",
                    rationale=strange_because("a changed epoch length would alter the baseline and post-event windows used for later decoding"),
                )

            # Check sampling rate
            sfreq = epochs.info["sfreq"]
            collector.add_result(
                subject_id,
                person,
                "âœ“",
                f"Sampling rate: {sfreq} Hz",
                category="metadata",
                rationale=seems_correct_because("epoching should preserve the sampling rate of the continuous input unless an explicit resampling step is documented"),
            )

            # Check dimensions
            n_channels = len(epochs.ch_names)
            n_samples = epochs.get_data().shape[2] if len(epochs) > 0 else 0
            collector.add_result(
                subject_id,
                person,
                "âœ“",
                f"Dimensions: ({len(epochs)} epochs, {n_channels} channels, {n_samples} samples)",
                category="structure",
                rationale=seems_correct_because("explicit array dimensions make it easier to verify that later decoding sees the expected trial-by-channel-by-time structure"),
            )

            # Check for baseline correction
            if epochs.baseline is not None:
                collector.add_result(
                    subject_id,
                    person,
                    "âœ“",
                    f"Baseline period: {epochs.baseline}",
                    category="baseline",
                    rationale=seems_correct_because("documenting the baseline window clarifies which part of each trial anchors amplitude comparisons"),
                )
            else:
                collector.add_result(
                    subject_id,
                    person,
                    "âš ",
                    "No baseline correction applied",
                    category="baseline",
                    rationale=strange_because("without a baseline, between-trial amplitude shifts can become harder to interpret"),
                )

            # Check bad channels
            bads = epochs.info.get("bads", [])
            if len(bads) == 0:
                collector.add_result(
                    subject_id,
                    person,
                    "âœ“",
                    "No bad channels marked",
                    category="signal_quality",
                    rationale=seems_correct_because("by the epoching stage, repaired channels should usually already be back in the usable sensor set"),
                )
            else:
                bad_pct = len(bads) / n_channels * 100 if n_channels > 0 else 0
                collector.add_result(
                    subject_id,
                    person,
                    "âš ",
                    f"Bad channels marked: {len(bads)}/{n_channels} ({bad_pct:.1f}%)",
                    category="signal_quality",
                    rationale=strange_because("persistent bad channels at this late stage may mean they were intentionally excluded or not fully repaired"),
                )

            # Check for data integrity
            if len(epochs) > 0:
                data = epochs.get_data()
                nan_count = int(np.isnan(data).sum())
                inf_count = int(np.isinf(data).sum())
                
                if nan_count == 0 and inf_count == 0:
                    collector.add_result(
                        subject_id,
                        person,
                        "âœ“",
                        "No NaN/Inf values",
                        category="data_integrity",
                        rationale=seems_correct_because("numerically valid epoch arrays are a hard prerequisite for all later statistics and decoding models"),
                    )
                else:
                    collector.add_result(
                        subject_id,
                        person,
                        "ERROR",
                        f"Data integrity issue: {nan_count} NaN and {inf_count} Inf values",
                        category="data_integrity",
                        rationale=strange_because("missing or infinite values will typically break downstream averaging, decoding, or visualization code"),
                    )

                # Check for extreme values
                data_abs = np.abs(data)
                max_val = np.nanmax(data_abs)
                if max_val > 1e4:  # > 10 mV in V units
                    collector.add_result(
                        subject_id,
                        person,
                        "âš ",
                        f"Large amplitude detected: {max_val*1e6:.0f} ÂµV (possible artifact)",
                        category="signal_quality",
                        rationale=strange_because("very large voltages after preprocessing often indicate remaining artifacts or a scaling problem"),
                    )

    collector.print_summary()

    # Export summary CSV
    output_csv = config.QC_DIR / "sc_07_epoch_summary.csv"
    collector.export_csv(output_csv)
    print(f"\nâœ“ Summary exported to {output_csv.name}\n")


def run_visualizations(subjects):
    from plots.sc_07_epoch_plots import main as viz_main

    argv = []
    if subjects:
        argv.extend(["--subjects", ",".join(subjects)])
    viz_main(argv)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Sanity check and visualization for step 07 (epoching).")
    add_subjects_argument(parser)
    add_mode_argument(parser, default="check")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    check_subjects = resolve_subjects(args.subjects, config.SUBJECTS, mode="check")
    viz_subjects = resolve_subjects(args.subjects, config.SUBJECTS, mode="viz")

    if args.mode in {"check", "both"}:
        sanity_check_epoch(check_subjects)
    if args.mode in {"viz", "both"}:
        run_visualizations(viz_subjects)


if __name__ == "__main__":
    main()


