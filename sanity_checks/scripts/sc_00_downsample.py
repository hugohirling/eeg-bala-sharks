"""
Sanity Check for Step 00: Downsample

Checks:
- Downsampling was applied successfully
- Sampling rate was reduced correctly
- Data duration and size are plausible
- No obvious artifacts were introduced by downsampling

REASONING:
- Purpose: verify that the memory-saving resampling step preserves the analyzable EEG time course.
- Reproducibility: the target sampling frequency is read from config and the same input should always produce the same metadata outcome.
- Parameter notes: a duration tolerance of 0.1 s is used because minor rounding can occur when sample counts are converted back to seconds.
"""
import sys
import argparse
from pathlib import Path

import mne

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

PIPELINE_DIR = CURRENT_DIR.parent.parent / "eeg_pipeline"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from preprocessing import config
from helpers.sc_cli import add_duration_argument, add_mode_argument, add_subjects_argument, resolve_subjects
from helpers.sc_utils import SanityCheckCollector, seems_correct_because, strange_because


DURATION_TOLERANCE_S = 0.1


def sanity_check_downsample(subjects):
    collector = SanityCheckCollector("00 - Downsample")
    collector.set_step_context(
        purpose="Downsampling reduces file size and runtime early in the pipeline, but it must not alter channel layout or noticeably distort trial timing.",
        reproducibility="The expected target frequency comes from preprocessing/config.py, so the same dataset and config should always produce the same sampling-rate check.",
        parameter_notes=[
            f"Expected sampling rate is config.DOWNSAMPLE_SFREQ = {config.DOWNSAMPLE_SFREQ} Hz.",
            f"Duration tolerance is {DURATION_TOLERANCE_S:.1f} s to allow for rounding when comparing sample-based recordings across sampling rates.",
        ],
    )

    from mne_bids import BIDSPath, read_raw_bids

    for subject_id in subjects:
        print(f"\n--- Checking subject {subject_id} ---\n")

        # Load original BIDS data
        bids_path = BIDSPath(
            subject=subject_id,
            task="RPS",
            datatype="eeg",
            suffix="eeg",
            root=config.BIDS_ROOT,
        )
        raw_original = read_raw_bids(bids_path, verbose=False)
        original_sfreq = raw_original.info["sfreq"]

        # Load downsampled output
        out_path = config.OUTPUT_DIR / f"sub-{subject_id}_downsampled.fif"
        if not out_path.exists():
            collector.add_result(
                subject_id,
                "P1",
                "ERROR",
                f"Downsampled file not found: {out_path.name}",
                category="file_io",
                rationale=strange_because("the resampling step should always create one shared subject-level FIF file before player splitting"),
            )
            continue

        raw_downsampled = mne.io.read_raw_fif(str(out_path), preload=False)
        downsampled_sfreq = raw_downsampled.info["sfreq"]

        collector.add_result(
            subject_id,
            "P1",
                "OK",
            f"Sampling rate: {original_sfreq:.1f} Hz -> {downsampled_sfreq:.1f} Hz ({original_sfreq / downsampled_sfreq:.1f}x)",
            category="metadata",
            rationale=seems_correct_because("the downsampled file should expose a lower sfreq while retaining the same recording content"),
            parameter_note=f"Target frequency comes from config.DOWNSAMPLE_SFREQ = {config.DOWNSAMPLE_SFREQ} Hz.",
        )

        expected_sfreq = config.DOWNSAMPLE_SFREQ
        if downsampled_sfreq != expected_sfreq:
            collector.add_result(
                subject_id,
                "P1",
                "WARN",
                f"Expected {expected_sfreq} Hz, got {downsampled_sfreq} Hz",
                category="metadata",
                rationale=strange_because("a mismatching sampling rate changes the effective Nyquist limit and weakens reproducibility across machines"),
            )

        # Channel count should remain the same
        if len(raw_original.ch_names) == len(raw_downsampled.ch_names):
            collector.add_result(
                subject_id,
                "P1",
                "OK",
                f"Channel count preserved: {len(raw_downsampled.ch_names)}",
                category="structure",
                rationale=seems_correct_because("resampling should change temporal resolution, not the recorded channel set"),
            )
        else:
            collector.add_result(
                subject_id,
                "P1",
                "ERROR",
                f"Channel count mismatch: {len(raw_original.ch_names)} -> {len(raw_downsampled.ch_names)}",
                category="structure",
                rationale=strange_because("downsampling should not add or remove channels; that would indicate corruption or a wrong intermediate file"),
            )

        # Duration should be approximately the same
        original_duration = raw_original.times[-1]
        downsampled_duration = raw_downsampled.times[-1]
        duration_diff = abs(original_duration - downsampled_duration)
        if duration_diff < DURATION_TOLERANCE_S:
            collector.add_result(
                subject_id,
                "P1",
                "OK",
                f"Duration preserved within tolerance: {original_duration:.2f}s vs {downsampled_duration:.2f}s (delta {duration_diff:.4f}s)",
                category="temporal_integrity",
                rationale=seems_correct_because("the recording should span the same time interval after resampling, aside from rounding at sample boundaries"),
                parameter_note=f"Acceptable delta < {DURATION_TOLERANCE_S:.1f} s.",
            )
        else:
            collector.add_result(
                subject_id,
                "P1",
                "WARN",
                f"Duration changed more than expected: {original_duration:.2f}s vs {downsampled_duration:.2f}s (delta {duration_diff:.4f}s)",
                category="temporal_integrity",
                rationale=strange_because("a larger duration shift suggests dropped or duplicated samples during resampling"),
                parameter_note=f"Acceptable delta < {DURATION_TOLERANCE_S:.1f} s.",
            )

        # Estimate file size reduction
        original_est_size = (len(raw_original.ch_names) * raw_original.n_times * 8) / 1e6
        downsampled_est_size = (len(raw_downsampled.ch_names) * raw_downsampled.n_times * 8) / 1e6
        size_ratio = downsampled_est_size / original_est_size * 100

        collector.add_result(
            subject_id,
            "P1",
            "OK",
            f"Estimated size: {original_est_size:.2f} MB -> {downsampled_est_size:.2f} MB ({size_ratio:.1f}% of original)",
            category="efficiency",
            rationale=seems_correct_because("the file size should shrink roughly in proportion to the lower sample count, which is the main motivation for downsampling early"),
        )

    collector.print_summary()
    output_csv = config.QC_DIR / "sc_00_downsample_summary.csv"
    collector.export_csv(output_csv)
    print(f"\nOK Summary exported to {output_csv.name}\n")


def run_visualizations(subjects, duration):
    from plots.sc_00_downsample_plots import main as viz_main

    argv = []
    if subjects:
        argv.extend(["--subjects", ",".join(subjects)])
    argv.extend(["--duration", str(duration)])
    viz_main(argv)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Sanity check and visualization for step 00 (downsample).")
    add_subjects_argument(parser)
    add_mode_argument(parser, default="check")
    add_duration_argument(parser, default=30)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    check_subjects = resolve_subjects(args.subjects, config.SUBJECTS, mode="check")
    viz_subjects = resolve_subjects(args.subjects, config.SUBJECTS, mode="viz")

    if args.mode in {"check", "both"}:
        sanity_check_downsample(check_subjects)
    if args.mode in {"viz", "both"}:
        run_visualizations(viz_subjects, args.duration)


if __name__ == "__main__":
    main()


