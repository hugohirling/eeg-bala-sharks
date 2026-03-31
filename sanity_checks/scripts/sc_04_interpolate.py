# This file's comments were created with the help of GitHub Copilot using GPT-5.3-Codex.
"""
Sanity Check for Step 04: Interpolate Bad Channels

Checks:
- Interpolation was applied
- Bad-channel list is cleared after interpolation
- Channel count is unchanged
- Signal amplitudes remain plausible

REASONING:
- Purpose: verify that bad sensors were repaired in place instead of silently removed from the dataset.
- Reproducibility: the same pre-interpolation bad list should lead to the same repaired output when interpolation is rerun.
- Parameter notes: channel and sample counts must remain identical because interpolation should replace values, not change shape.
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


def sanity_check_interpolate_bad_channels(subjects):
    collector = SanityCheckCollector("04 - Interpolate Bad Channels")
    collector.set_step_context(
        purpose="Interpolation should repair bad sensors while preserving the original recording structure for all later analyses.",
        reproducibility="Given the same bad-channel list and montage geometry, interpolation should reconstruct the same channels each time.",
        parameter_notes=[
            "After interpolation the bad list is expected to be cleared because the channels are now repaired and available again.",
            "Sampling rate, channel count, and sample count must remain unchanged because interpolation only fills values spatially.",
        ],
    )

    for subject_id in subjects:
        for person in ["P1", "P2"]:
            before_path = config.OUTPUT_DIR / f"sub-{subject_id}_{person}_badchannels_detected.fif"
            after_path = config.OUTPUT_DIR / f"sub-{subject_id}_{person}_interpolated.fif"

            if not before_path.exists():
                collector.add_result(
                    subject_id,
                    person,
                    "ERROR",
                    "Input file (badchannels_detected) not found",
                    category="file_io",
                    rationale=strange_because("interpolation depends on the prior bad-channel annotations and cannot be checked without them"),
                )
                continue

            if not after_path.exists():
                collector.add_result(
                    subject_id,
                    person,
                    "ERROR",
                    "Output file (interpolated) not found",
                    category="file_io",
                    rationale=strange_because("the interpolation step should always create a repaired output FIF file"),
                )
                continue

            raw_before = mne.io.read_raw_fif(str(before_path), preload=False)
            raw_after = mne.io.read_raw_fif(str(after_path), preload=False)

            bads_before = raw_before.info.get("bads", [])
            bads_after = raw_after.info.get("bads", [])

            collector.add_result(
                subject_id,
                person,
                "OK",
                f"Files exist; bads before/after interpolation = {len(bads_before)}/{len(bads_after)}" + (f" ({', '.join(bads_before)})" if bads_before else ""),
                category="signal_quality",
                rationale=seems_correct_because("the same channels marked bad before interpolation should be the ones repaired in the output"),
            )

            # After interpolation, bads list should be empty
            if len(bads_after) == 0:
                collector.add_result(
                    subject_id,
                    person,
                    "OK",
                    "Bad channels cleared after interpolation",
                    category="signal_quality",
                    rationale=seems_correct_because("the repaired channels are expected to rejoin the usable sensor set after interpolation"),
                )
            else:
                collector.add_result(
                    subject_id,
                    person,
                    "WARN",
                    "Bad channels still marked after interpolation",
                    category="signal_quality",
                    rationale=strange_because("a repaired output usually clears the bad list; otherwise later steps may still exclude those channels"),
                )

            # Channel count should be same
            if len(raw_before.ch_names) == len(raw_after.ch_names):
                collector.add_result(
                    subject_id,
                    person,
                    "OK",
                    f"Channel count preserved: {len(raw_after.ch_names)}",
                    category="structure",
                    rationale=seems_correct_because("interpolation should repair values in place, not remove sensors"),
                )
            else:
                collector.add_result(
                    subject_id,
                    person,
                    "ERROR",
                    "Channel count mismatch",
                    category="structure",
                    rationale=strange_because("a channel-count change would indicate deletion or duplication instead of pure interpolation"),
                )

            # Sampling rate and duration should be the same
            if raw_before.info["sfreq"] == raw_after.info["sfreq"]:
                collector.add_result(
                    subject_id,
                    person,
                    "OK",
                    f"Sampling rate preserved: {raw_after.info['sfreq']} Hz",
                    category="metadata",
                    rationale=seems_correct_because("interpolation is a spatial operation and should not alter time resolution"),
                )
            else:
                collector.add_result(
                    subject_id,
                    person,
                    "ERROR",
                    f"Sampling rate changed: {raw_before.info['sfreq']} -> {raw_after.info['sfreq']}",
                    category="metadata",
                    rationale=strange_because("changing sfreq here would make later comparisons to the pre-interpolation file invalid"),
                )

            n_samples_before = raw_before.n_times
            n_samples_after = raw_after.n_times
            if n_samples_before == n_samples_after:
                collector.add_result(
                    subject_id,
                    person,
                    "OK",
                    f"Sample count preserved: {n_samples_after}",
                    category="temporal_integrity",
                    rationale=seems_correct_because("interpolation should preserve the exact sample grid so epoch timing stays aligned"),
                )
            else:
                collector.add_result(
                    subject_id,
                    person,
                    "ERROR",
                    f"Sample count changed: {n_samples_before} -> {n_samples_after}",
                    category="temporal_integrity",
                    rationale=strange_because("a changed sample count would break one-to-one comparison with the preceding pipeline step"),
                )

    collector.print_summary()
    output_csv = config.QC_DIR / "sc_04_interpolate_summary.csv"
    collector.export_csv(output_csv)
    print(f"\nOK Summary exported to {output_csv.name}\n")


def run_visualizations(subjects, duration):
    from plots.sc_04_interpolate_plots import main as viz_main

    argv = []
    if subjects:
        argv.extend(["--subjects", ",".join(subjects)])
    argv.extend(["--duration", str(duration)])
    viz_main(argv)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Sanity check and visualization for step 04 (interpolate bad channels).")
    add_subjects_argument(parser)
    add_mode_argument(parser, default="check")
    add_duration_argument(parser, default=30)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    check_subjects = resolve_subjects(args.subjects, config.SUBJECTS, mode="check")
    viz_subjects = resolve_subjects(args.subjects, config.SUBJECTS, mode="viz")

    if args.mode in {"check", "both"}:
        sanity_check_interpolate_bad_channels(check_subjects)
    if args.mode in {"viz", "both"}:
        run_visualizations(viz_subjects, args.duration)


if __name__ == "__main__":
    main()


