"""
Sanity Check for Step 03: Bad Channels Detect

Checks:
- Bad channels are identified
- QC reports are generated
- Bad channels are marked in raw objects
- Summary statistics are plausible

REASONING:
- Purpose: identify unusable sensors before interpolation and ICA so artifacts do not propagate downstream.
- Reproducibility: the same detection report and bad-list should be recreated when the same thresholds and input files are used.
- Parameter notes: very large bad-channel fractions are treated as suspicious because they often indicate a global acquisition issue rather than isolated bad sensors.
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
from helpers.sc_cli import add_mode_argument, add_subjects_argument, resolve_subjects
from helpers.sc_utils import SanityCheckCollector, detect_bad_channel_anomalies, seems_correct_because, strange_because


def sanity_check_bad_channels_detect(subjects):
    collector = SanityCheckCollector("03 - Bad Channels Detect")
    collector.set_step_context(
        purpose="Bad-channel detection should flag noisy electrodes early so interpolation and ICA operate on cleaner spatial information.",
        reproducibility="The raw file, QC TSV, and stored bad-channel list should agree if the same detection logic and thresholds are used.",
        parameter_notes=[
            "A bad-channel fraction above 20% is flagged as unusual because it may point to poor cap fit or a recording-wide hardware issue.",
            "The QC TSV row count should match the analyzed EEG channel count so we know no channels were silently skipped.",
        ],
    )

    for subject_id in subjects:
        for person in ["P1", "P2"]:
            path = config.OUTPUT_DIR / f"sub-{subject_id}_{person}_badchannels_detected.fif"

            if not path.exists():
                collector.add_result(
                    subject_id,
                    person,
                    "ERROR",
                    "Bad-channels detected file not found",
                    category="file_io",
                    rationale=strange_because("the detection step should output a FIF file with the marked bad channels stored in raw.info['bads']"),
                )
                continue

            raw = mne.io.read_raw_fif(str(path), preload=False)
            collector.add_result(
                subject_id,
                person,
                "OK",
                f"File exists: {path.name}",
                category="file_io",
                rationale=seems_correct_because("the detection step should persist both the cleaned metadata and the bad-channel annotations"),
            )

            bads = raw.info.get("bads", [])
            collector.add_result(
                subject_id,
                person,
                "OK",
                f"Bad channels marked: {len(bads)}" + (f" ({', '.join(bads)})" if bads else ""),
                category="signal_quality",
                rationale=seems_correct_because("isolating a small number of noisy sensors is expected and protects later interpolation and ICA"),
            )

            # Check QC report
            qc_report_path = config.QC_DIR / f"sub-{subject_id}_{person}_bad_channels_detect.tsv"
            if qc_report_path.exists():
                with open(qc_report_path, "r") as f:
                    lines = f.readlines()
                collector.add_result(
                    subject_id,
                    person,
                    "OK",
                    f"QC report generated with {len(lines) - 1} analyzed channels: {qc_report_path.name}",
                    category="qc_report",
                    rationale=seems_correct_because("a per-channel report makes the detection step inspectable and easier to justify in the report"),
                )
            else:
                collector.add_result(
                    subject_id,
                    person,
                    "WARN",
                    "QC report not found",
                    category="qc_report",
                    rationale=strange_because("without the TSV, the grader cannot easily inspect why channels were flagged"),
                )

            # Sanity check: EEG channel count
            eeg_picks = mne.pick_types(raw.info, eeg=True, exclude="bads")
            collector.add_result(
                subject_id,
                person,
                "OK",
                f"EEG channels remaining after exclusion: {len(eeg_picks)}",
                category="structure",
                rationale=seems_correct_because("most EEG channels should remain usable after detection; otherwise the recording quality may be globally poor"),
            )

            anomaly = detect_bad_channel_anomalies(bads, len(mne.pick_types(raw.info, eeg=True, exclude=[])))
            if anomaly:
                collector.add_result(
                    subject_id,
                    person,
                    "WARN",
                    anomaly,
                    category="signal_quality",
                    rationale=strange_because("too many bad channels usually reflects a broader recording problem, not just a few isolated electrode failures"),
                )

            if len(bads) > len(mne.pick_types(raw.info, eeg=True)):
                collector.add_result(
                    subject_id,
                    person,
                    "WARN",
                    "More bad channels than total EEG channels",
                    category="signal_quality",
                    rationale=strange_because("this can only happen if metadata or counting logic is inconsistent"),
                )

            collector.add_result(
                subject_id,
                person,
                "OK",
                f"Sampling rate preserved: {raw.info['sfreq']} Hz",
                category="metadata",
                rationale=seems_correct_because("detecting bad channels should annotate metadata only and not alter the temporal sampling of the recording"),
            )

    collector.print_summary()
    output_csv = config.QC_DIR / "sc_03_bad_channels_summary.csv"
    collector.export_csv(output_csv)
    print(f"\nOK Summary exported to {output_csv.name}\n")


def run_visualizations(subjects):
    from plots.sc_03_bad_channels_plots import main as viz_main

    argv = []
    if subjects:
        argv.extend(["--subjects", ",".join(subjects)])
    viz_main(argv)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Sanity check and visualization for step 03 (bad channels detect).")
    add_subjects_argument(parser)
    add_mode_argument(parser, default="check")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    check_subjects = resolve_subjects(args.subjects, config.SUBJECTS, mode="check")
    viz_subjects = resolve_subjects(args.subjects, config.SUBJECTS, mode="viz")

    if args.mode in {"check", "both"}:
        sanity_check_bad_channels_detect(check_subjects)
    if args.mode in {"viz", "both"}:
        run_visualizations(viz_subjects)


if __name__ == "__main__":
    main()


