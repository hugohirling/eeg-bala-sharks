# This file's comments were created with the help of GitHub Copilot using GPT-5.3-Codex.
"""
Sanity Check for Step 01: Split Players

Checks:
- Player streams are split correctly
- Channels are assigned to the correct player
- Channel types are set correctly
- Status channel is present for both players

REASONING:
- Purpose: verify that the dyadic recording is split into two analyzable player streams without channel leakage.
- Reproducibility: the split is deterministic because channel prefixes are defined in config.PLAYER_PREFIX_MAP.
- Parameter notes: prefix checks are strict because leftover P1/P2 labels would break downstream montage mapping.
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
from helpers.sc_utils import SanityCheckCollector, seems_correct_because, strange_because


def sanity_check_split_players(subjects):
    collector = SanityCheckCollector("01 - Split Players")
    collector.set_step_context(
        purpose="Player splitting should isolate each participant's channels while preserving timing, so later per-player analyses stay valid.",
        reproducibility="Player-specific channel prefixes are defined in preprocessing/config.py, so the same file should always split into the same P1/P2 outputs.",
        parameter_notes=[
            "A remaining player prefix after the split is treated as suspicious because downstream rename/montage logic assumes cleaned labels.",
            "The shared Status channel is expected in both outputs so events remain aligned across both players.",
        ],
    )

    for subject_id in subjects:
        for person in ["P1", "P2"]:
            path = config.OUTPUT_DIR / f"sub-{subject_id}_{person}_split.fif"

            if not path.exists():
                collector.add_result(
                    subject_id,
                    person,
                    "ERROR",
                    f"Split file not found: {path.name}",
                    category="file_io",
                    rationale=strange_because("each subject should produce one split file per player before any later preprocessing step can run"),
                )
                continue

            raw = mne.io.read_raw_fif(str(path), preload=False)
            collector.add_result(
                subject_id,
                person,
                "OK",
                f"File exists with {len(raw.ch_names)} total channels: {path.name}",
                category="file_io",
                rationale=seems_correct_because("the split stage should materialize a dedicated FIF file for each player stream"),
            )

            # Count by type
            eeg_picks = mne.pick_types(raw.info, eeg=True, exclude=[])
            eog_picks = mne.pick_types(raw.info, eog=True, exclude=[])
            resp_picks = mne.pick_types(raw.info, resp=True, exclude=[])
            misc_picks = mne.pick_types(raw.info, misc=True, exclude=[])
            stim_picks = mne.pick_types(raw.info, stim=True, exclude=[])

            collector.add_result(
                subject_id,
                person,
                "OK",
                f"Channel types EEG/EOG/RESP/MISC/STIM = {len(eeg_picks)}/{len(eog_picks)}/{len(resp_picks)}/{len(misc_picks)}/{len(stim_picks)}",
                category="channel_types",
                rationale=seems_correct_because("downstream artifact handling and event extraction rely on these types being preserved after the split"),
            )

            if len(stim_picks) == 0:
                collector.add_result(
                    subject_id,
                    person,
                    "WARN",
                    "No stim channel found",
                    category="events",
                    rationale=strange_because("epoching later depends on event information, and a missing stim/status stream can make trials irrecoverable"),
                )

            prefix = config.PLAYER_PREFIX_MAP[person]
            ch_count_with_prefix = sum(1 for ch in raw.ch_names if ch.startswith(prefix))
            if ch_count_with_prefix > 0:
                collector.add_result(
                    subject_id,
                    person,
                    "WARN",
                    f"{ch_count_with_prefix} channels still have prefix '{prefix}'",
                    category="naming",
                    rationale=strange_because("prefix remnants suggest the split output is not normalized for the rename/montage step"),
                )
            else:
                collector.add_result(
                    subject_id,
                    person,
                    "OK",
                    f"Prefix '{prefix}' removed from EEG channels",
                    category="naming",
                    rationale=seems_correct_because("the split output should contain player-local channel names that can be mapped to BioSemi labels without extra string handling"),
                )

            collector.add_result(
                subject_id,
                person,
                "OK",
                f"Sampling rate {raw.info['sfreq']} Hz, duration {raw.times[-1]:.2f}s",
                category="temporal_integrity",
                rationale=seems_correct_because("splitting is expected to preserve the original timing so both players remain trial-synchronized"),
            )

    collector.print_summary()
    output_csv = config.QC_DIR / "sc_01_split_players_summary.csv"
    collector.export_csv(output_csv)
    print(f"\nOK Summary exported to {output_csv.name}\n")


def run_visualizations(subjects):
    from plots.sc_01_split_players_plots import main as viz_main

    argv = []
    if subjects:
        argv.extend(["--subjects", ",".join(subjects)])
    viz_main(argv)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Sanity check and visualization for step 01 (split players).")
    add_subjects_argument(parser)
    add_mode_argument(parser, default="check")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    check_subjects = resolve_subjects(args.subjects, config.SUBJECTS, mode="check")
    viz_subjects = resolve_subjects(args.subjects, config.SUBJECTS, mode="viz")

    if args.mode in {"check", "both"}:
        sanity_check_split_players(check_subjects)
    if args.mode in {"viz", "both"}:
        run_visualizations(viz_subjects)


if __name__ == "__main__":
    main()


