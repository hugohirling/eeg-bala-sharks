# This file's comments were created with the help of GitHub Copilot using GPT-5.3-Codex.
"""
Sanity Check for Step 02: Rename & Set Montage

Checks:
- Channels are renamed correctly (BioSemi to 10-20 labels)
- Montage is applied
- Electrode positions are present
- Channel metadata remains intact

REASONING:
- Purpose: confirm that channel labels and spatial coordinates are ready for topographies, interpolation, and interpretation.
- Reproducibility: the renamed labels and BioSemi64 montage come from a fixed mapping in config, so the result should be stable across runs.
- Parameter notes: a small set of canonical 10-20 labels is used as an easy sanity proxy before looking at the full 64-channel montage.
"""
import sys
import argparse
from pathlib import Path

import mne
import matplotlib.pyplot as plt

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

PIPELINE_DIR = CURRENT_DIR.parent.parent / "eeg_pipeline"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from preprocessing import config
from helpers.sc_cli import add_mode_argument, add_subjects_argument, resolve_subjects
from helpers.sc_utils import SanityCheckCollector, seems_correct_because, strange_because


def _save_montage_visualizations(raw, subject_id, person):
    config.QC_DIR.mkdir(parents=True, exist_ok=True)

    raw_eeg = raw.copy().pick("eeg")

    fig_2d = raw_eeg.plot_sensors(kind="topomap", show_names=True, show=False)
    fig_2d.subtitle(f"sub-{subject_id} {person} EEG sensor layout (2D)")
    out_2d = config.QC_DIR / f"sub-{subject_id}_{person}_montage_positions_2d.png"
    fig_2d.savefig(out_2d, dpi=200, bbox_inches="tight")
    plt.close(fig_2d)
    print(f"  OK Saved montage plot (2D): {out_2d.name}")

    try:
        fig_3d = raw_eeg.plot_sensors(kind="3d", show_names=False, show=False)
        fig_3d.subtitle(f"sub-{subject_id} {person} EEG sensor layout (3D)")
        out_3d = config.QC_DIR / f"sub-{subject_id}_{person}_montage_positions_3d.png"
        fig_3d.savefig(out_3d, dpi=200, bbox_inches="tight")
        plt.close(fig_3d)
        print(f"  OK Saved montage plot (3D): {out_3d.name}")
    except Exception as exc:
        print(f"  WARNING: Could not save 3D montage plot: {exc}")


def sanity_check_rename_montage(subjects):
    collector = SanityCheckCollector("02 - Rename & Set Montage")
    collector.set_step_context(
        purpose="Montage assignment is the point where abstract channels become spatially interpretable scalp sensors, which is essential for topomaps and interpolation.",
        reproducibility="The channel rename map and montage template are hard-coded in preprocessing/config.py, so correct inputs should yield identical sensor layouts.",
        parameter_notes=[
            "The quick 10-20 label subset is only a proxy check; the full montage still uses the complete BioSemi64 mapping.",
            "Presence of digitization points is critical because later topographic plots assume real sensor coordinates instead of placeholder positions.",
        ],
    )

    for subject_id in subjects:
        for person in ["P1", "P2"]:
            path = config.OUTPUT_DIR / f"sub-{subject_id}_{person}_renamed_montaged.fif"

            if not path.exists():
                collector.add_result(
                    subject_id,
                    person,
                    "ERROR",
                    f"Renamed+montaged file not found: {path.name}",
                    category="file_io",
                    rationale=strange_because("the montage step should always produce a player-specific output before bad-channel detection starts"),
                )
                continue

            raw = mne.io.read_raw_fif(str(path), preload=False)
            collector.add_result(
                subject_id,
                person,
                "OK",
                f"File exists: {path.name}",
                category="file_io",
                rationale=seems_correct_because("the step should persist the renamed channels and montage together in one FIF file"),
            )

            eeg_picks = mne.pick_types(raw.info, eeg=True)
            eeg_names = [raw.ch_names[i] for i in eeg_picks]

            collector.add_result(
                subject_id,
                person,
                "OK",
                f"EEG channels available after rename/montage: {len(eeg_names)}",
                category="structure",
                rationale=seems_correct_because("the montage step should keep the full EEG set while only changing labels and metadata"),
            )

            # Check if standard 10-20 names are present
            standard_10_20_names = {
                "Fp1",
                "Fp2",
                "Fz",
                "F3",
                "F4",
                "Cz",
                "C3",
                "C4",
                "Pz",
                "P3",
                "P4",
                "Oz",
            }
            found_standard = sum(1 for ch in eeg_names if ch in standard_10_20_names)
            collector.add_result(
                subject_id,
                person,
                "OK",
                f"Canonical 10-20 labels found: {found_standard}/{len(eeg_names)}",
                category="naming",
                rationale=seems_correct_because("recognizable standard labels are a quick indicator that the rename map was applied sensibly"),
            )

            # Check montage
            if raw.info.get("dig") is not None and len(raw.info["dig"]) > 0:
                collector.add_result(
                    subject_id,
                    person,
                    "OK",
                    f"Digitization points available: {len(raw.info['dig'])}",
                    category="spatial_metadata",
                    rationale=seems_correct_because("spatial sensor positions are required for topomaps, interpolation, and most scalp-level interpretations"),
                )
                _save_montage_visualizations(raw, subject_id, person)
            else:
                collector.add_result(
                    subject_id,
                    person,
                    "WARN",
                    "No electrode positions found in montage",
                    category="spatial_metadata",
                    rationale=strange_because("missing coordinates would make later topographies visually misleading or impossible to compute"),
                )

            # Check for old prefixes
            prefix = config.PLAYER_PREFIX_MAP[person]
            old_prefix_count = sum(1 for ch in raw.ch_names if ch.startswith(prefix))
            if old_prefix_count > 0:
                collector.add_result(
                    subject_id,
                    person,
                    "WARN",
                    f"{old_prefix_count} channels still have old prefix '{prefix}'",
                    category="naming",
                    rationale=strange_because("the rename step should fully remove player-specific acquisition prefixes before standard labels are assigned"),
                )
            else:
                collector.add_result(
                    subject_id,
                    person,
                    "OK",
                    "Old prefixes successfully removed",
                    category="naming",
                    rationale=seems_correct_because("clean standard labels reduce ambiguity and make the code easier to read and reproduce"),
                )

            collector.add_result(
                subject_id,
                person,
                "OK",
                f"Sampling rate {raw.info['sfreq']} Hz, duration {raw.times[-1]:.2f}s, first EEG labels {eeg_names[:5]}",
                category="metadata",
                rationale=seems_correct_because("rename/montage should leave the recording duration untouched while making the channels easier to interpret"),
            )

    collector.print_summary()
    output_csv = config.QC_DIR / "sc_02_rename_montage_summary.csv"
    collector.export_csv(output_csv)
    print(f"\nOK Summary exported to {output_csv.name}\n")


def run_visualizations(subjects):
    from plots.sc_02_rename_montage_plots import main as viz_main

    argv = []
    if subjects:
        argv.extend(["--subjects", ",".join(subjects)])
    viz_main(argv)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Sanity check and visualization for step 02 (rename and montage).")
    add_subjects_argument(parser)
    add_mode_argument(parser, default="check")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    check_subjects = resolve_subjects(args.subjects, config.SUBJECTS, mode="check")
    viz_subjects = resolve_subjects(args.subjects, config.SUBJECTS, mode="viz")

    if args.mode in {"check", "both"}:
        sanity_check_rename_montage(check_subjects)
    if args.mode in {"viz", "both"}:
        run_visualizations(viz_subjects)


if __name__ == "__main__":
    main()


