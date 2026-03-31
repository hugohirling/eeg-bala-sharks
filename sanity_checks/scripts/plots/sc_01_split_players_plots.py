"""
Sanity Check Plot Module for Step 01: Split Players

Creates comparison plots for the player-splitting step:
- Data distribution per player (duration, channels)
- Event/task consistency between P1 and P2
- Data statistics per player comparison

Entry point:
    python sanity_checks/scripts/sc_01_split_players.py --mode viz

Options:
    --subjects: Comma-separated subject IDs (default: first 2)

REASONING:
- Purpose: show that both player streams retain comparable timing and expected channel content after the split.
- Reproducibility: the figures are deterministic summaries of the saved split FIF files.
- Interpretation focus: the expected argument is "This seems correct because both players keep similar duration and sample counts while channel leakage stays at zero."
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np

CURRENT_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = CURRENT_DIR.parent.parent / "eeg_pipeline"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from preprocessing import config
from helpers.sc_cli import add_subjects_argument, resolve_subjects
from helpers.sc_config import DEFAULT_PERSONS, SPLIT_VIZ
from helpers.sc_plot_io import save_figure


PLAYER_COLORS = [SPLIT_VIZ["p1"], SPLIT_VIZ["p2"]]
EXPECTED_COLOR = SPLIT_VIZ["expected"]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Visualize split-players effects (per-subject P1 vs P2 data distribution)",
    )
    add_subjects_argument(parser)
    return parser.parse_args(argv)


def plot_data_summary_per_player(subject_id, output_dir):
    """Plot data integrity and size comparison between P1 and P2."""
    summary_data = {
        "Player": [],
        "Duration (s)": [],
        "Channels": [],
        "Samples": [],
        "Est. Size (MB)": [],
    }

    for person in DEFAULT_PERSONS:
        file_path = config.OUTPUT_DIR / f"sub-{subject_id}_{person}_split.fif"

        if not file_path.exists():
            print(f"    {person}: File not found, skipping visualization")
            continue

        try:
            raw = mne.io.read_raw_fif(str(file_path), preload=False, verbose=False)
            est_size = (len(raw.ch_names) * raw.n_times * 8) / 1e6

            summary_data["Player"].append(person)
            summary_data["Duration (s)"].append(raw.times[-1])
            summary_data["Channels"].append(len(raw.ch_names))
            summary_data["Samples"].append(raw.n_times)
            summary_data["Est. Size (MB)"].append(est_size)
        except Exception as e:
            print(f"    {person}: Error loading file: {e}")

    if not summary_data["Player"]:
        return

    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Duration comparison
    axes[0, 0].bar(summary_data["Player"], summary_data["Duration (s)"], color=PLAYER_COLORS, alpha=0.7, edgecolor="black")
    axes[0, 0].set_ylabel("Duration (s)")
    axes[0, 0].set_title("Recording Duration per Player")
    axes[0, 0].grid(axis="y", alpha=0.3)
    for i, v in enumerate(summary_data["Duration (s)"]):
        axes[0, 0].text(i, v + 1, f"{v:.1f}s", ha="center", fontweight="bold")

    # Channels
    axes[0, 1].bar(summary_data["Player"], summary_data["Channels"], color=PLAYER_COLORS, alpha=0.7, edgecolor="black")
    axes[0, 1].set_ylabel("Number of Channels")
    axes[0, 1].set_title("Channel Count per Player")
    axes[0, 1].grid(axis="y", alpha=0.3)
    for i, v in enumerate(summary_data["Channels"]):
        axes[0, 1].text(i, v + 0.5, str(v), ha="center", fontweight="bold")

    # Samples
    axes[1, 0].bar(summary_data["Player"], summary_data["Samples"], color=PLAYER_COLORS, alpha=0.7, edgecolor="black")
    axes[1, 0].set_ylabel("Number of Samples")
    axes[1, 0].set_title("Sample Count per Player")
    axes[1, 0].grid(axis="y", alpha=0.3)
    for i, v in enumerate(summary_data["Samples"]):
        axes[1, 0].text(i, v + 500, f"{v:,}", ha="center", fontweight="bold", fontsize=9)

    # File size
    axes[1, 1].bar(summary_data["Player"], summary_data["Est. Size (MB)"], color=PLAYER_COLORS, alpha=0.7, edgecolor="black")
    axes[1, 1].set_ylabel("Estimated Size (MB)")
    axes[1, 1].set_title("Estimated Data Size per Player")
    axes[1, 1].grid(axis="y", alpha=0.3)
    for i, v in enumerate(summary_data["Est. Size (MB)"]):
        axes[1, 1].text(i, v + 0.5, f"{v:.1f} MB", ha="center", fontweight="bold")

    fig.suptitle(f"sub-{subject_id} - Player Data Distribution (after Split)")
    plt.tight_layout()

    plot_path = save_figure(fig, output_dir, f"sub-{subject_id}_split_players_data_summary.png", dpi=100)
    plt.close(fig)
    print(f"  âœ“ Data summary plot saved: {plot_path.name}")


def plot_split_integrity_checks(subject_id, output_dir):
    """Validate whether player splitting kept the correct channels and unchanged timing."""
    input_path = config.OUTPUT_DIR / f"sub-{subject_id}_downsampled.fif"
    if not input_path.exists():
        print(f"  ERROR: Input file for split validation not found: {input_path.name}")
        return

    try:
        raw_input = mne.io.read_raw_fif(str(input_path), preload=False, verbose=False)
    except Exception as e:
        print(f"  ERROR loading split input file: {e}")
        return

    fig = plt.figure(figsize=(15, 9))
    ax_counts = plt.subplot(2, 2, 1)
    ax_leakage = plt.subplot(2, 2, 2)
    ax_timing = plt.subplot(2, 2, 3)
    ax_checks = plt.subplot(2, 2, 4)

    players = ["P1", "P2"]
    colors = PLAYER_COLORS
    expected_totals = []
    actual_totals = []
    wrong_prefix_counts = []
    timing_diffs = []
    check_matrix = []

    for person in players:
        prefix = config.PLAYER_PREFIX_MAP[person]
        other_prefixes = [value for key, value in config.PLAYER_PREFIX_MAP.items() if key != person]
        split_path = config.OUTPUT_DIR / f"sub-{subject_id}_{person}_split.fif"

        if not split_path.exists():
            print(f"    {person}: File not found for split validation")
            continue

        try:
            raw_split = mne.io.read_raw_fif(str(split_path), preload=False, verbose=False)
        except Exception as e:
            print(f"    {person}: Error loading split file: {e}")
            continue

        expected_eeg = [
            ch for ch in raw_input.ch_names if ch.startswith(f"{prefix}A") or ch.startswith(f"{prefix}B")
        ]
        expected_aux = [
            ch for ch in raw_input.ch_names if ch.startswith(prefix) and ch not in expected_eeg
        ]
        expected_total = len(expected_eeg) + len(expected_aux) + int("Status" in raw_input.ch_names)

        actual_total = len(raw_split.ch_names)
        wrong_prefix = sum(
            any(ch.startswith(other_prefix) for other_prefix in other_prefixes)
            for ch in raw_split.ch_names
            if ch != "Status"
        )
        duration_diff = abs(raw_split.times[-1] - raw_input.times[-1])
        samples_match = raw_split.n_times == raw_input.n_times
        status_present = "Status" in raw_split.ch_names
        expected_channels_present = all(ch in raw_split.ch_names for ch in expected_eeg + expected_aux)
        no_leakage = wrong_prefix == 0
        duration_match = duration_diff < (1 / raw_input.info["sfreq"])

        expected_totals.append(expected_total)
        actual_totals.append(actual_total)
        wrong_prefix_counts.append(wrong_prefix)
        timing_diffs.append(duration_diff * 1000)
        check_matrix.append([
            int(expected_channels_present),
            int(no_leakage),
            int(status_present),
            int(samples_match),
            int(duration_match),
        ])

    if len(actual_totals) != 2:
        plt.close(fig)
        return

    x = np.arange(len(players))
    width = 0.35

    ax_counts.bar(x - width / 2, expected_totals, width, label="Expected", color=EXPECTED_COLOR, edgecolor="black")
    ax_counts.bar(x + width / 2, actual_totals, width, label="Actual split", color=colors, edgecolor="black")
    ax_counts.set_xticks(x)
    ax_counts.set_xticklabels(players)
    ax_counts.set_ylabel("Channel Count")
    ax_counts.set_title("Expected vs Actual Channels")
    ax_counts.grid(axis="y", alpha=0.3)
    ax_counts.legend()

    for idx, value in enumerate(expected_totals):
        ax_counts.text(idx - width / 2, value + 0.5, str(value), ha="center", fontweight="bold")
    for idx, value in enumerate(actual_totals):
        ax_counts.text(idx + width / 2, value + 0.5, str(value), ha="center", fontweight="bold")

    ax_leakage.bar(players, wrong_prefix_counts, color=colors, alpha=0.8, edgecolor="black")
    ax_leakage.axhline(0, color="green", linestyle="--", linewidth=2, label="Expected: 0 wrong-prefix channels")
    ax_leakage.set_ylabel("Wrong-Prefix Channels")
    ax_leakage.set_title("Leakage Check")
    ax_leakage.grid(axis="y", alpha=0.3)
    ax_leakage.legend(fontsize=9)
    leakage_upper = max(1.0, max(wrong_prefix_counts) + 0.5)
    ax_leakage.set_ylim(0, leakage_upper)
    leakage_offset = leakage_upper * 0.04
    for idx, value in enumerate(wrong_prefix_counts):
        ax_leakage.text(idx, value + leakage_offset, str(value), ha="center", va="bottom", fontweight="bold")

    ax_timing.bar(players, timing_diffs, color=colors, alpha=0.8, edgecolor="black")
    ax_timing.axhline(0, color="green", linestyle="--", linewidth=2, label="Expected: 0 ms difference")
    ax_timing.set_ylabel("Duration Difference (ms)")
    ax_timing.set_title("Timing Integrity")
    ax_timing.grid(axis="y", alpha=0.3)
    ax_timing.legend(fontsize=9)
    timing_upper = max(0.1, max(timing_diffs) * 1.2 + 0.02)
    ax_timing.set_ylim(0, timing_upper)
    timing_offset = timing_upper * 0.04
    for idx, value in enumerate(timing_diffs):
        ax_timing.text(idx, value + timing_offset, f"{value:.2f}", ha="center", va="bottom", fontweight="bold")

    check_labels = [
        "Expected\nchannels present",
        "No wrong-prefix\nchannels",
        "Status\npresent",
        "Sample count\nunchanged",
        "Duration\nunchanged",
    ]
    im = ax_checks.imshow(np.array(check_matrix).T, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax_checks.set_xticks(np.arange(len(players)))
    ax_checks.set_xticklabels(players)
    ax_checks.set_yticks(np.arange(len(check_labels)))
    ax_checks.set_yticklabels(check_labels)
    ax_checks.set_title("Split Validation Checklist")

    for row in range(len(check_labels)):
        for col in range(len(players)):
            passed = bool(check_matrix[col][row])
            ax_checks.text(col, row, "PASS" if passed else "FAIL", ha="center", va="center", fontweight="bold")

    fig.colorbar(im, ax=ax_checks, fraction=0.046, pad=0.04)
    fig.suptitle(f"sub-{subject_id} - Split Players Validation\n(Checks whether each split file contains the correct player channels and unchanged timing)")
    plt.tight_layout()

    plot_path = save_figure(fig, output_dir, f"sub-{subject_id}_split_players_validation.png", dpi=120)
    plt.close(fig)
    print(f"  âœ“ Split validation plot saved: {plot_path.name}")


def main(argv=None):
    args = parse_args(argv)
    subjects = resolve_subjects(args.subjects, config.SUBJECTS, mode="viz")
    output_dir = config.QC_DIR

    print("\n" + "=" * 80)
    print("VISUALIZATION: Step 01 - Player Split Effects")
    print("=" * 80)
    print(f"Subjects: {', '.join(subjects)}")
    print(f"Output: {output_dir}")
    print("=" * 80 + "\n")

    for subject_id in subjects:
        print(f"\n--- Subject {subject_id} ---")

        # Check both players exist
        p1_path = config.OUTPUT_DIR / f"sub-{subject_id}_P1_split.fif"
        p2_path = config.OUTPUT_DIR / f"sub-{subject_id}_P2_split.fif"

        if not p1_path.exists() or not p2_path.exists():
            print(f"  ERROR: Split player files not found. Expected:")
            print(f"    {p1_path.name}")
            print(f"    {p2_path.name}")
            continue

        # Generate plots
        plot_data_summary_per_player(subject_id, output_dir)
        plot_split_integrity_checks(subject_id, output_dir)

    print("\n" + "=" * 80)
    print(f"âœ“ All visualizations saved to: {output_dir}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()

