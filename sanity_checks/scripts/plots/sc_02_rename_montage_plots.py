"""
Sanity Check Plot Module for Step 02: Rename & Set Montage

Creates comparison plots for channel renaming and montage setup:
- Channel name mapping (before: BioSemi -> after: 10-20 system)
- 2D topomap visualization of electrode positions
- 3D visualization of electrode coordinates (from montage)
- Montage quality and consistency checks

Entry point:
    python sanity_checks/scripts/sc_02_rename_montage.py --mode viz

Options:
    --subjects: Comma-separated subject IDs (default: first 2)

REASONING:
- Purpose: document the transition from acquisition-specific labels to interpretable scalp positions.
- Reproducibility: the mapping and montage are fixed, so repeated runs should yield the same label list and sensor geometry.
- Interpretation focus: the expected argument is "This seems correct because standard channel names and plausible scalp positions appear together after the step."
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import mne
import numpy as np

CURRENT_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = CURRENT_DIR.parent.parent / "eeg_pipeline"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from preprocessing import config
from helpers.sc_cli import add_subjects_argument, resolve_subjects
from helpers.sc_config import DEFAULT_PERSONS, MONTAGE_VIZ
from helpers.sc_plot_io import save_figure


COLOR_SENSOR = MONTAGE_VIZ["sensor"]
COLOR_BEFORE = MONTAGE_VIZ["before"]
COLOR_AFTER = MONTAGE_VIZ["after"]
COLOR_PASS = MONTAGE_VIZ["pass"]
COLOR_FAIL = MONTAGE_VIZ["fail"]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Visualize rename+montage effects (channel mapping and electrode layout)",
    )
    add_subjects_argument(parser)
    return parser.parse_args(argv)


def plot_montage_topomap(raw, subject_id, person, output_dir):
    """Plot 2D topomap showing electrode positions after montage setup."""
    raw_eeg = raw.copy().pick_types(eeg=True)

    if len(raw_eeg.ch_names) == 0:
        return

    fig = raw_eeg.plot_sensors(kind="topomap", show_names=True, show=False, sphere="eeglab")
    fig.subtitle(f"sub-{subject_id} {person} - EEG Sensor Layout (Montage)")

    plot_path = save_figure(fig, output_dir, f"sub-{subject_id}_{person}_montage_topomap.png", dpi=150)
    plt.close(fig)
    print(f"  OK Montage topomap saved: {plot_path.name}")


def plot_montage_3d(raw, subject_id, person, output_dir):
    """Plot 3D electrode coordinates from the applied montage."""
    raw_eeg = raw.copy().pick_types(eeg=True)
    montage = raw_eeg.get_montage()

    if montage is None:
        print(f"  {person}: No montage attached, skipping 3D montage plot")
        return

    ch_pos = montage.get_positions().get("ch_pos", {})
    names = [ch for ch in raw_eeg.ch_names if ch in ch_pos]
    if not names:
        print(f"  {person}: No channel positions found in montage, skipping 3D montage plot")
        return

    coords = np.array([ch_pos[ch] for ch in names], dtype=float)

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
               s=70, c=COLOR_SENSOR, edgecolors="black", alpha=0.9)

    for idx, ch in enumerate(names):
        ax.text(coords[idx, 0], coords[idx, 1], coords[idx, 2], ch, fontsize=7)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.yaxis.labelpad = 12
    ax.set_title(f"sub-{subject_id} {person} - 3D Electrode Coordinates")
    ax.view_init(elev=25, azim=45)
    ax.grid(True, alpha=0.3)
    fig.subplots_adjust(left=0.08, right=0.95, bottom=0.08, top=0.90)

    plot_path = save_figure(
        fig,
        output_dir,
        f"sub-{subject_id}_{person}_montage_3d.png",
        dpi=150,
        pad_inches=0.25,
    )
    plt.close(fig)
    print(f"  OK 3D montage plot saved: {plot_path.name}")


def plot_channel_naming_summary(raw_before, raw_after, subject_id, person, output_dir):
    """Visualize channel naming changes before and after montage setup."""
    # Extract EEG channels
    eeg_before = mne.pick_types(raw_before.info, eeg=True)
    eeg_after = mne.pick_types(raw_after.info, eeg=True)

    # Channel names
    names_before = [raw_before.ch_names[i] for i in eeg_before]
    names_after = [raw_after.ch_names[i] for i in eeg_after]

    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 8))

    # Before (BioSemi naming)
    axes[0].text(0.5, 0.95, "BEFORE (Split Step Output)", ha="center", va="top", fontsize=12, fontweight="bold", transform=axes[0].transAxes)
    y_pos = 0.90
    for i, name in enumerate(names_before[:16]):  # Show first 16
        axes[0].text(0.05, y_pos, f"{i+1:2d}. {name}", family="monospace", fontsize=9, transform=axes[0].transAxes)
        y_pos -= 0.055
    if len(names_before) > 16:
        axes[0].text(0.05, y_pos, f"... and {len(names_before) - 16} more channels", fontsize=9, style="italic", transform=axes[0].transAxes)
    
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1)
    axes[0].axis("off")

    # After (10-20 naming)
    axes[1].text(0.5, 0.95, "AFTER (Rename+Montage Step Output)", ha="center", va="top", fontsize=12, fontweight="bold", transform=axes[1].transAxes)
    y_pos = 0.90
    for i, name in enumerate(names_after[:16]):  # Show first 16
        axes[1].text(0.05, y_pos, f"{i+1:2d}. {name}", family="monospace", fontsize=9, transform=axes[1].transAxes)
        y_pos -= 0.055
    if len(names_after) > 16:
        axes[1].text(0.05, y_pos, f"... and {len(names_after) - 16} more channels", fontsize=9, style="italic", transform=axes[1].transAxes)
    
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1)
    axes[1].axis("off")

    fig.suptitle(f"sub-{subject_id} {person} - Channel Name Mapping")
    plt.tight_layout()

    plot_path = save_figure(fig, output_dir, f"sub-{subject_id}_{person}_montage_channel_mapping.png", dpi=100)
    plt.close(fig)
    print(f"  OK Channel mapping summary saved: {plot_path.name}")


def plot_montage_coverage_stats(raw_before, raw_after, subject_id, person, output_dir):
    """Plot montage coverage and channel statistics."""
    eeg_picks_before = mne.pick_types(raw_before.info, eeg=True)
    eeg_picks_after = mne.pick_types(raw_after.info, eeg=True)
    names_after = [raw_after.ch_names[i] for i in eeg_picks_after]

    # Check digit information
    has_montage = raw_after.info.get("dig") is not None and len(raw_after.info["dig"]) > 0
    montage_present = raw_after.get_montage() is not None
    montage_name = "standard_1020" if montage_present else "none"
    renamed_1020_names = set(config.channel_labels.values())
    recognized_renamed = sum(1 for ch in names_after if ch in renamed_1020_names)
    eeg_count_preserved = len(eeg_picks_before) == len(eeg_picks_after)
    all_eeg_labels_recognized = recognized_renamed == len(names_after)
    unique_channel_names = len(set(names_after)) == len(names_after)

    fig = plt.figure(figsize=(15, 8))
    grid = fig.add_gridspec(2, 2, width_ratios=[1.25, 0.85], height_ratios=[2.2, 1.3], hspace=0.35, wspace=0.30)
    ax_counts = fig.add_subplot(grid[0, 0])
    ax_pass = fig.add_subplot(grid[0, 1])
    ax_info = fig.add_subplot(grid[1, :])

    # Channel counts
    categories = ["Before\nSplit", "After\nRename + Montage"]
    channel_counts = [len(eeg_picks_before), len(eeg_picks_after)]
    colors = [COLOR_BEFORE, COLOR_AFTER]
    
    ax_counts.bar(categories, channel_counts, color=colors, alpha=0.7, edgecolor="black", width=0.6)
    ax_counts.set_ylabel("Count")
    ax_counts.set_title("EEG Channel Count")
    ax_counts.grid(axis="y", alpha=0.3)
    for i, v in enumerate(channel_counts):
        ax_counts.text(i, v + 0.5, str(v), ha="center", fontweight="bold")

    # Montage information
    montage_info = ["Montage Information", ""]
    if has_montage:
        montage_info.append(f"OK Electrode positions available: {len(raw_after.info['dig'])}")
        montage_info.append(f"OK Montage applied: {montage_name}")
    else:
        montage_info.append("FAIL No electrode positions")
    
    montage_info.extend([
        f"OK Total channels: {len(raw_after.ch_names)}",
        f"OK EEG channels: {len(eeg_picks_after)}",
        f"OK Renamed EEG labels: {recognized_renamed}/{len(names_after)}",
    ])

    check_labels = [
        "EEG count preserved",
        "All EEG labels renamed",
        "Unique EEG labels",
        "Montage attached",
        "Electrode positions present",
    ]
    check_values = [
        int(eeg_count_preserved),
        int(all_eeg_labels_recognized),
        int(unique_channel_names),
        int(montage_present),
        int(has_montage),
    ]
    passed_count = sum(check_values)
    total_checks = len(check_values)
    pass_rate = 100 * passed_count / total_checks
    check_colors = [COLOR_PASS if passed else COLOR_FAIL for passed in check_values]
    y_pos = np.arange(len(check_labels))
    ax_pass.barh(y_pos, np.ones(len(check_labels)), color=check_colors, alpha=0.85, edgecolor="black")
    ax_pass.set_xlim(0, 1)
    ax_pass.set_xticks([0, 1])
    ax_pass.set_xticklabels(["0", "1"])
    ax_pass.set_xlabel("Check outcome (0=FAIL, 1=PASS)")
    ax_pass.set_yticks(y_pos)
    ax_pass.set_yticklabels(check_labels)
    ax_pass.yaxis.tick_right()
    ax_pass.tick_params(axis="y", labelleft=False, labelright=True, pad=6)
    ax_pass.invert_yaxis()
    ax_pass.set_title(f"Rename + Montage Validation ({passed_count}/{total_checks}, {pass_rate:.0f}%)")
    ax_pass.grid(False)

    for idx, passed in enumerate(check_values):
        ax_pass.text(0.5, idx, "PASS" if passed else "FAIL", ha="center", va="center", fontweight="bold", color="white")

    ax_info.text(0.02, 0.92, "\n".join(montage_info), transform=ax_info.transAxes,
                 fontsize=10.5, verticalalignment="top", family="monospace",
                 bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    ax_info.axis("off")

    fig.suptitle(f"sub-{subject_id} {person} - Montage Coverage & Statistics")
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    plot_path = save_figure(fig, output_dir, f"sub-{subject_id}_{person}_montage_coverage_stats.png", dpi=100)
    plt.close(fig)
    print(f"  OK Montage coverage stats saved: {plot_path.name}")


def main(argv=None):
    args = parse_args(argv)
    subjects = resolve_subjects(args.subjects, config.SUBJECTS, mode="viz")
    output_dir = config.QC_DIR

    print("\n" + "=" * 80)
    print("VISUALIZATION: Step 02 - Rename & Montage Effects")
    print("=" * 80)
    print(f"Subjects: {', '.join(subjects)}")
    print(f"Output: {output_dir}")
    print("=" * 80 + "\n")

    for subject_id in subjects:
        print(f"\n--- Subject {subject_id} ---")

        for person in DEFAULT_PERSONS:
            before_path = config.OUTPUT_DIR / f"sub-{subject_id}_{person}_split.fif"
            after_path = config.OUTPUT_DIR / f"sub-{subject_id}_{person}_renamed_montaged.fif"

            if not before_path.exists():
                print(f"  {person}: Before file (split) not found")
                continue
            if not after_path.exists():
                print(f"  {person}: After file (renamed_montaged) not found")
                continue

            try:
                raw_before = mne.io.read_raw_fif(str(before_path), preload=False, verbose=False)
                raw_after = mne.io.read_raw_fif(str(after_path), preload=False, verbose=False)

                print(f"\n  {person}:")
                # Generate plots
                plot_channel_naming_summary(raw_before, raw_after, subject_id, person, output_dir)
                plot_montage_topomap(raw_after, subject_id, person, output_dir)
                plot_montage_3d(raw_after, subject_id, person, output_dir)
                plot_montage_coverage_stats(raw_before, raw_after, subject_id, person, output_dir)

            except Exception as e:
                print(f"  {person}: Error: {e}")

    print("\n" + "=" * 80)
    print(f"OK All visualizations saved to: {output_dir}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()

