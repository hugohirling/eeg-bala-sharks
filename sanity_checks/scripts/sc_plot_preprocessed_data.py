"""
Plot cleaned preprocessed EEG data after pipeline completion.

Usage:
    python sanity_checks/scripts/sc_plot_preprocessed_data.py [--subjects 01,02] [--step 06] [--duration 60]

Options:
    --subjects: Comma-separated subject IDs (default: first 3)
    --step: Pipeline step to visualize (05=filtered, 06=ica_cleaned, 07=epochs; default: 06)
    --duration: Duration in seconds to plot for continuous data (default: 60)
    --save-dir: Output directory for plots (default: output/qc/)

Examples:
    python sanity_checks/scripts/sc_plot_preprocessed_data.py --subjects 01,02,03
    python sanity_checks/scripts/sc_plot_preprocessed_data.py --step 07  # Plot epochs
    python sanity_checks/scripts/sc_plot_preprocessed_data.py --duration 120

REASONING:
- Purpose: give a lightweight report-ready visualization of later preprocessing outputs for quick qualitative inspection.
- Reproducibility: the selected pipeline step is explicit in the CLI and maps to fixed filename suffixes.
- Interpretation focus: the expected argument is "This seems correct because the displayed data quality matches the step-specific expectation, for example cleaner frontal activity after ICA."
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


STEP_SUFFIXES = {
    "05": "filtered",
    "06": "ica_cleaned",
    "07": "epoch",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize preprocessed EEG data after pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--subjects",
        type=str,
        default=None,
        help="Comma-separated subject IDs (e.g., '01,02,03'). Default: first 3.",
    )
    parser.add_argument(
        "--step",
        type=str,
        default="06",
        choices=["05", "06", "07"],
        help="Pipeline step to visualize (05=filtered, 06=ica_cleaned, 07=epochs; default: 06)",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=60,
        help="Duration in seconds to plot (for continuous data; default: 60)",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="Output directory for plots (default: output/qc/)",
    )
    return parser.parse_args()


def get_subjects(subject_str):
    """Parse subject selection from command line."""
    if subject_str:
        return subject_str.split(",")
    # Default: first 3 subjects
    return list(config.SUBJECTS)[:3]


def plot_raw_timeseries(raw, subject_id, person, step, duration, output_dir):
    """Plot raw time series with topography."""
    t_end = min(duration, raw.times[-1])
    t_idx_end = int(t_end * raw.info["sfreq"])

    eeg_picks = mne.pick_types(raw.info, eeg=True)
    if len(eeg_picks) == 0:
        print(f"  No EEG channels found for sub-{subject_id} {person}")
        return

    fig = raw.plot(
        picks=eeg_picks[:16],  # Plot first 16 channels
        start=0,
        duration=t_end,
        show=False,
        n_channels=16,
        scalings="auto",
    )
    fig.suptitle(f"sub-{subject_id} {person} - Step {step} - Time Series (first {t_end:.0f}s, first 16 EEG channels)")
    fig.set_size_inches(16, 10)

    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / f"sub-{subject_id}_{person}_step{step}_timeseries.png"
    fig.savefig(plot_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  OK Time series plot saved: {plot_path.name}")


def plot_psd(raw, subject_id, person, step, output_dir):
    """Plot power spectral density."""
    raw_eeg = raw.copy().pick_types(eeg=True)

    fig, ax = plt.subplots(figsize=(12, 6))
    raw_eeg.plot_psd(fmax=60, ax=ax, show=False)
    ax.set_title(f"sub-{subject_id} {person} - Step {step} - Power Spectral Density")

    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / f"sub-{subject_id}_{person}_step{step}_psd.png"
    fig.savefig(plot_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  OK PSD plot saved: {plot_path.name}")


def plot_topomap(raw, subject_id, person, step, output_dir):
    """Plot sensor layout."""
    raw_eeg = raw.copy().pick_types(eeg=True)

    # Keep MNE's standard topomap projection from the 3D montage coordinates.
    fig = raw_eeg.plot_sensors(
        kind="topomap",
        show_names=True,
        show=False,
        sphere="eeglab",
    )
    fig.suptitle(f"sub-{subject_id} {person} - Step {step} - Sensor Layout")

    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / f"sub-{subject_id}_{person}_step{step}_sensors.png"
    fig.savefig(plot_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  OK Sensor layout plot saved: {plot_path.name}")


def plot_epochs(epochs, subject_id, person, output_dir):
    """Plot sample epochs with statistics."""
    if len(epochs) == 0:
        print(f"  No epochs for sub-{subject_id} {person}")
        return

    # Plot first 4 epochs
    n_epochs_to_plot = min(4, len(epochs))
    eeg_picks = mne.pick_types(epochs.info, eeg=True)

    fig = epochs[:n_epochs_to_plot].plot(
        show=False,
        n_epochs=n_epochs_to_plot,
    )
    fig.suptitle(f"sub-{subject_id} {person} - First {n_epochs_to_plot} Epochs")
    fig.set_size_inches(14, 8)

    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / f"sub-{subject_id}_{person}_epochs_sample.png"
    fig.savefig(plot_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  OK Sample epochs plot saved: {plot_path.name}")

    # Plot event distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    event_counts = {}
    for event_type, event_id in epochs.event_id.items():
        count = len(epochs[event_type])
        event_counts[event_type] = count

    if event_counts:
        ax.bar(range(len(event_counts)), list(event_counts.values()))
        ax.set_xticks(range(len(event_counts)))
        ax.set_xticklabels(list(event_counts.keys()), rotation=45, ha="right")
        ax.set_ylabel("Count")
        ax.set_title(f"sub-{subject_id} {person} - Epoch Event Distribution (n={len(epochs)} total)")
        ax.grid(axis="y", alpha=0.3)

    plot_path = output_dir / f"sub-{subject_id}_{person}_epochs_distribution.png"
    fig.savefig(plot_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  OK Epoch distribution plot saved: {plot_path.name}")


def plot_data_quality(raw, subject_id, person, step, output_dir):
    """Plot data quality metrics."""
    eeg_picks = mne.pick_types(raw.info, eeg=True)
    if len(eeg_picks) == 0:
        return

    # Get sample of data
    max_samples = min(int(raw.info["sfreq"] * 120), raw.n_times)
    data = raw.get_data(picks=eeg_picks, start=0, stop=max_samples)

    # Compute metrics per channel
    channel_names = [raw.ch_names[idx] for idx in eeg_picks]
    std_values = np.std(data, axis=1)
    min_values = np.min(data, axis=1)
    max_values = np.max(data, axis=1)

    # Plot amplitude distribution
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].hist(std_values * 1e6, bins=20, edgecolor="black", alpha=0.7)
    axes[0].set_xlabel("Standard Deviation (uV)")
    axes[0].set_ylabel("Channel Count")
    axes[0].set_title("STD Distribution (sample)")
    axes[0].grid(alpha=0.3)

    axes[1].scatter(range(len(std_values)), std_values * 1e6, alpha=0.6)
    axes[1].set_xlabel("Channel Index")
    axes[1].set_ylabel("Std (uV)")
    axes[1].set_title("Per-Channel Amplitude")
    axes[1].grid(alpha=0.3)

    axes[2].scatter(min_values * 1e6, max_values * 1e6, alpha=0.6)
    axes[2].set_xlabel("Min (uV)")
    axes[2].set_ylabel("Max (uV)")
    axes[2].set_title("Min vs Max (sample)")
    axes[2].grid(alpha=0.3)

    fig.suptitle(f"sub-{subject_id} {person} - Step {step} - Data Quality Metrics")
    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / f"sub-{subject_id}_{person}_step{step}_quality.png"
    fig.savefig(plot_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  OK Quality metrics plot saved: {plot_path.name}")


def main():
    args = parse_args()

    subjects = get_subjects(args.subjects)
    step = args.step
    suffix = STEP_SUFFIXES.get(step, "ica_cleaned")
    duration = args.duration

    output_dir = Path(args.save_dir) if args.save_dir else config.QC_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("PLOT PREPROCESSED DATA")
    print("=" * 80)
    print(f"Step: {step} ({suffix})")
    print(f"Subjects: {', '.join(subjects)}")
    print(f"Duration: {duration}s")
    print(f"Output: {output_dir}")
    print("=" * 80 + "\n")

    for subject_id in subjects:
        print(f"\n--- Subject {subject_id} ---")

        for person in ["P1", "P2"]:
            if step == "07":  # Epochs
                file_path = config.OUTPUT_DIR / f"sub-{subject_id}_{person}_{suffix}.fif"
                if not file_path.exists():
                    print(f"  {person}: Epoch file not found: {file_path.name}")
                    continue

                try:
                    epochs = mne.read_epochs(str(file_path), preload=False, verbose=False)
                    print(f"\n  {person}:")
                    print(f"    Epochs: {len(epochs)}")
                    print(f"    Channels: {len(epochs.ch_names)}")
                    print(f"    Duration per epoch: {epochs.times[-1] - epochs.times[0]:.3f}s")

                    plot_epochs(epochs, subject_id, person, output_dir)
                except Exception as e:
                    print(f"  {person}: Error loading epochs: {e}")

            else:  # Raw data (step 05 or 06)
                file_path = config.OUTPUT_DIR / f"sub-{subject_id}_{person}_{suffix}.fif"
                if not file_path.exists():
                    print(f"  {person}: File not found: {file_path.name}")
                    continue

                try:
                    raw = mne.io.read_raw_fif(str(file_path), preload=False, verbose=False)
                    print(f"\n  {person}:")
                    print(f"    Duration: {raw.times[-1]:.1f}s")
                    print(f"    Channels: {len(raw.ch_names)}")
                    print(f"    Sampling rate: {raw.info['sfreq']:.0f} Hz")
                    print(f"    Bad channels: {len(raw.info.get('bads', []))}")

                    plot_raw_timeseries(raw, subject_id, person, step, duration, output_dir)
                    plot_psd(raw, subject_id, person, step, output_dir)
                    plot_topomap(raw, subject_id, person, step, output_dir)
                    plot_data_quality(raw, subject_id, person, step, output_dir)

                except Exception as e:
                    print(f"  {person}: Error: {e}")

    print("\n" + "=" * 80)
    print(f"OK All plots saved to: {output_dir}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()

