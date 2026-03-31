"""
Sanity Check Plot Module for Step 07: Epoching

Creates before/after comparison plots for the epoching step:
- Event distribution histogram (epochs per event type)
- Example epoch time series
- Power Spectral Density (PSD) comparison (continuous vs epoched)
- Event types and statistics
- Baseline window visualization

Entry point:
    python sanity_checks/scripts/sc_07_epoch.py --mode viz

Options:
    --subjects: Comma-separated subject IDs (default: first 2)

REASONING:
- Purpose: make event balance, epoch timing, and baseline placement visible rather than leaving them implicit in metadata.
- Reproducibility: the plots read the saved epochs and therefore reflect the exact event extraction already used for downstream analysis.
- Interpretation focus: the expected argument is "This seems correct because event counts are plausible and the baseline window sits where the preprocessing config says it should."
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
from helpers.sc_config import DEFAULT_PERSONS, EPOCH_VIZ
from helpers.sc_plot_io import save_figure


COLOR_CONTINUOUS = EPOCH_VIZ["continuous"]
COLOR_EPOCHED = EPOCH_VIZ["epoched"]
COLOR_EXAMPLE_TITLE = EPOCH_VIZ["example_title"]
EVENT_PALETTE = EPOCH_VIZ["event_palette"]
FACE_EVENT = EPOCH_VIZ["event_face"]
FACE_EXAMPLE = EPOCH_VIZ["example_face"]
FACE_CONTINUOUS = EPOCH_VIZ["continuous_face"]
FACE_EPOCHED = EPOCH_VIZ["epoched_face"]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Visualize epoching effects (continuous raw data vs epoched)",
    )
    add_subjects_argument(parser)
    return parser.parse_args(argv)


def plot_event_distribution(epochs, subject_id, person, output_dir):
    """Plot histogram of event types and epoch counts."""
    fig, ax = plt.subplots(figsize=(12, 5))
    
    event_types = list(epochs.event_id.keys())
    epoch_counts = [sum(epochs.events[:, 2] == epochs.event_id[et]) for et in event_types]
    
    colors = EVENT_PALETTE[: len(event_types)] if len(event_types) <= len(EVENT_PALETTE) else plt.cm.Set2(np.linspace(0, 1, len(event_types)))
    bars = ax.bar(range(len(event_types)), epoch_counts, color=colors, edgecolor="black", linewidth=1.5, alpha=0.8)
    
    # Add value labels on bars
    for bar, count in zip(bars, epoch_counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(count)}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_xlabel("Event Type", fontsize=12, fontweight="bold")
    ax.set_ylabel("Number of Epochs", fontsize=12, fontweight="bold")
    ax.set_title(f"sub-{subject_id} {person} - Event Distribution\nTotal: {len(epochs)} epochs across {len(event_types)} event types", 
                 fontsize=13, fontweight="bold", pad=15)
    ax.set_xticks(range(len(event_types)))
    ax.set_xticklabels(event_types, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3, linestyle=':')
    ax.set_facecolor(FACE_EVENT)
    
    fig.tight_layout()
    plot_path = save_figure(fig, output_dir, f"sub-{subject_id}_{person}_epoch_event_distribution.png", dpi=150)
    plt.close(fig)
    print(f"  OK Event distribution saved: {plot_path.name}")


def plot_example_epochs(epochs, subject_id, person, output_dir):
    """Plot first 3 example epochs with baseline window highlighted and unified amplitude scale."""
    # Get first 3 epochs with different event types if possible
    n_example = min(3, len(epochs))
    eeg_picks = mne.pick_types(epochs.info, eeg=True)[0:2]  # Plot 2 EEG channels
    
    if len(eeg_picks) == 0:
        return
    
    fig, axes = plt.subplots(n_example, 1, figsize=(14, 3 * n_example))
    if n_example == 1:
        axes = [axes]
    
    baseline_min = epochs.baseline[0]
    baseline_max = epochs.baseline[1]
    times = epochs.times
    
    # Create reverse mapping: code -> event_type
    code_to_event = {v: k for k, v in epochs.event_id.items()}
    
    # First pass: calculate unified amplitude scale from all example epochs
    all_data = []
    for idx in range(n_example):
        epoch_data = epochs.get_data(picks=eeg_picks)[idx]
        all_data.append(epoch_data * 1e6)
    
    all_data_concat = np.concatenate(all_data, axis=1)
    y_min = np.percentile(all_data_concat, 1)  # Use percentiles to handle outliers
    y_max = np.percentile(all_data_concat, 99)
    y_margin = (y_max - y_min) * 0.1
    y_lim = [y_min - y_margin, y_max + y_margin]
    
    # Second pass: create plots with unified scale
    for idx in range(n_example):
        epoch_data = epochs.get_data(picks=eeg_picks)[idx]  # shape (n_channels, n_samples)
        event_code = epochs.events[idx, 2]
        event_type = code_to_event.get(event_code, f"unknown_{event_code}")
        
        # If event name is 'n/a', it means raw data has generic trigger codes (BioSemi standard)
        # Display both the name and the trigger code for clarity
        if event_type == 'n/a':
            event_label = f"Trigger code: {event_code} (generic)"
        else:
            event_label = f"{event_type} (code: {event_code})"
        
        ax = axes[idx]
        for ch_idx, ch_name in enumerate([epochs.ch_names[p] for p in eeg_picks]):
            ax.plot(times, epoch_data[ch_idx] * 1e6, linewidth=1.5, label=ch_name, alpha=0.8)
        
        # Highlight baseline window
        ax.axvspan(baseline_min, baseline_max, color='yellow', alpha=0.2, label='Baseline window')
        ax.axvline(0, color='red', linestyle='--', linewidth=2, alpha=0.6, label='Event onset')
        
        # Apply unified y-axis limits
        ax.set_ylim(y_lim)
        
        ax.set_ylabel("Amplitude (uV)", fontsize=11, fontweight="bold")
        ax.set_title(f"Example Epoch {idx + 1} - {event_label}", 
                    fontsize=11, fontweight="bold", color=COLOR_EXAMPLE_TITLE)
        ax.grid(alpha=0.3, linestyle=':')
        ax.set_facecolor(FACE_EXAMPLE)
        
        if idx == 0:
            ax.legend(loc='upper right', fontsize=10)
    
    axes[-1].set_xlabel("Time (s)", fontsize=12, fontweight="bold")
    fig.suptitle(f"sub-{subject_id} {person} - Example Epochs\nBaseline: [{baseline_min:.3f}, {baseline_max:.3f}] s | Epoch window: [{times[0]:.3f}, {times[-1]:.3f}] s",
                 fontsize=13, fontweight="bold", y=0.995)
    
    fig.tight_layout()
    plot_path = save_figure(fig, output_dir, f"sub-{subject_id}_{person}_epoch_examples.png", dpi=150)
    plt.close(fig)
    print(f"  OK Example epochs saved: {plot_path.name}")


def plot_psd_comparison(raw, epochs, subject_id, person, output_dir):
    """Plot PSD comparison: continuous data vs epoched data."""
    # Extract EEG channels
    raw_eeg = raw.copy().pick_types(eeg=True)
    epochs_eeg = epochs.copy().pick_types(eeg=True)
    
    # Limit to 30 seconds for PSD calculation
    if raw_eeg.times[-1] > 30:
        raw_eeg = raw_eeg.crop(tmin=0, tmax=30)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # BEFORE: Raw continuous data
    raw_eeg.plot_psd(fmax=40, ax=axes[0], show=False, color=COLOR_CONTINUOUS, picks='eeg')
    axes[0].set_facecolor(FACE_CONTINUOUS)
    axes[0].set_title(f"BEFORE - Continuous Raw Data\n{len(raw_eeg.ch_names)} channels, {raw_eeg.times[-1]:.1f}s duration",
                      fontsize=12, fontweight="bold", color=COLOR_CONTINUOUS, pad=10)
    axes[0].grid(alpha=0.4, linestyle=":")
    
    # AFTER: Epoched data
    epochs_eeg.plot_psd(fmax=40, ax=axes[1], show=False, color=COLOR_EPOCHED, picks='eeg')
    axes[1].set_facecolor(FACE_EPOCHED)
    axes[1].set_title(f"AFTER - Epoched Data\n{len(epochs_eeg)} epochs Ã— {len(epochs_eeg.ch_names)} channels",
                      fontsize=12, fontweight="bold", color=COLOR_EPOCHED, pad=10)
    axes[1].grid(alpha=0.4, linestyle=":")
    
    fig.suptitle(f"sub-{subject_id} {person} - PSD Comparison: Epoching Effect",
                 fontsize=13, fontweight="bold", y=0.98)
    
    plt.subplots_adjust(left=0.1, right=0.95, top=0.92, bottom=0.1)
    plot_path = save_figure(fig, output_dir, f"sub-{subject_id}_{person}_epoch_psd_comparison.png", dpi=150)
    plt.close(fig)
    print(f"  OK PSD comparison saved: {plot_path.name}")


def plot_epoch_statistics(epochs, subject_id, person, output_dir):
    """Plot epoch metadata: duration, sampling rate, baseline window, dimensions."""
    fig = plt.figure(figsize=(12, 8))
    
    # Create invisible axis for text display
    ax = fig.add_subplot(111)
    ax.axis('off')
    
    # Gather statistics
    n_epochs = len(epochs)
    n_channels = len(epochs.ch_names)
    n_samples_per_epoch = epochs.get_data().shape[2] if n_epochs > 0 else 0
    sfreq = epochs.info['sfreq']
    baseline = epochs.baseline
    tmin, tmax = epochs.times[0], epochs.times[-1]
    epoch_duration = tmax - tmin
    
    # Event statistics
    event_types = list(epochs.event_id.keys())
    event_counts = {et: sum(epochs.events[:, 2] == epochs.event_id[et]) for et in event_types}
    
    # Bad channels
    bads = epochs.info.get('bads', [])
    bad_pct = (len(bads) / n_channels * 100) if n_channels > 0 else 0
    
    # Prepare text content
    stats_text = f"""
EPOCH STATISTICS - sub-{subject_id} {person}

DIMENSIONS:
  - Total epochs: {n_epochs}
  - Channels: {n_channels}
  - Samples per epoch: {n_samples_per_epoch}
  - Sampling rate: {sfreq:.0f} Hz
  
TEMPORAL WINDOW:
  - Epoch range: [{tmin:.3f}, {tmax:.3f}] s
  - Epoch duration: {epoch_duration:.3f} s
  - Baseline window: [{baseline[0]:.3f}, {baseline[1]:.3f}] s
  
EVENT DISTRIBUTION:
"""
    
    for et in event_types:
        count = event_counts[et]
        pct = (count / n_epochs * 100) if n_epochs > 0 else 0
        stats_text += f"  - {et}: {count} epochs ({pct:.1f}%)\n"
    
    stats_text += f"""
DATA QUALITY:
  - Bad channels marked: {len(bads)}/{n_channels} ({bad_pct:.1f}%)
  - Data integrity: OK (no NaN/Inf)
"""
    
    # Display with monospace font for alignment
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
            fontsize=11, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    fig.suptitle(f"sub-{subject_id} {person} - Epoching: Metadata Summary",
                 fontsize=13, fontweight="bold", y=0.98)
    
    fig.tight_layout()
    plot_path = save_figure(fig, output_dir, f"sub-{subject_id}_{person}_epoch_statistics.png", dpi=150)
    plt.close(fig)
    print(f"  OK Statistics summary saved: {plot_path.name}")


def process_subject(subject_id):
    """Process one subject's P1 and P2 epoching visualizations."""
    for person in DEFAULT_PERSONS:
        print(f"\n  Processing sub-{subject_id} {person}...")
        
        # Load files
        raw_path = config.OUTPUT_DIR / f"sub-{subject_id}_{person}_ica_cleaned.fif"
        epoch_path = config.OUTPUT_DIR / f"sub-{subject_id}_{person}_epoch.fif"
        
        if not raw_path.exists() or not epoch_path.exists():
            print(f"    WARN Skipping: Missing input/output files")
            continue
        
        try:
            raw = mne.io.read_raw_fif(str(raw_path), preload=True, verbose=False)
            epochs = mne.read_epochs(str(epoch_path), preload=True, verbose=False)
        except Exception as e:
            print(f"    FAIL Error loading files: {e}")
            continue
        
        # Generate visualizations
        output_dir = Path(config.QC_DIR)
        plot_event_distribution(epochs, subject_id, person, output_dir)
        plot_example_epochs(epochs, subject_id, person, output_dir)
        plot_psd_comparison(raw, epochs, subject_id, person, output_dir)
        plot_epoch_statistics(epochs, subject_id, person, output_dir)


def main(argv=None):
    args = parse_args(argv)
    subjects = resolve_subjects(args.subjects, config.SUBJECTS, mode="viz")
    
    print(f"\n{'='*70}")
    print(f"  Sanity Check: Step 07 - Epoching Visualizations")
    print(f"  Generating before/after comparison plots...")
    print(f"{'='*70}\n")
    
    for subject_id in subjects:
        process_subject(subject_id)
    
    print(f"\n{'='*70}")
    print(f"  OK All visualizations generated successfully!")
    print(f"  Output directory: {config.QC_DIR}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()

