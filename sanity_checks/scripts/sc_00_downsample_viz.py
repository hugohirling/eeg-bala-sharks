"""
Sanity Check Visualization for Step 00: Downsample

Creates before/after comparison plots for the downsampling step:
- Time series comparison (original vs downsampled)
- Power Spectral Density (PSD) comparison
- Data statistics comparison

Usage:
    python sanity_checks/scripts/sc_00_downsample_viz.py [--subjects 01,02] [--duration 30]

Options:
    --subjects: Comma-separated subject IDs (default: first 2)
    --duration: Duration in seconds to plot (default: 30)
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
from mne_bids import BIDSPath, read_raw_bids

CURRENT_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = CURRENT_DIR.parent.parent / "eeg_pipeline"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from preprocessing import config


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize downsample effects (original vs downsampled)",
    )
    parser.add_argument(
        "--subjects",
        type=str,
        default=None,
        help="Comma-separated subject IDs. Default: first 2.",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=30,
        help="Duration in seconds to plot (default: 30)",
    )
    return parser.parse_args()


def get_subjects(subject_str):
    if subject_str:
        return subject_str.split(",")
    return list(config.SUBJECTS)[:2]


def plot_timeseries_comparison(raw_orig, raw_downsampled, subject_id, duration, output_dir):
    """Plot simple comparison: Original (many samples) vs Downsampled (few samples)."""
    # Use only 1 second for clarity
    t_end = 1.0
    t_end = min(t_end, raw_orig.times[-1], raw_downsampled.times[-1])
    
    eeg_picks = mne.pick_types(raw_orig.info, eeg=True)[0:1]  # Only one channel

    if len(eeg_picks) == 0:
        return

    # Load data
    data_orig = raw_orig.get_data(picks=eeg_picks, start=0, stop=int(t_end * raw_orig.info["sfreq"]))
    data_downsampled = raw_downsampled.get_data(
        picks=eeg_picks, start=0, stop=int(t_end * raw_downsampled.info["sfreq"])
    )

    times_orig = raw_orig.times[: data_orig.shape[1]]
    times_downsampled = raw_downsampled.times[: data_downsampled.shape[1]]
    ch_name = raw_orig.ch_names[eeg_picks[0]]

    # Create figure with two subplots
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # ORIGINAL: Many samples — show ALL points
    axes[0].plot(times_orig, data_orig[0] * 1e6, color="#1f77b4", linewidth=1, alpha=0.5, label="Continuous signal")
    axes[0].scatter(times_orig, data_orig[0] * 1e6,
                   s=20, color="#1f77b4", marker="o", edgecolors="darkblue", linewidth=0.5, zorder=3, alpha=0.6)
    axes[0].set_ylabel("Amplitude (µV)", fontsize=11, fontweight="bold")
    axes[0].set_title(f"ORIGINAL — {raw_orig.info['sfreq']:.0f} Hz sampling rate\n{len(times_orig)} samples in {t_end:.1f}s = {raw_orig.info['sfreq']:.0f} samples/second", 
                     fontsize=12, fontweight="bold", color="#1f77b4", pad=10)
    axes[0].set_xlim([0, t_end])
    axes[0].grid(alpha=0.3, linestyle=":")
    axes[0].set_facecolor("#f0f8ff")
    axes[0].legend(loc="upper right", fontsize=10)

    # DOWNSAMPLED: Few samples — show ALL points
    axes[1].plot(times_downsampled, data_downsampled[0] * 1e6, color="#ff7f0e", linewidth=1, label="Downsampled", zorder=1)
    axes[1].scatter(times_downsampled, data_downsampled[0] * 1e6,
                   s=20, color="#ff7f0e", marker="o", edgecolors="darkred", linewidth=0.5, zorder=4, alpha=0.8)
    axes[1].set_xlabel("Time (s)", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("Amplitude (µV)", fontsize=11, fontweight="bold")
    axes[1].set_title(f"DOWNSAMPLED — {raw_downsampled.info['sfreq']:.0f} Hz sampling rate\n{len(times_downsampled)} samples in {t_end:.1f}s = {raw_downsampled.info['sfreq']:.0f} samples/second ({raw_orig.info['sfreq']/raw_downsampled.info['sfreq']:.0f}x reduction)", 
                     fontsize=12, fontweight="bold", color="#ff7f0e", pad=10)
    axes[1].set_xlim([0, t_end])
    axes[1].grid(alpha=0.3, linestyle=":")
    axes[1].set_facecolor("#fff5e6")
    axes[1].legend(loc="upper right", fontsize=10)

    fig.suptitle(f"sub-{subject_id} {ch_name} — Downsampling Effect: {raw_orig.info['sfreq']:.0f} Hz → {raw_downsampled.info['sfreq']:.0f} Hz\n(Top: {len(times_orig)} samples | Bottom: {len(times_downsampled)} samples — Clear reduction, signal preserved)", 
                fontsize=13, fontweight="bold", y=0.995)
    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / f"sub-{subject_id}_downsample_timeseries_comparison.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Time series comparison saved: {plot_path.name}")


def plot_psd_comparison(raw_orig, raw_downsampled, subject_id, duration, output_dir):
    """Plot PSD comparison before and after downsampling with visual distinction."""
    psd_duration = min(float(duration), 30.0, raw_orig.times[-1], raw_downsampled.times[-1])
    raw_orig_eeg = raw_orig.copy().pick_types(eeg=True).crop(tmin=0, tmax=psd_duration)
    raw_downsampled_eeg = raw_downsampled.copy().pick_types(eeg=True).crop(tmin=0, tmax=psd_duration)
    nyquist_orig = raw_orig.info["sfreq"] / 2
    nyquist_down = raw_downsampled.info["sfreq"] / 2
    shared_fmax = min(100, nyquist_down * 0.95)

    fig = plt.figure(figsize=(16, 6))
    
    # Original PSD (left)
    ax_orig = plt.subplot(1, 2, 1)
    raw_orig_eeg.plot_psd(fmax=shared_fmax, ax=ax_orig, show=False, color="#1f77b4")
    ax_orig.set_facecolor("#f0f8ff")
    ax_orig.set_title(f"ORIGINAL DATA — {raw_orig.info['sfreq']:.0f} Hz\nShown on same 0-{shared_fmax:.0f} Hz scale for comparison",
                     fontsize=12, fontweight="bold", color="#1f77b4")
    ax_orig.grid(alpha=0.4, linestyle=":")
    ax_orig.set_xlim(0, shared_fmax)

    # Nyquist line for original
    ax_orig.axvline(nyquist_orig, color="#1f77b4", linestyle="--", linewidth=2, alpha=0.5, label=f"Nyquist: {nyquist_orig:.0f} Hz")
    ax_orig.legend()

    # Downsampled PSD (right)
    ax_down = plt.subplot(1, 2, 2)
    raw_downsampled_eeg.plot_psd(fmax=shared_fmax, ax=ax_down, show=False, color="#ff7f0e")
    ax_down.set_facecolor("#fff5e6")
    ax_down.set_title(f"DOWNSAMPLED DATA — {raw_downsampled.info['sfreq']:.0f} Hz\nShown on same 0-{shared_fmax:.0f} Hz scale; cutoff at Nyquist ({nyquist_down:.0f} Hz)",
                     fontsize=12, fontweight="bold", color="#ff7f0e")
    ax_down.grid(alpha=0.4, linestyle=":")
    ax_down.set_xlim(0, shared_fmax)

    # Nyquist line for downsampled
    ax_down.axvline(nyquist_down, color="#ff7f0e", linestyle="--", linewidth=2, alpha=0.5, label=f"Nyquist: {nyquist_down:.0f} Hz")
    ax_down.legend()

    shared_ymin = min(ax_orig.get_ylim()[0], ax_down.get_ylim()[0])
    shared_ymax = max(ax_orig.get_ylim()[1], ax_down.get_ylim()[1])
    ax_orig.set_ylim(shared_ymin, shared_ymax)
    ax_down.set_ylim(shared_ymin, shared_ymax)

    fig.suptitle(f"sub-{subject_id} - Power Spectral Density: Before vs After Downsampling\n(Computed from first {psd_duration:.0f}s; higher frequencies disappear in downsampled version)", 
                fontsize=13, fontweight="bold")
    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / f"sub-{subject_id}_downsample_psd_comparison.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ PSD comparison saved: {plot_path.name}")


def plot_statistics_comparison(raw_orig, raw_downsampled, subject_id, output_dir):
    """Plot data quality statistics before and after downsampling with clear visual distinction."""
    # Get sample data
    eeg_picks = mne.pick_types(raw_orig.info, eeg=True)
    max_samples_orig = min(int(raw_orig.info["sfreq"] * 120), raw_orig.n_times)
    max_samples_downsampled = min(int(raw_downsampled.info["sfreq"] * 120), raw_downsampled.n_times)

    data_orig = raw_orig.get_data(picks=eeg_picks, start=0, stop=max_samples_orig)
    data_downsampled = raw_downsampled.get_data(picks=eeg_picks, start=0, stop=max_samples_downsampled)

    std_orig = np.std(data_orig, axis=1) * 1e6
    std_downsampled = np.std(data_downsampled, axis=1) * 1e6

    fig = plt.figure(figsize=(16, 5))

    # Std distribution (left)
    ax1 = plt.subplot(1, 3, 1)
    ax1.hist(std_orig, bins=20, alpha=0.6, label="Original", color="#1f77b4", edgecolor="black", linewidth=1.2)
    ax1.hist(std_downsampled, bins=20, alpha=0.6, label="Downsampled", color="#ff7f0e", edgecolor="black", linewidth=1.2)
    ax1.set_xlabel("Standard Deviation (µV)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Channel Count", fontsize=11, fontweight="bold")
    ax1.set_title("Amplitude Distribution\n(Should be nearly identical)", fontsize=11, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3, axis="y")

    # Per-channel comparison (center)
    ax2 = plt.subplot(1, 3, 2)
    channels = np.arange(len(std_orig))
    ax2.scatter(channels, std_orig, alpha=0.6, s=50, label="Original", color="#1f77b4", edgecolor="black")
    ax2.scatter(channels, std_downsampled, alpha=0.6, s=50, label="Downsampled", color="#ff7f0e", edgecolor="black", marker="^")
    # Connect dots to show correspondence
    for i in range(0, len(channels), 8):
        ax2.plot([i, i], [std_orig[i], std_downsampled[i]], "k--", alpha=0.2, linewidth=1)
    ax2.set_xlabel("Channel Index", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Std (µV)", fontsize=11, fontweight="bold")
    ax2.set_title("Per-Channel Amplitude\n(Patterns should match)", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3)

    # File size comparison (right)
    ax3 = plt.subplot(1, 3, 3)
    est_size_orig = (len(raw_orig.ch_names) * raw_orig.n_times * 8) / 1e6
    est_size_downsampled = (len(raw_downsampled.ch_names) * raw_downsampled.n_times * 8) / 1e6
    
    categories = ["Original", "Downsampled"]
    sizes = [est_size_orig, est_size_downsampled]
    colors_bar = ["#1f77b4", "#ff7f0e"]
    
    bars = ax3.bar(categories, sizes, color=colors_bar, alpha=0.7, edgecolor="black", linewidth=2, width=0.6)
    ax3.set_ylabel("Estimated Size (MB)", fontsize=11, fontweight="bold")
    reduction_pct = (1 - est_size_downsampled / est_size_orig) * 100
    ax3.set_title(f"Data Size Comparison\nReduction: {reduction_pct:.1f}%", fontsize=11, fontweight="bold")
    ax3.grid(axis="y", alpha=0.3)
    
    # Add size labels on bars
    for i, (bar, size) in enumerate(zip(bars, sizes)):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 2, f"{size:.1f} MB\n({int(height*1024)} MB)", 
                ha="center", va="bottom", fontweight="bold", fontsize=10)

    fig.suptitle(f"sub-{subject_id} - Downsampling Impact: Data Quality & Size\n(Amplitude preserved, size reduced by {reduction_pct:.0f}%)", 
                fontsize=13, fontweight="bold")
    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / f"sub-{subject_id}_downsample_statistics_comparison.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Statistics comparison saved: {plot_path.name}")


def main():
    args = parse_args()
    subjects = get_subjects(args.subjects)
    duration = args.duration
    output_dir = config.QC_DIR

    print("\n" + "=" * 80)
    print("VISUALIZATION: Step 00 - Downsample Effects")
    print("=" * 80)
    print(f"Subjects: {', '.join(subjects)}")
    print(f"Duration: {duration}s")
    print(f"Output: {output_dir}")
    print("=" * 80 + "\n")

    for subject_id in subjects:
        print(f"\n--- Subject {subject_id} ---")

        # Load original BIDS data
        try:
            bids_path = BIDSPath(
                subject=subject_id,
                task="RPS",
                datatype="eeg",
                suffix="eeg",
                root=config.BIDS_ROOT,
            )
            raw_original = read_raw_bids(bids_path, verbose=False)
        except Exception as e:
            print(f"  ERROR loading original BIDS data: {e}")
            continue

        # Load downsampled data
        downsampled_path = config.OUTPUT_DIR / f"sub-{subject_id}_downsampled.fif"
        if not downsampled_path.exists():
            print(f"  ERROR: Downsampled file not found: {downsampled_path}")
            continue

        try:
            raw_downsampled = mne.io.read_raw_fif(str(downsampled_path), preload=False, verbose=False)
        except Exception as e:
            print(f"  ERROR loading downsampled data: {e}")
            continue

        # Print info
        print(f"  Original sampling rate: {raw_original.info['sfreq']:.0f} Hz")
        print(f"  Downsampled sampling rate: {raw_downsampled.info['sfreq']:.0f} Hz")
        print(f"  Downsampling factor: {raw_original.info['sfreq'] / raw_downsampled.info['sfreq']:.1f}x")

        # Generate plots
        plot_timeseries_comparison(raw_original, raw_downsampled, subject_id, duration, output_dir)
        plot_psd_comparison(raw_original, raw_downsampled, subject_id, duration, output_dir)
        plot_statistics_comparison(raw_original, raw_downsampled, subject_id, output_dir)

    print("\n" + "=" * 80)
    print(f"✓ All visualizations saved to: {output_dir}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
