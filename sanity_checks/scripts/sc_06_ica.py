"""
Sanity Check Visualization for Step 06: ICA Component Removal

Creates verification plots for the ICA decomposition and bad component removal:
- Component topomaps showing spatial patterns of all ICA components
- Variance explained by each component (with bad components highlighted)
- Time-domain signatures of bad components (2-second samples)
- Before/after EEG PSD comparison to verify artifact removal
- Summary of removed components and their characteristics

Usage:
    python sanity_checks/scripts/sc_06_ica.py [--subjects 01,02] [--duration 30]

Options:
    --subjects: Comma-separated subject IDs (default: first 2)
    --duration: Duration in seconds used for PSD comparison (default: 30)
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
from mne.time_frequency import psd_array_welch

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

PIPELINE_DIR = CURRENT_DIR.parent.parent / "eeg_pipeline"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from preprocessing import config
from sc_utils import SanityCheckCollector, compare_amplitudes, detect_amplitude_anomaly

COLOR_BEFORE = "#c44e52"
COLOR_AFTER = "#2a7f62"
COLOR_MEAN = "#111111"
COLOR_BAD = "#e74c3c"
COLOR_GOOD = "#3498db"
COLOR_PASSBAND = "#d8ead3"
COLOR_STOPBAND = "#f6d6d6"
PSD_FMAX = 60.0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize ICA decomposition and component removal",
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
        help="Duration in seconds for PSD plots (default: 30)",
    )
    return parser.parse_args()


def _normalize_subject_id(subject_id):
    value = str(subject_id).strip()
    if not value:
        return value
    return value.zfill(2) if value.isdigit() else value


def get_subjects(subject_str):
    if subject_str:
        return [_normalize_subject_id(part) for part in subject_str.split(",") if part.strip()]
    return [_normalize_subject_id(subject_id) for subject_id in list(config.SUBJECTS)[:2]]


def _prepare_eeg_crop(raw, duration_s):
    eeg_picks = mne.pick_types(raw.info, eeg=True, exclude=[])
    if len(eeg_picks) == 0:
        return None

    t_end = min(float(duration_s), float(raw.times[-1]))
    if t_end <= 1.0:
        return None

    return raw.copy().pick(eeg_picks).crop(tmin=0.0, tmax=t_end)


def _compute_psd(raw, duration_s, fmax=PSD_FMAX):
    raw_eeg = _prepare_eeg_crop(raw, duration_s)
    if raw_eeg is None:
        return None, None, None, None

    data = raw_eeg.get_data()
    sfreq = float(raw_eeg.info["sfreq"])
    n_fft = min(int(round(sfreq * 4.0)), data.shape[1])
    if n_fft < 32:
        return None, None, None, None
    n_per_seg = min(n_fft, data.shape[1])
    n_overlap = n_per_seg // 2

    psd, freqs = psd_array_welch(
        data,
        sfreq=sfreq,
        fmin=0.0,
        fmax=fmax,
        n_fft=n_fft,
        n_per_seg=n_per_seg,
        n_overlap=n_overlap,
        average="mean",
        verbose=False,
    )
    return freqs, np.mean(psd, axis=0), psd, raw_eeg


def plot_component_topomaps(ica, raw_before, subject_id, person, output_dir):
    """Plot topographic maps of all ICA components with bad components highlighted."""
    if ica.n_components < 1:
        return
    
    # Organize components in a grid (4 components per row, up to 12 components shown)
    n_comps_show = min(ica.n_components, 12)
    n_rows = (n_comps_show + 3) // 4
    
    fig, axes = plt.subplots(n_rows, 4, figsize=(16, n_rows * 3.5), constrained_layout=True)
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    else:
        axes = axes.reshape(n_rows, -1)
    axes = axes.flatten()
    
    for idx in range(n_comps_show):
        ax = axes[idx]
        is_bad = idx in ica.exclude
        
        # Plot component topomap
        try:
            ica.plot_components(picks=[idx], show=False, axes=ax, sphere="eeglab")
        except Exception:
            try:
                ica.plot_components(picks=[idx], show=False, axes=ax, sphere="auto")
            except Exception:
                ax.text(0.5, 0.5, f"Component {idx}\n(plot failed)", ha="center", va="center", fontsize=10)
                ax.set_xlim(-0.1, 0.1)
                ax.set_ylim(-0.1, 0.1)
        
        # Highlight bad components with border
        title_color = COLOR_BAD if is_bad else COLOR_GOOD
        title_suffix = " (BAD)" if is_bad else ""
        title_weight = "bold" if is_bad else "normal"
        ax.set_title(f"Component {idx}{title_suffix}", fontsize=11, fontweight=title_weight, color=title_color)
        
        if is_bad:
            for spine in ax.spines.values():
                spine.set_edgecolor(COLOR_BAD)
                spine.set_linewidth(2.5)
    
    # Hide unused subplots
    for idx in range(n_comps_show, len(axes)):
        axes[idx].set_visible(False)
    
    fig.suptitle(
        f"sub-{subject_id} {person} - ICA Component Topomaps\n"
        f"Total: {ica.n_components} components | Bad (red): {len(ica.exclude)} | Good (blue): {ica.n_components - len(ica.exclude)}",
        fontsize=14,
        fontweight="bold",
    )
    
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / f"sub-{subject_id}_{person}_ica_components.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] Component topomaps saved: {plot_path.name}")


def plot_variance_explained(ica, raw_data, subject_id, person, output_dir):
    """Plot variance explained by each component, highlighting bad components."""
    if ica.n_components < 1:
        return
    
    # Compute explained variance - need to load a portion of raw data
    try:
        # Load a small portion of data for variance calculation (first 60 seconds)
        raw_crop = raw_data.copy().crop(tmin=0, tmax=min(60.0, raw_data.times[-1]))
        raw_crop.load_data()
        var_ratio_dict = ica.get_explained_variance_ratio(raw_crop)
        
        # Extract EEG channel variance (or combined if multiple)
        if isinstance(var_ratio_dict, dict):
            # Use EEG if available, otherwise use first available channel type
            if 'eeg' in var_ratio_dict:
                explained_var = var_ratio_dict['eeg']
            else:
                explained_var = list(var_ratio_dict.values())[0]
        else:
            explained_var = var_ratio_dict
    except Exception:
        # Fallback: estimate from component mixing matrix magnitude
        explained_var = np.abs(ica.mixing_matrix_).mean(axis=0)
        explained_var = explained_var / explained_var.sum()
    
    fig, ax = plt.subplots(figsize=(14, 6), constrained_layout=True)
    
    x = np.arange(ica.n_components)
    colors = [COLOR_BAD if idx in ica.exclude else COLOR_GOOD for idx in range(ica.n_components)]
    
    bars = ax.bar(x, explained_var * 100, color=colors, alpha=0.75, edgecolor="black", linewidth=1.0)
    
    ax.set_xlabel("Component Index", fontsize=12, fontweight="bold")
    ax.set_ylabel("Explained Variance (%)", fontsize=12, fontweight="bold")
    ax.set_title("Variance Explained by Each ICA Component", fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.28, linestyle=":")
    ax.set_ylim(bottom=0.0)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLOR_GOOD, edgecolor="black", label=f"Good ({ica.n_components - len(ica.exclude)})"),
        Patch(facecolor=COLOR_BAD, edgecolor="black", label=f"Bad ({len(ica.exclude)})"),
    ]
    ax.legend(handles=legend_elements, fontsize=10, loc="upper right")
    
    bad_variance = sum(explained_var[ica.exclude] * 100) if len(ica.exclude) > 0 else 0.0
    fig.suptitle(
        f"sub-{subject_id} {person} - ICA Variance Explained\n"
        f"Total variance: 100% | Bad components explain: {bad_variance:.1f}%",
        fontsize=13,
        fontweight="bold",
    )
    
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / f"sub-{subject_id}_{person}_ica_variance.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] Variance explained saved: {plot_path.name}")


def plot_bad_component_timeseries(ica, raw, subject_id, person, output_dir):
    """Plot time-domain signatures of bad ICA components."""
    if len(ica.exclude) == 0:
        return
    
    # Get ICA sources (component time series)
    sources = ica.get_sources(raw).get_data()
    sfreq = float(raw.info["sfreq"])
    times = raw.times
    
    # Take first 2 seconds
    t_end = min(2.0, float(times[-1]))
    n_samples = int(t_end * sfreq)
    
    n_bad = len(ica.exclude)
    n_cols = min(3, n_bad)
    n_rows = (n_bad + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4.5, n_rows * 3.2), constrained_layout=True)
    if n_bad == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    else:
        axes = axes.reshape(n_rows, -1)
    axes = axes.flatten()
    
    for plot_idx, comp_idx in enumerate(ica.exclude):
        ax = axes[plot_idx]
        component_data = sources[comp_idx, :n_samples]
        component_times = times[:n_samples]
        
        ax.plot(component_times, component_data, color=COLOR_BAD, linewidth=1.2)
        ax.axhline(0, color="#777777", linestyle="--", linewidth=0.8)
        ax.set_title(f"Component {comp_idx}", fontsize=11, fontweight="bold", color=COLOR_BAD)
        ax.set_ylabel("Amplitude (a.u.)")
        ax.grid(alpha=0.22, linestyle=":")
        
        if plot_idx >= (n_rows - 1) * n_cols:
            ax.set_xlabel("Time (s)")
    
    # Hide unused subplots
    for idx in range(n_bad, len(axes)):
        axes[idx].set_visible(False)
    
    fig.suptitle(
        f"sub-{subject_id} {person} - Bad ICA Component Time Series\n"
        f"First 2 seconds | {n_bad} component(s) marked for removal",
        fontsize=13,
        fontweight="bold",
    )
    
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / f"sub-{subject_id}_{person}_ica_bad_timeseries.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] Bad component time series saved: {plot_path.name}")


def plot_psd_comparison_ica(raw_before, raw_after, subject_id, person, duration, output_dir):
    """Plot PSD before and after ICA component removal."""
    freqs_before, mean_before, psd_before, raw_before_eeg = _compute_psd(raw_before, duration)
    freqs_after, mean_after, psd_after, _ = _compute_psd(raw_after, duration)
    if freqs_before is None or freqs_after is None or raw_before_eeg is None:
        return
    
    mean_before_db = 10.0 * np.log10(np.maximum(mean_before, np.finfo(float).tiny))
    mean_after_db = 10.0 * np.log10(np.maximum(mean_after, np.finfo(float).tiny))
    psd_before_db = 10.0 * np.log10(np.maximum(psd_before, np.finfo(float).tiny))
    psd_after_db = 10.0 * np.log10(np.maximum(psd_after, np.finfo(float).tiny))
    
    fig, (ax_before, ax_after) = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True, sharey=True)
    
    y_min = float(min(np.min(psd_before_db), np.min(psd_after_db))) - 2.0
    y_max = float(max(np.max(psd_before_db), np.max(psd_after_db))) + 2.0
    
    # Before ICA
    ax_before.axvspan(0.0, config.FREQ_LOWER, color=COLOR_STOPBAND, alpha=0.65, lw=0)
    ax_before.axvspan(config.FREQ_LOWER, config.FREQ_UPPER, color=COLOR_PASSBAND, alpha=0.8, lw=0)
    ax_before.axvspan(config.FREQ_UPPER, PSD_FMAX, color=COLOR_STOPBAND, alpha=0.65, lw=0)
    ax_before.plot(freqs_before, mean_before_db, color=COLOR_BEFORE, linewidth=2.5, label="Mean PSD")
    for ch_idx in range(psd_before_db.shape[0]):
        ax_before.plot(freqs_before, psd_before_db[ch_idx], color=COLOR_BEFORE, linewidth=0.7, alpha=0.15)
    ax_before.set_xlim(0.0, PSD_FMAX)
    ax_before.set_ylim(y_min, y_max)
    ax_before.set_title("Before ICA", fontsize=12, fontweight="bold")
    ax_before.set_xlabel("Frequency (Hz)")
    ax_before.set_ylabel("PSD per channel (dB)")
    ax_before.grid(alpha=0.22, linestyle=":")
    ax_before.legend(fontsize=10)
    
    # After ICA
    ax_after.axvspan(0.0, config.FREQ_LOWER, color=COLOR_STOPBAND, alpha=0.65, lw=0)
    ax_after.axvspan(config.FREQ_LOWER, config.FREQ_UPPER, color=COLOR_PASSBAND, alpha=0.8, lw=0)
    ax_after.axvspan(config.FREQ_UPPER, PSD_FMAX, color=COLOR_STOPBAND, alpha=0.65, lw=0)
    ax_after.plot(freqs_after, mean_after_db, color=COLOR_AFTER, linewidth=2.5, label="Mean PSD")
    for ch_idx in range(psd_after_db.shape[0]):
        ax_after.plot(freqs_after, psd_after_db[ch_idx], color=COLOR_AFTER, linewidth=0.7, alpha=0.15)
    ax_after.set_xlim(0.0, PSD_FMAX)
    ax_after.set_title("After ICA", fontsize=12, fontweight="bold")
    ax_after.set_xlabel("Frequency (Hz)")
    ax_after.grid(alpha=0.22, linestyle=":")
    ax_after.legend(fontsize=10)
    
    # Compute passband change
    passband_before = np.mean(psd_before[:, (freqs_before >= config.FREQ_LOWER) & (freqs_before <= config.FREQ_UPPER)])
    passband_after = np.mean(psd_after[:, (freqs_after >= config.FREQ_LOWER) & (freqs_after <= config.FREQ_UPPER)])
    passband_change = ((passband_after - passband_before) / passband_before * 100.0) if passband_before > 0 else np.nan
    
    fig.suptitle(
        f"sub-{subject_id} {person} - ICA PSD Comparison\n"
        f"Passband (1-40 Hz) change: {passband_change:+.1f}%",
        fontsize=13,
        fontweight="bold",
    )
    
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / f"sub-{subject_id}_{person}_ica_psd_comparison.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] PSD comparison saved: {plot_path.name}")


def sanity_check_ica(subjects, duration):
    collector = SanityCheckCollector("06 - ICA Component Removal")

    print("\n" + "=" * 80)
    print("VISUALIZATION: Step 06 - ICA Decomposition Verification")
    print("=" * 80)
    print(f"Subjects: {', '.join(subjects)}")
    print(f"Duration: {duration}s")
    print(f"Output: {config.QC_DIR}")
    print("=" * 80 + "\n")

    for subject_id in subjects:
        print(f"\n--- Subject {subject_id} ---")
        for person in ["P1", "P2"]:
            before_path = config.OUTPUT_DIR / f"sub-{subject_id}_{person}_filtered.fif"
            after_path = config.OUTPUT_DIR / f"sub-{subject_id}_{person}_ica_cleaned.fif"
            ica_path = config.ICA_DIR / f"sub-{subject_id}_{person}_ica.fif"

            if not before_path.exists():
                collector.add_result(subject_id, person, "ERROR", "Input file (filtered) not found")
                continue
            if not after_path.exists():
                collector.add_result(subject_id, person, "ERROR", "Output file (ICA) not found")
                continue
            if not ica_path.exists():
                collector.add_result(subject_id, person, "ERROR", "ICA decomposition file not found")
                continue

            try:
                raw_before = mne.io.read_raw_fif(str(before_path), preload=False, verbose=False)
                raw_after = mne.io.read_raw_fif(str(after_path), preload=False, verbose=False)
                ica = mne.preprocessing.read_ica(str(ica_path))
            except Exception as exc:
                collector.add_result(subject_id, person, "ERROR", f"Cannot load files: {exc}")
                continue

            collector.add_result(subject_id, person, "✓", "Files exist")
            
            # Verify channel count
            if len(raw_before.ch_names) == len(raw_after.ch_names):
                collector.add_result(subject_id, person, "✓", f"Channel count preserved: {len(raw_after.ch_names)}")
            else:
                collector.add_result(subject_id, person, "ERROR", f"Channel count mismatch: {len(raw_before.ch_names)} -> {len(raw_after.ch_names)}")
            
            # Verify sampling rate
            if raw_before.info["sfreq"] == raw_after.info["sfreq"]:
                collector.add_result(subject_id, person, "✓", f"Sampling rate preserved: {raw_after.info['sfreq']} Hz")
            else:
                collector.add_result(subject_id, person, "ERROR", f"Sampling rate changed: {raw_before.info['sfreq']} -> {raw_after.info['sfreq']}")
            
            # Verify sample count
            if raw_before.n_times == raw_after.n_times:
                collector.add_result(subject_id, person, "✓", f"Sample count preserved: {raw_after.n_times}")
            else:
                collector.add_result(subject_id, person, "ERROR", f"Sample count changed: {raw_before.n_times} -> {raw_after.n_times}")
            
            # Check ICA components
            if ica.n_components > 0:
                collector.add_result(subject_id, person, "✓", f"ICA fitted with {ica.n_components} components")
            else:
                collector.add_result(subject_id, person, "ERROR", "ICA has zero components")
            
            # Check for removed components
            if len(ica.exclude) > 0:
                collector.add_result(subject_id, person, "✓", f"Removed {len(ica.exclude)}/{ica.n_components} components: {ica.exclude}")
            else:
                collector.add_result(subject_id, person, "⚠", "No components marked for removal")
            
            # Compare amplitudes
            std_before, std_after, change_pct = compare_amplitudes(raw_before, raw_after, duration_s=60, pick_type="eeg")
            if not (np.isnan(std_before) or np.isnan(std_after)):
                collector.add_result(subject_id, person, "✓", f"EEG amplitude: {std_before:.2f} µV -> {std_after:.2f} µV ({change_pct:+.1f}%)")
                if change_pct > 20.0:
                    collector.add_result(subject_id, person, "⚠", "Larger than expected amplitude increase after ICA")
            
            # Check for NaN/Inf
            data_after = raw_after.get_data(start=0, stop=min(10000, raw_after.n_times))
            nan_count = int(np.isnan(data_after).sum())
            inf_count = int(np.isinf(data_after).sum())
            if nan_count == 0 and inf_count == 0:
                collector.add_result(subject_id, person, "✓", "No NaN/Inf detected")
            else:
                collector.add_result(subject_id, person, "ERROR", f"Found {nan_count} NaN and {inf_count} Inf values")

            print(f"\n  {person}:")
            plot_component_topomaps(ica, raw_before, subject_id, person, config.QC_DIR)
            plot_variance_explained(ica, raw_before, subject_id, person, config.QC_DIR)
            plot_bad_component_timeseries(ica, raw_before, subject_id, person, config.QC_DIR)
            plot_psd_comparison_ica(raw_before, raw_after, subject_id, person, duration, config.QC_DIR)

    collector.print_summary()
    output_csv = config.QC_DIR / "sc_06_ica_summary.csv"
    collector.export_csv(output_csv)
    print(f"\n[OK] Summary exported to {output_csv.name}\n")


def main():
    args = parse_args()
    subjects = get_subjects(args.subjects)
    sanity_check_ica(subjects, args.duration)


if __name__ == "__main__":
    main()

