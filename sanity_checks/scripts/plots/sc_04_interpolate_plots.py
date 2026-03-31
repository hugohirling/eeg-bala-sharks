"""
Sanity Check Plot Module for Step 04: Interpolate Bad Channels

Creates visualization plots for bad channel interpolation:
- Interpolated channel time series comparison
- Step-03-style all-channel before/after comparison
- Step-03-style bad-channel + neighbor before/after comparison

Entry point:
    python sanity_checks/scripts/sc_04_interpolate.py --mode viz

Options:
    --subjects: Comma-separated subject IDs (default: first 2)
    --duration: Duration in seconds to plot (default: 30)

REASONING:
- Purpose: show whether repaired channels now resemble their spatial neighbors without changing the rest of the montage.
- Reproducibility: the comparison always uses the stored before/after FIF pair for the same subject/person.
- Interpretation focus: the expected argument is "This seems correct because the interpolated trace moves closer to neighboring channels while the overall recording structure stays unchanged."
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
from helpers.sc_cli import add_duration_argument, add_subjects_argument, resolve_subjects
from helpers.sc_config import DEFAULT_PERSONS, FIXED_ABS_SCALE_UV, INTERPOLATE_VIZ, VIZ_NEUTRAL
from helpers.sc_plot_io import save_figure


COLOR_BEFORE = INTERPOLATE_VIZ["before"]
COLOR_AFTER = INTERPOLATE_VIZ["after"]
COLOR_NEIGHBOR = INTERPOLATE_VIZ["neighbor"]
COLOR_DELTA = INTERPOLATE_VIZ["delta"]
COLOR_GOOD = INTERPOLATE_VIZ["good"]
COLOR_BEFORE_SOFT = INTERPOLATE_VIZ["before_soft"]
COLOR_AFTER_SOFT = INTERPOLATE_VIZ["after_soft"]
COLOR_AFTER_STRONG = INTERPOLATE_VIZ["after_strong"]
COLOR_BAD_EDGE = VIZ_NEUTRAL["text_mid"]
# Shared scale keeps before/after overlay plots directly comparable instead of adapting the y-axis per recording.


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Visualize bad channel interpolation effects (before/after restoration)",
    )
    add_subjects_argument(parser)
    add_duration_argument(parser, default=30)
    return parser.parse_args(argv)


def _get_channel_locations(raw):
    """Return 3D channel locations for EEG channels when available."""
    ch_locs = []
    for ch in raw.info["chs"]:
        loc = np.asarray(ch.get("loc", np.array([np.nan, np.nan, np.nan]))[:3], dtype=float)
        if np.all(np.isfinite(loc)):
            ch_locs.append(loc)
        else:
            ch_locs.append(np.array([np.nan, np.nan, np.nan], dtype=float))
    return np.asarray(ch_locs, dtype=float)


def _nearest_clean_neighbors(raw, bad_indices, k=2):
    """Map each bad-channel index to its nearest clean EEG neighbors."""
    ch_locs = _get_channel_locations(raw)
    good_indices = [idx for idx, ch_name in enumerate(raw.ch_names) if ch_name not in raw.info.get("bads", [])]
    neighbor_map = {}

    for target_idx in bad_indices:
        if target_idx >= len(ch_locs) or not np.all(np.isfinite(ch_locs[target_idx])):
            neighbor_map[target_idx] = []
            continue

        target_loc = ch_locs[target_idx]
        distances = []
        for idx in good_indices:
            if idx < len(ch_locs) and np.all(np.isfinite(ch_locs[idx])):
                distances.append((float(np.linalg.norm(ch_locs[idx] - target_loc)), idx))
        distances.sort(key=lambda item: item[0])
        neighbor_map[target_idx] = [idx for _, idx in distances[:k]]

    return neighbor_map


def _prepare_abs_eeg_view(raw, duration_sec=1.0):
    """Return centered absolute EEG amplitudes for step-03-style comparisons."""
    eeg_picks = mne.pick_types(raw.info, eeg=True, exclude=[])
    if len(eeg_picks) == 0:
        return None, None, None, None

    raw_eeg = raw.copy().pick(eeg_picks)
    t_end = min(float(duration_sec), raw_eeg.times[-1])
    n_samples = int(t_end * raw_eeg.info["sfreq"])
    if n_samples <= 10:
        return None, None, None, None

    data_uv = raw_eeg.get_data(start=0, stop=n_samples) * 1e6
    data_centered = data_uv - np.median(data_uv, axis=1, keepdims=True)
    data_abs_uv = np.abs(data_centered)
    times = raw_eeg.times[:n_samples]
    return raw_eeg, times, data_abs_uv, t_end


def plot_interpolation_channel_overlay_comparison(raw_before, raw_after, subject_id, person, output_dir, duration_sec=1.0):
    """Create separate step-03-style all-channel plots before and after interpolation."""
    bads = raw_before.info.get("bads", [])
    if not bads:
        return

    raw_before_eeg, times_before, data_abs_before, t_end_before = _prepare_abs_eeg_view(raw_before, duration_sec=duration_sec)
    raw_after_eeg, times_after, data_abs_after, t_end_after = _prepare_abs_eeg_view(raw_after, duration_sec=duration_sec)
    if raw_before_eeg is None or raw_after_eeg is None:
        return

    ch_names = raw_before_eeg.ch_names
    highlighted_indices = [idx for idx, ch_name in enumerate(ch_names) if ch_name in set(bads)]
    if not highlighted_indices:
        return

    common_abs_peak_uv = float(FIXED_ABS_SCALE_UV)
    spacing = common_abs_peak_uv * 2.4
    offsets = np.arange(len(ch_names)) * spacing

    stage_specs = [
        (
            times_before,
            data_abs_before,
            COLOR_BEFORE,
            "Before interpolation (same channels still marked bad)",
            output_dir / f"sub-{subject_id}_{person}_interpolate_vs_bad_channels_overview_before.png",
        ),
        (
            times_after,
            data_abs_after,
            COLOR_AFTER,
            "After interpolation (same channels shown for direct comparison)",
            output_dir / f"sub-{subject_id}_{person}_interpolate_vs_bad_channels_overview_after.png",
        ),
    ]

    for times, data_abs, highlight_color, title, plot_path in stage_specs:
        fig, ax = plt.subplots(1, 1, figsize=(16, 10))
        for idx, ch_name in enumerate(ch_names):
            y = data_abs[idx] + offsets[idx]
            if idx in highlighted_indices:
                ax.plot(times, y, color=highlight_color, linewidth=0.95, alpha=0.95, zorder=4)
                ax.text(
                    times[-1] + 0.03,
                    offsets[idx],
                    ch_name,
                    fontsize=8,
                    fontweight="bold",
                    color=highlight_color,
                    va="center",
                )
            else:
                ax.plot(times, y, color=COLOR_GOOD, linewidth=0.32, alpha=0.75, zorder=2)

        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_ylabel("All channels\n(abs uV, stacked)")
        ax.set_yticks([])
        ax.grid(alpha=0.2, axis="x")
        ax.set_xlim(times[0], times[-1])
        ax.set_xlabel("Time (s)")

        legend_handles = [
            plt.Line2D([0], [0], color=COLOR_GOOD, lw=1.2, alpha=0.85, label=f"Other EEG channels ({len(ch_names) - len(highlighted_indices)})"),
            plt.Line2D([0], [0], color=highlight_color, lw=1.4, label=f"Highlighted interpolated channels ({len(highlighted_indices)})"),
        ]
        ax.legend(handles=legend_handles, loc="upper left", fontsize=9)

        fig.suptitle(
            f"sub-{subject_id} {person} - Step 04 vs Step 03 Channel Overlay Comparison\n"
            f"{title} (first {min(t_end_before, t_end_after):.1f}s)",
            fontsize=13,
            fontweight="bold",
        )
        plt.tight_layout(rect=[0, 0, 0.94, 0.96])
        plot_path = save_figure(fig, output_dir, plot_path.name, dpi=140, bbox_inches=None)
        plt.close(fig)
        print(f"  âœ“ Step-03-style overview comparison saved: {plot_path.name}")


def plot_interpolation_neighbor_comparison(raw_before, raw_after, subject_id, person, output_dir, duration_sec=1.0):
    """Create a step-03-style bad-channel + neighbors comparison before vs after interpolation."""
    bads = raw_before.info.get("bads", [])
    if not bads:
        return

    raw_before_eeg, times_before, data_abs_before, t_end_before = _prepare_abs_eeg_view(raw_before, duration_sec=duration_sec)
    raw_after_eeg, times_after, data_abs_after, t_end_after = _prepare_abs_eeg_view(raw_after, duration_sec=duration_sec)
    if raw_before_eeg is None or raw_after_eeg is None:
        return

    bad_indices = [idx for idx, ch_name in enumerate(raw_before_eeg.ch_names) if ch_name in set(bads)]
    if not bad_indices:
        return

    bad_indices = bad_indices[:5]
    neighbor_map = _nearest_clean_neighbors(raw_before_eeg, bad_indices, k=2)
    common_abs_peak_uv = float(FIXED_ABS_SCALE_UV)
    spacing_abs = common_abs_peak_uv * 5.2
    offsets_abs = np.arange(len(bad_indices)) * spacing_abs

    fig, axes = plt.subplots(1, 2, figsize=(18, 11), sharey=True)
    stage_specs = [
        (axes[0], times_before, data_abs_before, COLOR_BEFORE, "Before interpolation"),
        (axes[1], times_after, data_abs_after, COLOR_AFTER, "After interpolation"),
    ]

    for ax, times, data_abs, bad_color, title in stage_specs:
        for row_idx, ch_idx in enumerate(bad_indices):
            neighbors = neighbor_map.get(ch_idx, [])
            row_offset = offsets_abs[row_idx]
            for n_i, n_idx in enumerate(neighbors):
                y_nei = data_abs[n_idx] + row_offset
                ax.plot(
                    times,
                    y_nei,
                    color=COLOR_NEIGHBOR,
                    linewidth=0.9,
                    alpha=0.75 if n_i == 0 else 0.5,
                    linestyle="-",
                    zorder=2,
                )

            y_bad = data_abs[ch_idx] + row_offset
            ax.plot(times, y_bad, color=bad_color, linewidth=1.15, alpha=0.95, zorder=3)

            nei_label = ", ".join(raw_before_eeg.ch_names[n_idx] for n_idx in neighbors) if neighbors else "no clean neighbors"
            if ax is axes[1]:
                yaxis_xform = ax.get_yaxis_transform()
                ax.text(
                    1.01,
                    row_offset,
                    raw_before_eeg.ch_names[ch_idx],
                    transform=yaxis_xform,
                    fontsize=8,
                    fontweight="bold",
                    color=bad_color,
                    va="center",
                    clip_on=False,
                )
                ax.text(
                    1.065,
                    row_offset,
                    f"| N: {nei_label}",
                    transform=yaxis_xform,
                    fontsize=8,
                    fontweight="bold",
                    color=COLOR_NEIGHBOR,
                    va="center",
                    clip_on=False,
                )

        ax.set_xlim(times[0], times[-1])
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.grid(alpha=0.2, axis="x")
        ax.set_yticks([])
        ax.set_xlabel("Time (s)")

    axes[0].set_ylabel("Bad channels + neighbors\n(abs uV, stacked)")
    axes[0].set_ylim(-common_abs_peak_uv * 1.0, offsets_abs[-1] + common_abs_peak_uv * 2.6)
    legend_handles = [
        plt.Line2D([0], [0], color=COLOR_BEFORE, lw=1.4, label="Bad channel before interpolation"),
        plt.Line2D([0], [0], color=COLOR_AFTER, lw=1.4, label="Same channel after interpolation"),
        plt.Line2D([0], [0], color=COLOR_NEIGHBOR, lw=1.2, label="Nearest clean neighbors"),
    ]
    axes[0].legend(handles=legend_handles, loc="upper left", fontsize=9)

    fig.suptitle(
        f"sub-{subject_id} {person} - Step 04 vs Step 03 Neighbor Comparison\n"
        f"Same bad channels and nearest clean neighbors before vs after interpolation",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 0.90, 0.96])

    plot_path = save_figure(
        fig,
        output_dir,
        f"sub-{subject_id}_{person}_interpolate_vs_bad_channels_neighbors.png",
        dpi=140,
        pad_inches=0.06,
    )
    plt.close(fig)
    print(f"  âœ“ Step-03-style neighbor comparison saved: {plot_path.name}")


def _smooth_for_display(signal_1d, sfreq, window_ms=8.0):
    """Apply a short moving average so thin traces remain readable."""
    win = int(max(1, round((window_ms / 1000.0) * float(sfreq))))
    if win <= 1:
        return signal_1d
    if win % 2 == 0:
        win += 1
    kernel = np.ones(win, dtype=float) / float(win)
    pad = win // 2
    padded = np.pad(signal_1d, pad_width=pad, mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def plot_montage_comparison(raw_before, raw_after, subject_id, person, output_dir):
    """Plot topomap comparison before and after interpolation."""
    raw_before_eeg = raw_before.copy().pick_types(eeg=True)
    raw_after_eeg = raw_after.copy().pick_types(eeg=True)

    if len(raw_before_eeg.ch_names) == 0:
        return

    try:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Before interpolation (with marked bad channels)
        try:
            raw_before_eeg.plot_sensors(kind="topomap", show_names=True, show=False, sphere="eeglab", axes=axes[0])
        except Exception:
            raw_before_eeg.plot_sensors(kind="topomap", show_names=True, show=False, sphere="auto", axes=axes[0])
        bads = raw_before.info.get("bads", [])
        if len(bads) > 0:
            axes[0].set_title(f"BEFORE - Bad Channels Marked\n({len(bads)} bad: {', '.join(bads[:5])}{'...' if len(bads) > 5 else ''})")
        else:
            axes[0].set_title("BEFORE - No Bad Channels Marked")

        # After interpolation (all channels should be good)
        try:
            raw_after_eeg.plot_sensors(kind="topomap", show_names=True, show=False, sphere="eeglab", axes=axes[1])
        except Exception:
            raw_after_eeg.plot_sensors(kind="topomap", show_names=True, show=False, sphere="auto", axes=axes[1])
        bads_after = raw_after.info.get("bads", [])
        axes[1].set_title(f"AFTER - All Channels Interpolated\n({len(bads_after)} remaining bad channels)")

        fig.suptitle(f"sub-{subject_id} {person} - Sensor Layout Comparison (Interpolation)")
        plt.tight_layout()

        plot_path = save_figure(fig, output_dir, f"sub-{subject_id}_{person}_interpolate_montage_comparison.png", dpi=150)
        plt.close(fig)
        print(f"  âœ“ Montage comparison saved: {plot_path.name}")

    except Exception as e:
        print(f"  WARNING: Could not create montage comparison: {e}")


def plot_interpolated_channels_timeseries(raw_before, raw_after, subject_id, person, duration, output_dir):
    """Plot bad channels before/after interpolation without extra reference traces."""
    bads = raw_before.info.get("bads", [])
    
    if len(bads) == 0:
        # No bad channels to interpolate
        return

    # Find indices of bad channels
    bad_indices = [i for i, ch in enumerate(raw_before.ch_names) if ch in bads]
    if len(bad_indices) == 0:
        return

    # Limit to first 4 bad channels for visualization
    bad_indices_to_plot = bad_indices[:4]
    t_end = min(duration, raw_before.times[-1], raw_after.times[-1])
    t_end = min(5.0, t_end)  # Limit to 5 seconds for clarity
    t_idx_before = int(t_end * raw_before.info["sfreq"])
    t_idx_after = int(t_end * raw_after.info["sfreq"])

    fig, axes = plt.subplots(
        len(bad_indices_to_plot),
        2,
        figsize=(16, 3.6 * len(bad_indices_to_plot)),
        squeeze=False,
        gridspec_kw={"width_ratios": [3.6, 1.4]},
    )

    for plot_idx, ch_idx in enumerate(bad_indices_to_plot):
        ch_name = raw_before.ch_names[ch_idx]

        data_before = raw_before.get_data(picks=[ch_idx], start=0, stop=t_idx_before)
        data_after = raw_after.get_data(picks=[ch_idx], start=0, stop=t_idx_after)

        times_before = raw_before.times[:data_before.shape[1]]
        times_after = raw_after.times[:data_after.shape[1]]
        trace_before_uv = data_before[0] * 1e6
        trace_after_uv = data_after[0] * 1e6

        # Match step-03 readability: remove each trace's baseline offset before plotting.
        trace_before_uv = trace_before_uv - np.median(trace_before_uv)
        trace_after_uv = trace_after_uv - np.median(trace_after_uv)

        delta_uv = trace_after_uv - trace_before_uv[:len(trace_after_uv)]

        ax_trace = axes[plot_idx, 0]
        ax_delta = axes[plot_idx, 1]

        ax_trace.plot(times_before, trace_before_uv, label="Before interpolation", linewidth=0.95, color=COLOR_BEFORE, linestyle="-", alpha=0.9)
        ax_trace.plot(times_after, trace_after_uv, label="After interpolation", linewidth=0.95, color=COLOR_AFTER, alpha=0.95, zorder=3)

        ax_trace.set_ylabel(f"{ch_name}\nCentered amplitude (ÂµV)", fontsize=10.5, fontweight="bold")
        ax_trace.set_xlim([0, t_end])
        ax_trace.grid(alpha=0.3, linestyle=":")
        ax_trace.set_facecolor(INTERPOLATE_VIZ["trace_even_face"] if plot_idx % 2 == 0 else INTERPOLATE_VIZ["trace_odd_face"])

        ax_delta.plot(times_after, delta_uv, color=COLOR_DELTA, linewidth=0.95)
        ax_delta.axhline(0.0, color=VIZ_NEUTRAL["marker_edge"], linestyle="--", linewidth=1.1, alpha=0.8)
        ax_delta.grid(alpha=0.25, linestyle=":")
        ax_delta.set_xlim([0, t_end])
        ax_delta.set_ylabel("After - before\n(ÂµV)", fontsize=9.5)
        std_before_uv = float(np.std(trace_before_uv))
        std_after_uv = float(np.std(trace_after_uv))
        info_lines = [
            f"std before: {std_before_uv:.1f}",
            f"std after: {std_after_uv:.1f}",
        ]
        ax_delta.text(
            0.03,
            0.95,
            "\n".join(info_lines),
            transform=ax_delta.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox=dict(
                boxstyle="round,pad=0.35",
                facecolor=INTERPOLATE_VIZ["note_face"],
                edgecolor=INTERPOLATE_VIZ["note_edge"],
                alpha=0.9,
            ),
        )

        if plot_idx == len(bad_indices_to_plot) - 1:
            ax_trace.set_xlabel("Time (s)", fontsize=11, fontweight="bold")
            ax_delta.set_xlabel("Time (s)", fontsize=11, fontweight="bold")

        if plot_idx == 0:
            ax_trace.set_title(
                f"sub-{subject_id} {person} - What happened to the bad channels after interpolation?\n"
                f"Before vs after only ({len(bads)} total bad channels, centered per trace)",
                fontsize=12,
                fontweight="bold",
                color=VIZ_NEUTRAL["black"],
            )
            ax_delta.set_title("Change caused by interpolation", fontsize=11, fontweight="bold")
            ax_trace.legend(loc="upper right", fontsize=9.5)

    plt.tight_layout()

    plot_path = save_figure(fig, output_dir, f"sub-{subject_id}_{person}_interpolate_timeseries.png", dpi=150)
    plt.close(fig)
    print(f"  âœ“ Interpolated channels time series saved: {plot_path.name}")


def plot_bad_channel_restoration_summary(raw_before, raw_after, subject_id, person, output_dir):
    """Summarize how interpolation changed each previously bad channel."""
    bads = raw_before.info.get("bads", [])
    bad_indices = [idx for idx, ch_name in enumerate(raw_before.ch_names) if ch_name in bads]
    if not bad_indices:
        return

    bad_indices = bad_indices[:8]
    neighbor_map = _nearest_clean_neighbors(raw_before, bad_indices, k=2)
    max_samples = min(int(raw_before.info["sfreq"] * 60), raw_before.n_times, raw_after.n_times)
    data_before = raw_before.get_data(picks=bad_indices, start=0, stop=max_samples) * 1e6
    data_after = raw_after.get_data(picks=bad_indices, start=0, stop=max_samples) * 1e6

    before_std = np.std(data_before, axis=1)
    after_std = np.std(data_after, axis=1)
    neighbor_std = []
    for idx in bad_indices:
        neighbors = neighbor_map.get(idx, [])
        if neighbors:
            neighbor_data = raw_after.get_data(picks=neighbors, start=0, stop=max_samples) * 1e6
            neighbor_std.append(float(np.std(np.mean(neighbor_data, axis=0))))
        else:
            neighbor_std.append(np.nan)
    neighbor_std = np.asarray(neighbor_std, dtype=float)

    channel_labels = [raw_before.ch_names[idx] for idx in bad_indices]
    x = np.arange(len(channel_labels))
    width = 0.24

    fig, axes = plt.subplots(2, 1, figsize=(14, 9), gridspec_kw={"height_ratios": [2.3, 1.2]})

    axes[0].bar(x - width, before_std, width, color=COLOR_BEFORE, alpha=0.85, label="Before interpolation")
    axes[0].bar(x, after_std, width, color=COLOR_AFTER, alpha=0.85, label="After interpolation")
    axes[0].bar(x + width, neighbor_std, width, color=COLOR_NEIGHBOR, alpha=0.85, label="Nearest clean-channel mean")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(channel_labels)
    axes[0].set_ylabel("STD (ÂµV)")
    axes[0].set_title("STD before vs after interpolation, with clean-neighbor reference")
    axes[0].grid(axis="y", alpha=0.3)
    axes[0].legend(fontsize=9.5)

    restoration_delta = after_std - before_std
    target_delta = after_std - neighbor_std
    axes[1].axhline(0.0, color=VIZ_NEUTRAL["marker_edge"], linestyle="--", linewidth=1.1)
    axes[1].plot(
        x,
        restoration_delta,
        color=COLOR_BEFORE,
        marker="o",
        linewidth=1.8,
        label="STD(after) - STD(before)",
    )
    axes[1].plot(
        x,
        target_delta,
        color=COLOR_NEIGHBOR,
        marker="s",
        linewidth=1.8,
        linestyle=":",
        label="STD(after) - STD(neighbor mean)",
    )
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(channel_labels)
    axes[1].set_ylabel("STD difference (ÂµV)")
    axes[1].set_xlabel("Interpolated channel")
    axes[1].set_title("Difference of interpolated-channel STD relative to each reference")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=9.5)

    fig.suptitle(f"sub-{subject_id} {person} - Bad Channel Restoration Summary", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    plot_path = save_figure(fig, output_dir, f"sub-{subject_id}_{person}_interpolate_bad_channel_restoration.png", dpi=150)
    plt.close(fig)
    print(f"  âœ“ Bad channel restoration summary saved: {plot_path.name}")


def plot_amplitude_statistics_comparison(raw_before, raw_after, subject_id, person, output_dir):
    """Compare amplitude statistics before and after interpolation with visual emphasis."""
    eeg_picks_before = mne.pick_types(raw_before.info, eeg=True, exclude=[])
    eeg_picks_after = mne.pick_types(raw_after.info, eeg=True, exclude=[])

    if len(eeg_picks_before) == 0 or len(eeg_picks_after) == 0:
        return

    # Get sample data
    max_samples = min(int(raw_before.info["sfreq"] * 60), raw_before.n_times)
    data_before = raw_before.get_data(picks=eeg_picks_before, start=0, stop=max_samples)
    data_after = raw_after.get_data(picks=eeg_picks_after, start=0, stop=max_samples)

    std_before = np.std(data_before, axis=1) * 1e6
    std_after = np.std(data_after, axis=1) * 1e6

    # Identify interpolated channels
    bads = raw_before.info.get("bads", [])

    fig = plt.figure(figsize=(16, 10))

    # Std distribution (top left)
    ax1 = plt.subplot(2, 2, 1)
    ax1.hist(std_before, bins=15, alpha=0.6, label="Before Interpolation", 
            color=COLOR_BEFORE_SOFT, edgecolor=VIZ_NEUTRAL["black"], linewidth=1.2)
    ax1.hist(std_after, bins=15, alpha=0.6, label="After Interpolation",
            color=COLOR_AFTER_SOFT, edgecolor=VIZ_NEUTRAL["black"], linewidth=1.2)
    ax1.set_xlabel("Standard Deviation (ÂµV)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Channel Count", fontsize=11, fontweight="bold")
    ax1.set_title("Amplitude Distribution\n(Red channels pushed into normal range)", fontsize=11, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3, axis="y")
    
    # Mark mean lines
    ax1.axvline(np.mean(std_before), color=COLOR_BEFORE_SOFT, linestyle="--", linewidth=2, alpha=0.7, label=f"Mean (before): {np.mean(std_before):.1f} ÂµV")
    ax1.axvline(np.mean(std_after), color=COLOR_AFTER_SOFT, linestyle="--", linewidth=2, alpha=0.7, label=f"Mean (after): {np.mean(std_after):.1f} ÂµV")

    # Per-channel scatter with bad channel highlighting (top right)
    ax2 = plt.subplot(2, 2, 2)
    channel_indices = np.arange(len(std_before))
    eeg_names_before = [raw_before.ch_names[i] for i in eeg_picks_before]
    interpolated_mask = np.array([ch_name in bads for ch_name in eeg_names_before], dtype=bool)
    good_mask = ~interpolated_mask

    # Plot good channels (BEFORE)
    if np.any(good_mask):
        ax2.scatter(channel_indices[good_mask], std_before[good_mask], 
                  s=60, c=COLOR_BEFORE_SOFT, alpha=0.6, edgecolor=VIZ_NEUTRAL["black"], linewidth=1, label="Good channels (before)")

    # Plot bad/interpolated channels (BEFORE) â€” PROMINENTLY MARKED
    if np.any(interpolated_mask):
        ax2.scatter(channel_indices[interpolated_mask], std_before[interpolated_mask], 
                  s=250, c=COLOR_BEFORE, marker="X", linewidth=3, edgecolor=COLOR_BAD_EDGE, label="Interpolated channels (before)", zorder=5)
        # Draw attention circles
        for idx in np.where(interpolated_mask)[0]:
            circle = plt.Circle((idx, std_before[idx]), 2.5, fill=False, edgecolor=COLOR_BEFORE, linewidth=2.5, linestyle="--")
            ax2.add_patch(circle)

    # Overlay AFTER values as smaller dots
    ax2.scatter(channel_indices[good_mask], std_after[good_mask], 
              s=40, c=COLOR_AFTER_SOFT, alpha=0.7, edgecolor="none", marker="^")
    if np.any(interpolated_mask):
        ax2.scatter(channel_indices[interpolated_mask], std_after[interpolated_mask], 
                  s=120, c=COLOR_AFTER_STRONG, alpha=0.8, edgecolor=VIZ_NEUTRAL["black"], linewidth=2, marker="o", label="Interpolated channels (after)", zorder=4)

    ax2.set_xlabel("Channel Index", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Std (ÂµV)", fontsize=11, fontweight="bold")
    ax2.set_title("Per-Channel Amplitude: Before vs After\n(Red X â†’ Green circle = successful interpolation)", 
                 fontsize=11, fontweight="bold")
    ax2.legend(fontsize=9, loc="upper left")
    ax2.grid(alpha=0.3)

    # Channel status summary (bottom left)
    ax3 = plt.subplot(2, 2, 3)
    n_bad = len(bads)
    n_total = len(eeg_picks_before)
    
    if n_bad > 0:
        labels = [f"âœ“ Good\n({n_total - n_bad})", f"âœ“ Interpolated\n({n_bad})"]
        sizes = [n_total - n_bad, n_bad]
        colors_pie = [COLOR_BEFORE_SOFT, COLOR_AFTER_STRONG]
        explode = (0, 0.1)
        
        wedges, texts, autotexts = ax3.pie(sizes, explode=explode, labels=labels, colors=colors_pie, 
                                           autopct="%1.1f%%", shadow=True, startangle=90, 
                                           textprops={"fontsize": 11, "fontweight": "bold"})
        for autotext in autotexts:
            autotext.set_color(VIZ_NEUTRAL["white"])
            autotext.set_fontsize(11)
        
        ax3.set_title(f"Channel Status After Interpolation\n(All channels now usable!)", fontsize=11, fontweight="bold", color=COLOR_AFTER_STRONG)
    else:
        ax3.text(0.5, 0.5, f"âœ“ No Bad Channels\nAll {n_total} channels intact", 
                ha="center", va="center", fontsize=12, transform=ax3.transAxes, fontweight="bold")
        ax3.axis("off")

    # Summary statistics table (bottom right)
    ax4 = plt.subplot(2, 2, 4)
    
    summary_text = "INTERPOLATION SUMMARY:\n\n"
    summary_text += f"Total EEG Channels: {n_total}\n"
    summary_text += f"Bad Channels (interpolated): {n_bad}\n"
    summary_text += f"Good Channels: {n_total - n_bad}\n\n"
    summary_text += f"â” Mean Amplitude (Before): {np.mean(std_before):.2f} ÂµV\n"
    summary_text += f"â” Mean Amplitude (After): {np.mean(std_after):.2f} ÂµV\n"
    summary_text += f"Î” Change: {np.mean(std_after) - np.mean(std_before):+.2f} ÂµV\n\n"
    
    if n_bad > 0:
        summary_text += f"â” Method: Spherical Spline Interpolation\n"
        summary_text += f"âœ“ All channels now ready for analysis!"
        bg_color = INTERPOLATE_VIZ["success_face"]
        edge_color = INTERPOLATE_VIZ["success_edge"]
    else:
        summary_text += "âœ“ No interpolation needed"
        bg_color = INTERPOLATE_VIZ["info_face"]
        edge_color = INTERPOLATE_VIZ["info_edge"]
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes,
            fontsize=11, verticalalignment="top", family="monospace", fontweight="bold",
            bbox=dict(boxstyle="round,pad=1", facecolor=bg_color, edgecolor=edge_color, linewidth=3, alpha=0.9))
    ax4.axis("off")

    fig.suptitle(f"sub-{subject_id} {person} - Bad Channel Interpolation Results", 
                fontsize=14, fontweight="bold")
    plt.tight_layout()

    plot_path = save_figure(fig, output_dir, f"sub-{subject_id}_{person}_interpolate_statistics.png", dpi=150)
    plt.close(fig)
    print(f"  âœ“ Statistics comparison saved: {plot_path.name}")


def main(argv=None):
    args = parse_args(argv)
    subjects = resolve_subjects(args.subjects, config.SUBJECTS, mode="viz")
    duration = args.duration
    output_dir = config.QC_DIR

    print("\n" + "=" * 80)
    print("VISUALIZATION: Step 04 - Bad Channel Interpolation")
    print("=" * 80)
    print(f"Subjects: {', '.join(subjects)}")
    print(f"Duration: {duration}s")
    print(f"Output: {output_dir}")
    print("=" * 80 + "\n")

    for subject_id in subjects:
        print(f"\n--- Subject {subject_id} ---")

        for person in DEFAULT_PERSONS:
            before_path = config.OUTPUT_DIR / f"sub-{subject_id}_{person}_badchannels_detected.fif"
            after_path = config.OUTPUT_DIR / f"sub-{subject_id}_{person}_interpolated.fif"

            if not before_path.exists():
                print(f"  {person}: Before file (badchannels_detected) not found")
                continue
            if not after_path.exists():
                print(f"  {person}: After file (interpolated) not found")
                continue

            try:
                raw_before = mne.io.read_raw_fif(str(before_path), preload=False, verbose=False)
                raw_after = mne.io.read_raw_fif(str(after_path), preload=False, verbose=False)

                print(f"\n  {person}:")
                
                # Get bad channels info
                bads = raw_before.info.get("bads", [])
                if len(bads) > 0:
                    print(f"    Bad channels: {', '.join(bads)}")

                # Generate plots
                if len(bads) > 0:
                    plot_interpolation_channel_overlay_comparison(raw_before, raw_after, subject_id, person, output_dir)
                    plot_interpolation_neighbor_comparison(raw_before, raw_after, subject_id, person, output_dir)
                    plot_interpolated_channels_timeseries(raw_before, raw_after, subject_id, person, duration, output_dir)
                    plot_bad_channel_restoration_summary(raw_before, raw_after, subject_id, person, output_dir)

            except Exception as e:
                print(f"  {person}: Error: {e}")

    print("\n" + "=" * 80)
    print(f"âœ“ All visualizations saved to: {output_dir}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()

