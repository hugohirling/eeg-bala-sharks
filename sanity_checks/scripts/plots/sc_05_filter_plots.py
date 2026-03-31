"""
Internal plot helpers for step 05 (filter).

Entry point:
    python sanity_checks/scripts/sc_05_filter.py --mode viz
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import mne
import numpy as np
from mne.channels.layout import _find_topomap_coords

from preprocessing import config
from helpers.sc_config import FILTER_VIZ, SCALP_CMAP, VIZ_NEUTRAL
from helpers.sc_plot_io import save_figure
from helpers.sc_signal import compute_psd as _compute_psd

COLOR_BEFORE = FILTER_VIZ["before"]
COLOR_AFTER = FILTER_VIZ["after"]
COLOR_MEAN = FILTER_VIZ["mean"]
COLOR_DELTA = FILTER_VIZ["delta"]
COLOR_PASSBAND = FILTER_VIZ["passband"]
COLOR_STOPBAND = FILTER_VIZ["stopband"]
PSD_FMAX = 60.0
LOW_BAND = (0.0, float(config.FREQ_LOWER))
PASS_BAND = (float(config.FREQ_LOWER), float(config.FREQ_UPPER))
HIGH_BAND = (float(config.FREQ_UPPER), PSD_FMAX)


def band_mean(psd_values, freqs, band):
    low, high = band
    if low == 0.0:
        mask = (freqs >= low) & (freqs < high)
    else:
        mask = (freqs >= low) & (freqs <= high)
    if not np.any(mask):
        return np.nan
    return np.mean(psd_values[..., mask], axis=-1)


def _band_integral(psd_values, freqs, band):
    low, high = band
    if low == 0.0:
        mask = (freqs >= low) & (freqs < high)
    else:
        mask = (freqs >= low) & (freqs <= high)
    if not np.any(mask):
        return np.nan
    return np.trapezoid(psd_values[..., mask], freqs[mask], axis=-1)


def _pick_representative_channel(raw):
    eeg_names = [raw.ch_names[idx] for idx in mne.pick_types(raw.info, eeg=True, exclude=[])]
    preferred = ["Fz", "Cz", "Pz", "F3", "F4", "C3", "C4", "P3", "P4", "Fp1", "Fp2", "Oz"]
    for channel_name in preferred:
        if channel_name in eeg_names:
            return channel_name
    return eeg_names[0] if eeg_names else None


def _get_channel_topomap_colors(raw_eeg):
    picks = np.arange(len(raw_eeg.ch_names))
    coords = _find_topomap_coords(raw_eeg.info, picks)
    angles = (np.arctan2(coords[:, 1], coords[:, 0]) + np.pi) / (2.0 * np.pi)
    colors = SCALP_CMAP(angles)
    return coords, angles, colors


def _annotate_frequency_bands(ax):
    ymin, ymax = ax.get_ylim()
    y_text = ymax - 0.04 * (ymax - ymin)
    labels = [
        ((0.0 + config.FREQ_LOWER) / 2.0, "<1 Hz"),
        (((config.FREQ_LOWER + config.FREQ_UPPER) / 2.0), "1-40 Hz"),
        (((config.FREQ_UPPER + PSD_FMAX) / 2.0), ">40 Hz"),
    ]
    for xpos, label in labels:
        ax.text(xpos, y_text, label, ha="center", va="top", fontsize=9, fontweight="bold", color=VIZ_NEUTRAL["text_mid"])


def plot_psd_comparison(raw_before, raw_after, subject_id, person, duration, output_dir):
    freqs_before, mean_before, psd_before, raw_before_eeg = _compute_psd(raw_before, duration)
    freqs_after, mean_after, psd_after, raw_after_eeg = _compute_psd(raw_after, duration)
    if freqs_before is None or freqs_after is None or raw_before_eeg is None or raw_after_eeg is None:
        return

    mean_before_db = 10.0 * np.log10(np.maximum(mean_before, np.finfo(float).tiny))
    mean_after_db = 10.0 * np.log10(np.maximum(mean_after, np.finfo(float).tiny))
    psd_before_db = 10.0 * np.log10(np.maximum(psd_before, np.finfo(float).tiny))
    psd_after_db = 10.0 * np.log10(np.maximum(psd_after, np.finfo(float).tiny))
    coords, angles, channel_colors = _get_channel_topomap_colors(raw_before_eeg)

    fig = plt.figure(figsize=(16.6, 5.9), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.12, 2.15, 2.15], wspace=0.18)
    ax_topo = fig.add_subplot(gs[0, 0])
    ax_before = fig.add_subplot(gs[0, 1])
    ax_after = fig.add_subplot(gs[0, 2], sharey=ax_before)
    y_min = float(min(np.min(psd_before_db), np.min(psd_after_db))) - 2.0
    y_max = float(max(np.max(psd_before_db), np.max(psd_after_db))) + 2.0

    try:
        raw_before_eeg.plot_sensors(kind="topomap", show_names=False, show=False, sphere="eeglab", axes=ax_topo)
    except Exception:
        raw_before_eeg.plot_sensors(kind="topomap", show_names=False, show=False, sphere="auto", axes=ax_topo)
    scalp = ax_topo.scatter(coords[:, 0], coords[:, 1], c=angles, cmap=SCALP_CMAP, s=104, edgecolors="white", linewidth=0.8, clip_on=False, zorder=5)
    ax_topo.set_title("Sensor color key", fontsize=12, fontweight="bold", pad=10)
    ax_topo.set_aspect("equal")
    ax_topo.set_axis_off()
    cbar = fig.colorbar(scalp, ax=ax_topo, fraction=0.065, pad=0.04)
    cbar.set_label("Scalp position", fontsize=9)
    cbar.set_ticks([0.125, 0.375, 0.625, 0.875])
    cbar.set_ticklabels(["L front", "L back", "R back", "R front"])
    cbar.ax.tick_params(labelsize=8.5)

    for ax, spectra_db, mean_db, stage_title in [
        (ax_before, psd_before_db, mean_before_db, "Before filter"),
        (ax_after, psd_after_db, mean_after_db, "After filter"),
    ]:
        ax.axvspan(0.0, config.FREQ_LOWER, color=COLOR_STOPBAND, alpha=0.65, lw=0)
        ax.axvspan(config.FREQ_LOWER, config.FREQ_UPPER, color=COLOR_PASSBAND, alpha=0.8, lw=0)
        ax.axvspan(config.FREQ_UPPER, PSD_FMAX, color=COLOR_STOPBAND, alpha=0.65, lw=0)
        for channel_idx, color in enumerate(channel_colors):
            ax.plot(freqs_before if ax is ax_before else freqs_after, spectra_db[channel_idx], color=color, linewidth=0.9, alpha=0.24)
        ax.plot(freqs_before if ax is ax_before else freqs_after, mean_db, color=COLOR_MEAN, linewidth=3.2, label="Mean across 64 channels")
        ax.axvline(config.FREQ_LOWER, color=FILTER_VIZ["threshold"], linestyle="--", linewidth=1.3)
        ax.axvline(config.FREQ_UPPER, color=FILTER_VIZ["threshold"], linestyle="--", linewidth=1.3)
        ax.set_xlim(0.0, PSD_FMAX)
        ax.set_ylim(y_min, y_max)
        ax.set_title(stage_title, fontsize=12, fontweight="bold", pad=8)
        ax.set_xlabel("Frequency (Hz)")
        ax.grid(alpha=0.22, linestyle=":")
        ax.legend(loc="upper right", fontsize=9.5, frameon=True, facecolor=VIZ_NEUTRAL["white"], framealpha=0.9)
        _annotate_frequency_bands(ax)

    ax_before.set_ylabel("PSD per channel (dB)")
    ax_after.tick_params(labelleft=False)

    low_before = band_mean(mean_before, freqs_before, LOW_BAND)
    low_after = band_mean(mean_after, freqs_after, LOW_BAND)
    pass_before = band_mean(mean_before, freqs_before, PASS_BAND)
    pass_after = band_mean(mean_after, freqs_after, PASS_BAND)
    high_before = band_mean(mean_before, freqs_before, HIGH_BAND)
    high_after = band_mean(mean_after, freqs_after, HIGH_BAND)
    low_change = ((low_after - low_before) / low_before * 100.0) if low_before > 0 else np.nan
    pass_change = ((pass_after - pass_before) / pass_before * 100.0) if pass_before > 0 else np.nan
    high_change = ((high_after - high_before) / high_before * 100.0) if high_before > 0 else np.nan

    fig.suptitle(
        f"sub-{subject_id} {person} - Filter PSD Comparison\n"
        f"All EEG channels with scalp-based colors | low: {low_change:+.1f}% | passband: {pass_change:+.1f}% | high: {high_change:+.1f}%",
        fontsize=14,
        fontweight="bold",
    )

    plot_path = save_figure(fig, output_dir, f"sub-{subject_id}_{person}_filter_psd_comparison.png", dpi=150)
    plt.close(fig)
    print(f"  âœ“ PSD comparison saved: {plot_path.name}")


def plot_timeseries_comparison(raw_before, raw_after, subject_id, person, duration, output_dir):
    channel_name = _pick_representative_channel(raw_before)
    if channel_name is None:
        return

    t_end = min(float(duration), 5.0, float(raw_before.times[-1]), float(raw_after.times[-1]))
    if t_end <= 0.5:
        return

    stop_before = int(round(t_end * float(raw_before.info["sfreq"])))
    stop_after = int(round(t_end * float(raw_after.info["sfreq"])))
    data_before = raw_before.get_data(picks=[channel_name], start=0, stop=stop_before)[0] * 1e6
    data_after = raw_after.get_data(picks=[channel_name], start=0, stop=stop_after)[0] * 1e6
    times_before = raw_before.times[: len(data_before)]
    times_after = raw_after.times[: len(data_after)]

    data_before = data_before - np.median(data_before)
    data_after = data_after - np.median(data_after)
    n_common = min(len(data_before), len(data_after))
    delta_uv = data_after[:n_common] - data_before[:n_common]

    fig, axes = plt.subplots(3, 1, figsize=(14, 8.5), sharex=True)
    axes[0].plot(times_before, data_before, color=COLOR_BEFORE, linewidth=0.9)
    axes[0].set_title(f"Before filter ({channel_name})", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Centered ÂµV")
    axes[0].grid(alpha=0.28, linestyle=":")

    axes[1].plot(times_after, data_after, color=COLOR_AFTER, linewidth=0.9)
    axes[1].set_title(f"After filter ({channel_name})", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("Centered ÂµV")
    axes[1].grid(alpha=0.28, linestyle=":")

    axes[2].plot(times_after[:n_common], delta_uv, color=COLOR_DELTA, linewidth=0.9)
    axes[2].axhline(0.0, color=VIZ_NEUTRAL["line_soft"], linestyle="--", linewidth=1.0)
    axes[2].set_title("Difference introduced by filter (after - before)", fontsize=11, fontweight="bold")
    axes[2].set_ylabel("Delta ÂµV")
    axes[2].set_xlabel("Time (s)")
    axes[2].grid(alpha=0.28, linestyle=":")

    before_std = float(np.std(data_before))
    after_std = float(np.std(data_after))
    delta_std = float(np.std(delta_uv))
    fig.suptitle(
        f"sub-{subject_id} {person} - Time-Domain Filter Check\n"
        f"Representative channel {channel_name} | std before={before_std:.2f} ÂµV | "
        f"std after={after_std:.2f} ÂµV | std delta={delta_std:.2f} ÂµV",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])

    plot_path = save_figure(fig, output_dir, f"sub-{subject_id}_{person}_filter_timeseries_comparison.png", dpi=150)
    plt.close(fig)
    print(f"  âœ“ Time series comparison saved: {plot_path.name}")


def plot_bandpower_summary(raw_before, raw_after, subject_id, person, duration, output_dir):
    freqs_before, _, psd_before, raw_before_eeg = _compute_psd(raw_before, duration)
    freqs_after, _, psd_after, _ = _compute_psd(raw_after, duration)
    if freqs_before is None or freqs_after is None or raw_before_eeg is None:
        return

    band_labels = ["0-1 Hz", "1-40 Hz", "40-60 Hz"]
    bands = [LOW_BAND, PASS_BAND, HIGH_BAND]
    before_bandpower = np.array([_band_integral(psd_before, freqs_before, band) for band in bands], dtype=float)
    after_bandpower = np.array([_band_integral(psd_after, freqs_after, band) for band in bands], dtype=float)
    before_values = np.mean(before_bandpower, axis=1)
    after_values = np.mean(after_bandpower, axis=1)
    before_std = np.std(before_bandpower, axis=1)
    after_std = np.std(after_bandpower, axis=1)
    change_pct = np.full(len(bands), np.nan, dtype=float)
    valid = before_values > 0
    change_pct[valid] = (after_values[valid] - before_values[valid]) / before_values[valid] * 100.0

    scale_factor = 1e12
    before_scaled = before_values * scale_factor
    after_scaled = after_values * scale_factor

    fig, ax = plt.subplots(1, 1, figsize=(10.5, 5.8))
    x = np.arange(len(bands))
    width = 0.34

    ax.bar(x - width / 2, before_scaled, width, yerr=before_std * scale_factor, capsize=5, color=COLOR_BEFORE, alpha=0.85, edgecolor="black", label="Before filter")
    ax.bar(x + width / 2, after_scaled, width, yerr=after_std * scale_factor, capsize=5, color=COLOR_AFTER, alpha=0.85, edgecolor="black", label="After filter")
    ax.set_xticks(x)
    ax.set_xticklabels(band_labels)
    ax.set_ylabel("Integrated band power (x10^-12 V^2)")
    ax.set_title("Bandpower before vs after filter", fontsize=11, fontweight="bold")
    ax.grid(axis="y", alpha=0.28)
    ax.legend(fontsize=9)
    ax.set_ylim(bottom=0.0)
    for idx, pct in enumerate(change_pct):
        ax.text(x[idx], max(before_scaled[idx], after_scaled[idx]) * 1.03, f"{pct:+.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    fig.suptitle(
        f"sub-{subject_id} {person} - Filter Bandpower Summary\n"
        f"Mean integrated band power across {len(raw_before_eeg.ch_names)} EEG channels (error bars = SD)",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])

    plot_path = save_figure(fig, output_dir, f"sub-{subject_id}_{person}_filter_bandpower_summary.png", dpi=150)
    plt.close(fig)
    print(f"  âœ“ Bandpower summary saved: {plot_path.name}")

