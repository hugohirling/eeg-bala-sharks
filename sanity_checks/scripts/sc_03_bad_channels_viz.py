"""
Sanity Check Visualization for Step 03: Bad Channels Detect

Creates visualization plots for bad channel detection:
- Topomap highlighting detected bad channels
- Channel amplitude comparison (good vs bad)
- QC report visualization (noise levels per channel)

Usage:
    python sanity_checks/scripts/sc_03_bad_channels_viz.py [--subjects 01,02]

Options:
    --subjects: Comma-separated subject IDs (default: first 2)
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import mne
import numpy as np
import pandas as pd
from textwrap import wrap
from mne.channels.layout import _find_topomap_coords

CURRENT_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = CURRENT_DIR.parent.parent / "eeg_pipeline"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from preprocessing import config


# Fixed absolute-amplitude scale in uV (derived from sub-04_P2 reference plot)
# so all generated all-channel timeseries plots are directly comparable.
FIXED_ABS_SCALE_UV = 50.949631

# Shared category colors/labels across all sanity-check figures.
COLOR_GOOD = "black"
COLOR_MANUAL = "#ff7f0e"
COLOR_AUTO = "#d62728"
COLOR_OVERLAP = "#1f77b4"
LABEL_MANUAL = "Manual (tsv file)"
LABEL_AUTO = "Auto (our script)"
LABEL_OVERLAP = "Overlap (tsv + script)"
QC_COLOR_MANUAL = COLOR_MANUAL
QC_COLOR_AUTO = COLOR_AUTO
QC_COLOR_OVERLAP = COLOR_OVERLAP
QC_COLOR_THRESHOLD = "#e377c2"


def _wrap_prefixed_line(prefix, text, width=56):
    """Wrap long text while keeping subsequent lines aligned under the prefix."""
    text = text.strip()
    if not text:
        return [prefix.rstrip()]

    wrapped = wrap(text, width=width, break_long_words=False, break_on_hyphens=False)
    if not wrapped:
        return [prefix.rstrip()]

    indent = " " * len(prefix)
    lines = [f"{prefix}{wrapped[0]}"]
    lines.extend(f"{indent}{part}" for part in wrapped[1:])
    return lines


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize bad channel detection (channels marked for exclusion/interpolation)",
    )
    parser.add_argument(
        "--subjects",
        type=str,
        default=None,
        help="Comma-separated subject IDs. Default: first 2.",
    )
    return parser.parse_args()


def _normalize_subject_id(subject_id):
    """Return subject IDs in the zero-padded format used by the pipeline files."""
    value = str(subject_id).strip()
    if not value:
        return value
    return value.zfill(2) if value.isdigit() else value


def get_subjects(subject_str):
    if subject_str:
        return [_normalize_subject_id(part) for part in subject_str.split(",") if part.strip()]
    return [_normalize_subject_id(subject_id) for subject_id in list(config.SUBJECTS)[:2]]


def _get_report_channel_sets(subject_id, person, channels):
    """Return TSV-suggested and auto-detected channel sets from step-03 report."""
    report_path = config.BAD_CHANNELS_DIR / f"sub-{subject_id}_{person}_bad_channels_detect.tsv"
    if not report_path.exists():
        return set(), set()

    report = pd.read_csv(report_path, sep="\t")
    if report.empty or "channel" not in report.columns:
        return set(), set()

    report = report.set_index("channel", drop=False)

    suggested_set = set()
    detected_set = set()
    if "reason" in report.columns:
        auto_tokens = {"outlier_std", "flat"}
        for ch in channels:
            if ch not in report.index:
                continue
            reason_text = str(report.loc[ch, "reason"]) if pd.notna(report.loc[ch, "reason"]) else ""
            tokens = {token.strip() for token in reason_text.split(",") if token.strip()}
            if "manual_tsv" in tokens:
                suggested_set.add(ch)
            if tokens.intersection(auto_tokens):
                detected_set.add(ch)

    # Fallback for older reports without reason tokens.
    if not suggested_set and "suggested" in report.columns:
        suggested_mask = report["suggested"].astype(str).str.strip().str.lower() == "yes"
        suggested_set = set(ch for ch in report.loc[suggested_mask, "channel"].astype(str).tolist() if ch in channels)

    return suggested_set, detected_set


def _smooth_for_display(signal_1d, sfreq, window_ms=12.0):
    """Apply a short moving-average to reduce jagged rendering in QC plots."""
    win = int(max(1, round((window_ms / 1000.0) * float(sfreq))))
    if win <= 1:
        return signal_1d
    if win % 2 == 0:
        win += 1
    kernel = np.ones(win, dtype=float) / float(win)
    return np.convolve(signal_1d, kernel, mode="same")


def plot_all_channels_timeseries(raw, subject_id, person, output_dir, duration_sec=1.0):
    """Plot all EEG channel traces with bad channels highlighted in color and others in black."""
    eeg_picks = mne.pick_types(raw.info, eeg=True, exclude=[])
    if len(eeg_picks) == 0:
        return

    raw_eeg = raw.copy().pick(eeg_picks)
    t_end = min(float(duration_sec), raw_eeg.times[-1])
    n_samples = int(t_end * raw_eeg.info["sfreq"])
    if n_samples <= 10:
        return

    data_uv = raw_eeg.get_data(start=0, stop=n_samples) * 1e6
    times = raw_eeg.times[:n_samples]
    ch_names = raw_eeg.ch_names

    suggested_set, detected_set = _get_report_channel_sets(subject_id, person, ch_names)
    if not suggested_set and not detected_set:
        detected_set = set(raw_eeg.info.get("bads", []))

    bad_union = suggested_set.union(detected_set)
    both_set = suggested_set.intersection(detected_set)
    suggested_only_set = suggested_set - detected_set
    detected_only_set = detected_set - suggested_set
    bad_indices = [idx for idx, ch in enumerate(ch_names) if ch in bad_union]
    good_indices = [idx for idx, ch in enumerate(ch_names) if ch not in bad_union]
    detected_indices = [idx for idx, ch in enumerate(ch_names) if ch in detected_set]

    # Center first, then use absolute amplitudes for plotting.
    data_centered = data_uv - np.median(data_uv, axis=1, keepdims=True)
    data_abs_uv = np.abs(data_centered)

    # Channel locations for spatial nearest-neighbor lookup.
    ch_locs = []
    for ch in raw_eeg.info["chs"]:
        loc = np.asarray(ch.get("loc", np.array([np.nan, np.nan, np.nan]))[:3], dtype=float)
        if np.all(np.isfinite(loc)):
            ch_locs.append(loc)
        else:
            ch_locs.append(np.array([np.nan, np.nan, np.nan], dtype=float))
    ch_locs = np.asarray(ch_locs, dtype=float)

    def _nearest_clean_neighbors(target_idx, k=2):
        if target_idx >= len(ch_locs) or not np.all(np.isfinite(ch_locs[target_idx])):
            return []
        target_loc = ch_locs[target_idx]
        dists = []
        for idx in good_indices:
            if idx < len(ch_locs) and np.all(np.isfinite(ch_locs[idx])):
                dists.append((float(np.linalg.norm(ch_locs[idx] - target_loc)), idx))
        dists.sort(key=lambda item: item[0])
        return [idx for _, idx in dists[:k]]

    # Shared absolute-amplitude scale for both panels so traces remain directly comparable.
    common_abs_peak_uv = float(FIXED_ABS_SCALE_UV)

    # Figure 1: all channels stacked, bad channels highlighted.
    fig_all, ax_all = plt.subplots(1, 1, figsize=(16, 10))
    spacing_all = common_abs_peak_uv * 2.4
    offsets_all = np.arange(len(ch_names)) * spacing_all
    for idx, ch_name in enumerate(ch_names):
        y = _smooth_for_display(data_abs_uv[idx], raw_eeg.info["sfreq"], window_ms=8.0) + offsets_all[idx]
        if ch_name in both_set:
            color = COLOR_OVERLAP
        elif ch_name in detected_only_set:
            color = COLOR_AUTO
        elif ch_name in suggested_only_set:
            color = COLOR_MANUAL
        else:
            color = COLOR_GOOD

        if idx in bad_indices:
            ax_all.plot(times, y, color=color, linewidth=0.95, alpha=0.9, zorder=4)
            ax_all.text(
                times[-1] + 0.03,
                offsets_all[idx],
                ch_name,
                fontsize=8,
                fontweight="bold",
                color=color,
                va="center",
            )
        else:
            ax_all.plot(times, y, color=COLOR_GOOD, linewidth=0.32, alpha=0.75, zorder=2)

    ax_all.set_ylabel("All channels (stacked)")
    ax_all.set_title(
        f"All EEG channels (first {t_end:.1f}s) - absolute amplitude (|uV|), stacked"
        "\nGood=black | Manual (TSV file)=orange | Auto (our script)=red | Overlap=blue"
    )
    ax_all.set_xlim(times[0], times[-1])
    ax_all.grid(alpha=0.2, axis="x")
    ax_all.set_yticks([])

    n_bad = len(bad_indices)
    n_good = len(ch_names) - n_bad
    n_manual_only = len(suggested_only_set)
    n_auto_only = len(detected_only_set)
    n_overlap = len(both_set)
    legend_handles = [
        plt.Line2D([0], [0], color=COLOR_GOOD, lw=1.2, alpha=0.85, label=f"Good channels ({n_good})"),
        plt.Line2D([0], [0], color=COLOR_MANUAL, lw=1.4, label=f"{LABEL_MANUAL} ({n_manual_only})"),
        plt.Line2D([0], [0], color=COLOR_AUTO, lw=1.4, label=f"{LABEL_AUTO} ({n_auto_only})"),
        plt.Line2D([0], [0], color=COLOR_OVERLAP, lw=1.4, label=f"{LABEL_OVERLAP} ({n_overlap})"),
    ]
    ax_all.legend(handles=legend_handles, loc="upper left", fontsize=9)

    fig_all.suptitle(
        (
            f"sub-{subject_id} {person} - Step 03 Channel Overlay (Overview)"
            f" (manual-only={n_manual_only}, auto-only={n_auto_only}, overlap={n_overlap})"
        ),
        fontsize=13,
        fontweight="bold",
    )
    plt.figure(fig_all.number)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Figure 2: detected channels + nearest clean neighbors.
    fig_bad, ax_bad_abs = plt.subplots(1, 1, figsize=(16, 12))

    # Bottom panel: same time window as top panel for direct comparison.
    zoom_end = t_end
    n_zoom = max(20, int(zoom_end * raw_eeg.info["sfreq"]))
    times_zoom = times[:n_zoom]

    # Lower panel: bad-channel zoom with absolute amplitudes in uV (shared |uV| scale).
    if bad_indices:
        focus_indices = bad_indices
        neighbor_color = "#4d4d4d"
        row_specs = []
        for ch_idx in focus_indices:
            neighbors = _nearest_clean_neighbors(ch_idx, k=2)
            row_specs.append((ch_idx, neighbors))

        spacing_abs = common_abs_peak_uv * 5.2
        n_rows_abs = len(row_specs)
        offsets_abs = np.arange(n_rows_abs) * spacing_abs

        for row_idx, (ch_idx, neighbors) in enumerate(row_specs):
            ch_name = ch_names[ch_idx]
            if ch_name in both_set:
                bad_color = COLOR_OVERLAP
            elif ch_name in detected_only_set:
                bad_color = COLOR_AUTO
            elif ch_name in suggested_only_set:
                bad_color = COLOR_MANUAL
            else:
                bad_color = COLOR_AUTO

            row_offset = offsets_abs[row_idx]
            for n_i, n_idx in enumerate(neighbors):
                y_nei = _smooth_for_display(data_abs_uv[n_idx, :n_zoom], raw_eeg.info["sfreq"], window_ms=8.0)
                y_nei = y_nei + row_offset
                ax_bad_abs.plot(
                    times_zoom,
                    y_nei,
                    color=neighbor_color,
                    linewidth=0.9,
                    alpha=0.75 if n_i == 0 else 0.5,
                    linestyle="-",
                    zorder=2,
                )

            y_abs = _smooth_for_display(data_abs_uv[ch_idx, :n_zoom], raw_eeg.info["sfreq"], window_ms=8.0)
            y_abs = y_abs + row_offset
            ax_bad_abs.plot(times_zoom, y_abs, color=bad_color, linewidth=1.25, alpha=0.95, zorder=3)

            nei_label = ", ".join(ch_names[n_idx] for n_idx in neighbors) if neighbors else "no clean neighbors"
            bad_label = f"{ch_name}"
            yaxis_xform = ax_bad_abs.get_yaxis_transform()
            ax_bad_abs.text(
                1.002,
                row_offset,
                bad_label,
                transform=yaxis_xform,
                fontsize=8,
                fontweight="bold",
                color=bad_color,
                va="center",
                clip_on=False,
            )
            ax_bad_abs.text(
                1.028,
                row_offset,
                f"| N: {nei_label}",
                transform=yaxis_xform,
                fontsize=8,
                fontweight="bold",
                color=neighbor_color,
                va="center",
                clip_on=False,
            )

        ax_bad_abs.set_xlim(times_zoom[0], times_zoom[-1])
        ax_bad_abs.set_ylim(-common_abs_peak_uv * 1.0, offsets_abs[-1] + common_abs_peak_uv * 2.6)
        ax_bad_abs.set_ylabel("Detected + neighbors\n(abs uV, stacked)")
        ax_bad_abs.set_yticks([])
        ax_bad_abs.set_title(
            f"Detected channels + nearest clean neighbors ({zoom_end:.1f}s) - shared uV scale={common_abs_peak_uv:.1f} uV"
        )
    else:
        ax_bad_abs.text(0.5, 0.5, "No bad channels to display", transform=ax_bad_abs.transAxes,
                        ha="center", va="center", fontsize=11)
        ax_bad_abs.set_yticks([])
        ax_bad_abs.set_ylabel("Bad channels")

    ax_bad_abs.set_xlabel("Time (s)")
    ax_bad_abs.grid(alpha=0.2, axis="x")

    fig_bad.suptitle(
        (
            f"sub-{subject_id} {person} - Step 03 Channel Overlay (Neighbor Zoom)"
            f" (manual-only={n_manual_only}, auto-only={n_auto_only}, overlap={n_overlap})"
        ),
        fontsize=13,
        fontweight="bold",
    )
    # Keep extra right margin for per-row channel labels.
    plt.figure(fig_bad.number)
    plt.tight_layout(rect=[0, 0, 0.91, 0.96])

    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path_overview = output_dir / f"sub-{subject_id}_{person}_bad_channels_all_channels_timeseries_overview.png"
    plot_path_neighbors = output_dir / f"sub-{subject_id}_{person}_bad_channels_all_channels_timeseries_neighbors.png"
    fig_all.savefig(plot_path_overview, dpi=140)
    fig_bad.savefig(plot_path_neighbors, dpi=140, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig_all)
    plt.close(fig_bad)
    print(f"  ✓ All-channel overview saved: {plot_path_overview.name}")
    print(f"  ✓ Neighbor zoom saved: {plot_path_neighbors.name}")


def plot_bad_channels_topomap(raw, subject_id, person, output_dir):
    """Plot step-03-consistent topomap with robust z-scores and bad-channel reasons."""
    eeg_picks_all = mne.pick_types(raw.info, eeg=True, exclude=[])
    raw_eeg = raw.copy().pick(eeg_picks_all)
    
    if len(raw_eeg.ch_names) == 0:
        return

    report_path = config.BAD_CHANNELS_DIR / f"sub-{subject_id}_{person}_bad_channels_detect.tsv"
    if not report_path.exists():
        print(f"  {person}: QC report not found for topomap evidence ({report_path.name})")
        return

    report = pd.read_csv(report_path, sep="\t")
    if report.empty:
        print(f"  {person}: QC report is empty, skipping evidence topomap")
        return

    report = report.set_index("channel", drop=False)
    channels = raw_eeg.ch_names
    missing_channels = [ch for ch in channels if ch not in report.index]
    if missing_channels:
        print(f"  {person}: Missing {len(missing_channels)} channels in QC report, skipping topomap")
        return

    z_scores = np.array([float(report.loc[ch, "robust_z"]) for ch in channels], dtype=float)
    std_uv = np.array([float(report.loc[ch, "std"]) * 1e6 for ch in channels], dtype=float)
    reasons = [str(report.loc[ch, "reason"]) if pd.notna(report.loc[ch, "reason"]) else "" for ch in channels]

    suggested_mask = report["suggested"].astype(str).str.strip().str.lower() == "yes"
    suggested_all_set = set(ch for ch in report.loc[suggested_mask, "channel"].tolist() if ch in channels)

    # Important: raw.info['bads'] contains merged auto+manual channels after step 03.
    # For interpretation in sanity plots we split by reason tokens:
    # - detected_set: automatically detected by rules (outlier_std / flat)
    # - suggested_set: manually suggested from TSV (manual_tsv)
    detected_set = set()
    suggested_set = set()
    for ch in channels:
        reason_text = str(report.loc[ch, "reason"]) if pd.notna(report.loc[ch, "reason"]) else ""
        tokens = {token.strip() for token in reason_text.split(",") if token.strip()}
        if "outlier_std" in tokens or "flat" in tokens:
            detected_set.add(ch)
        if "manual_tsv" in tokens:
            suggested_set.add(ch)

    # Keep a fallback to report suggestion flag for channels without explicit reason tokens.
    if not suggested_set and suggested_all_set:
        suggested_set = set(suggested_all_set)

    both_set = suggested_set.intersection(detected_set)
    suggested_only_set = suggested_set - detected_set
    detected_only_set = detected_set - suggested_set

    bads = sorted(suggested_set.union(detected_set))
    bad_indices = [channels.index(ch) for ch in bads]

    reason_by_channel = {
        ch: [token.strip() for token in reason_text.split(",") if token.strip()]
        for ch, reason_text in zip(channels, reasons)
    }

    auto_tokens = {"outlier_std", "flat"}
    suggested_idx = [channels.index(ch) for ch in sorted(suggested_only_set)]
    detected_idx = [channels.index(ch) for ch in sorted(detected_only_set)]
    both_idx = [channels.index(ch) for ch in sorted(both_set)]

    fig = plt.figure(figsize=(12, 9))
    ax_topo = plt.subplot(1, 1, 1)

    threshold = float(getattr(config, "BAD_CHANNEL_ZSCORE_THRESHOLD", 4.0))
    robust_vlim = max(threshold, float(np.percentile(np.abs(z_scores), 95)))
    robust_vlim = min(robust_vlim, threshold * 3.0)
    z_scores_plot = np.clip(z_scores, -robust_vlim, robust_vlim)

    eeg_picks_for_topomap = mne.pick_types(raw_eeg.info, eeg=True, exclude=[])
    pos2d = _find_topomap_coords(raw_eeg.info, eeg_picks_for_topomap)

    try:
        im, _ = mne.viz.plot_topomap(
            z_scores_plot,
            raw_eeg.info,
            axes=ax_topo,
            show=False,
            cmap="RdBu_r",
            vlim=(-robust_vlim, robust_vlim),
            contours=0,
            sphere="eeglab",
        )
    except Exception:
        im, _ = mne.viz.plot_topomap(
            z_scores_plot,
            raw_eeg.info,
            axes=ax_topo,
            show=False,
            cmap="RdBu_r",
            vlim=(-robust_vlim, robust_vlim),
            contours=0,
            sphere="auto",
        )

    # Highlight bad channels directly on the topomap with separate marker styles per source.
    # Important: manual-only and auto-only are exclusive; overlap gets its own marker.
    marker_size = 150
    if suggested_idx:
        ax_topo.scatter(
            pos2d[suggested_idx, 0],
            pos2d[suggested_idx, 1],
            s=marker_size,
            facecolors=COLOR_MANUAL,
            alpha=0.35,
            edgecolors=COLOR_MANUAL,
            linewidth=2.0,
            zorder=5,
            label=LABEL_MANUAL,
        )
    if detected_idx:
        ax_topo.scatter(
            pos2d[detected_idx, 0],
            pos2d[detected_idx, 1],
            s=marker_size,
            c=COLOR_AUTO,
            marker="x",
            linewidth=1.2,
            zorder=6,
            label=LABEL_AUTO,
        )
    if both_idx:
        ax_topo.scatter(
            pos2d[both_idx, 0],
            pos2d[both_idx, 1],
            s=marker_size + 40,
            facecolors=COLOR_OVERLAP,
            alpha=0.35,
            edgecolors=COLOR_OVERLAP,
            linewidth=2.0,
            zorder=6,
            label=LABEL_OVERLAP,
        )
        # If a channel is both manual+auto, draw the auto "x" above the circle.
        ax_topo.scatter(
            pos2d[both_idx, 0],
            pos2d[both_idx, 1],
            s=marker_size,
            c=COLOR_AUTO,
            marker="x",
            linewidth=1.4,
            zorder=9,
        )
    if bad_indices:
        for idx in bad_indices:
            ax_topo.text(
                pos2d[idx, 0] + 0.011,
                pos2d[idx, 1] + 0.011,
                channels[idx],
                ha="left",
                va="bottom",
                fontsize=7.2,
                fontweight="bold",
                color="black",
                zorder=7,
                bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="#444444", linewidth=0.9, alpha=0.95),
            )

    cbar = fig.colorbar(im, ax=ax_topo, shrink=0.85, pad=0.03)
    cbar.set_label("Robust z-score of channel STD")
    ax_topo.set_title(
        f"Detection evidence topomap\nColor = robust z-score (clipped ±{robust_vlim:.1f}), threshold = ±{threshold:.1f}",
        fontsize=11,
        fontweight="bold",
    )
    if bad_indices:
        ax_topo.legend(loc="lower left", fontsize=8, markerscale=0.70, handletextpad=0.30, borderpad=0.20)

    # Build reason text and save it as a separate figure (no overlap with topomap).
    if bad_indices:
        lines = ["BAD CHANNEL REASONS", ""]
        for rank, idx in enumerate(bad_indices, start=1):
            reason_text = reasons[idx] if reasons[idx] else "unknown"
            tokens = set(reason_by_channel.get(channels[idx], []))
            has_manual = "manual_tsv" in tokens
            has_auto = len(tokens.intersection(auto_tokens)) > 0
            channel_name = channels[idx]
            in_suggested = channel_name in suggested_set
            in_detected = channel_name in detected_set
            if in_suggested and in_detected:
                source_label = "detected+manual"
            elif in_suggested:
                source_label = "manual"
            elif in_detected:
                source_label = "detected"
            elif has_manual and has_auto:
                source_label = "auto+manual"
            elif has_manual:
                source_label = "manual_reason"
            else:
                source_label = "auto_reason"
            lines.append(f"{rank}. {channel_name} [{source_label}]")
            lines.append(f"   z={z_scores[idx]:+.2f} | std={std_uv[idx]:.2f} µV")
            lines.extend(_wrap_prefixed_line("   reasons: ", reason_text, width=54))
            lines.append("")
        lines.extend([
            "Reason key:",
            "- outlier_std: |robust z| >= threshold",
            f"- flat: std <= {config.BAD_CHANNEL_FLAT_STD_THRESHOLD:.1e}",
            "- manual_tsv: listed manually in TSV",
        ])
        box_color = "#ffe6e6"
        edge_color = "#d62728"
        lines.extend([
            "",
            f"Manual-only: {len(suggested_only_set)} | Auto-only: {len(detected_only_set)} | Overlap: {len(both_set)}",
            f"Manual-total: {len(suggested_set)} | Auto-total: {len(detected_set)}",
        ])
        info_text = "\n".join(lines)
    else:
        info_text = "No bad channels detected or suggested.\nAll channels are within thresholds."
        box_color = "#e6ffe6"
        edge_color = "#2ca02c"

    fig.suptitle(
        (
            f"sub-{subject_id} {person} - Step 03 Bad Channel Detection "
            f"({len(detected_only_set)} auto-only, {len(suggested_only_set)} manual-only, {len(both_set)} overlap)"
        ),
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / f"sub-{subject_id}_{person}_bad_channels_topomap.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Bad channels topomap saved: {plot_path.name}")

    # Save reasons as its own standalone text figure for reporting.
    n_lines = max(3, len(info_text.splitlines()))
    max_line_len = max(len(line) for line in info_text.splitlines()) if info_text else 0
    fig_w = min(12.5, max(8.8, 0.115 * max_line_len + 1.6))
    fig_h = max(5.4, 0.34 * n_lines + 1.8)
    fig_text, ax_text = plt.subplots(figsize=(fig_w, fig_h))
    fig_text.patch.set_facecolor("white")
    ax_text.axis("off")
    title_text = f"sub-{subject_id} {person} - Bad Channel Reasons"
    ax_text.text(0.02, 0.98, title_text, ha="left", va="top", fontsize=14, fontweight="bold", transform=ax_text.transAxes)
    if bad_indices:
        reason_legend_handles = [
            plt.Line2D([0], [0], color=COLOR_MANUAL, lw=6, label=LABEL_MANUAL),
            plt.Line2D([0], [0], color=COLOR_AUTO, lw=6, label=LABEL_AUTO),
            plt.Line2D([0], [0], color=COLOR_OVERLAP, lw=6, label=LABEL_OVERLAP),
        ]
        fig_text.legend(
            handles=reason_legend_handles,
            loc="upper left",
            bbox_to_anchor=(0.02, 0.93),
            ncol=3,
            fontsize=9.5,
            frameon=False,
            handlelength=1.8,
            columnspacing=1.6,
        )
    ax_text.text(
        0.02,
        0.875 if bad_indices else 0.92,
        info_text,
        va="top",
        ha="left",
        fontsize=11,
        family="monospace",
        linespacing=1.32,
        transform=ax_text.transAxes,
        bbox=dict(boxstyle="round,pad=0.75", facecolor=box_color, edgecolor=edge_color, linewidth=2, alpha=0.9),
    )
    fig_text.subplots_adjust(left=0.02, right=0.985, top=0.985, bottom=0.02)
    text_plot_path = output_dir / f"sub-{subject_id}_{person}_bad_channels_topomap_reasons.png"
    fig_text.savefig(text_plot_path, dpi=180)
    plt.close(fig_text)
    print(f"  ✓ Bad channel reasons saved: {text_plot_path.name}")


def plot_bad_channels_amplitudes(raw_before, raw_after, subject_id, person, output_dir):
    """Compare channel amplitudes before and after bad channel detection marking."""
    eeg_picks_before = mne.pick_types(raw_before.info, eeg=True, exclude=[])
    eeg_picks_after = mne.pick_types(raw_after.info, eeg=True, exclude=[])

    if len(eeg_picks_before) == 0 or len(eeg_picks_after) == 0:
        return

    # Get sample data
    max_samples_before = min(int(raw_before.info["sfreq"] * 60), raw_before.n_times)
    max_samples_after = min(int(raw_after.info["sfreq"] * 60), raw_after.n_times)

    data_before = raw_before.get_data(picks=eeg_picks_before, start=0, stop=max_samples_before)
    data_after = raw_after.get_data(picks=eeg_picks_after, start=0, stop=max_samples_after)

    std_before = np.std(data_before, axis=1) * 1e6
    std_after = np.std(data_after, axis=1) * 1e6

    # Identify bad channels from the step-03 QC report to keep all plots consistent.
    report_path = config.BAD_CHANNELS_DIR / f"sub-{subject_id}_{person}_bad_channels_detect.tsv"
    bads = []
    if report_path.exists():
        report = pd.read_csv(report_path, sep="\t")
        if "suggested" in report.columns and "channel" in report.columns:
            suggested_mask = report["suggested"].astype(str).str.strip().str.lower() == "yes"
            bads = report.loc[suggested_mask, "channel"].astype(str).tolist()
    if not bads:
        bads = raw_after.info.get("bads", [])
    bad_mask_after = np.array([raw_after.ch_names[i] in bads for i in eeg_picks_after])

    fig = plt.figure(figsize=(16, 10))

    # Std distribution histogram (top left)
    ax1 = plt.subplot(2, 2, 1)
    if len(std_before) > 0 and len(std_after) > 0:
        ax1.hist(std_before, bins=15, alpha=0.6, label="Before Detection", 
                color="#1f77b4", edgecolor="black", linewidth=1.2)
        ax1.hist(std_after, bins=15, alpha=0.6, label="After Detection", 
                color="#ff7f0e", edgecolor="black", linewidth=1.2)
        ax1.axvline(np.mean(std_before), color="#1f77b4", linestyle="--", linewidth=2, label=f"Mean (before): {np.mean(std_before):.1f} µV")
        
        if len(bads) > 0:
            # Mark outlier threshold
            outlier_threshold = np.mean(std_before) + 3 * np.std(std_before)
            ax1.axvline(outlier_threshold, color="#d62728", linestyle="--", linewidth=2.5, label=f"Outlier threshold: {outlier_threshold:.1f} µV")
        
        ax1.set_xlabel("Standard Deviation (µV)", fontsize=11, fontweight="bold")
        ax1.set_ylabel("Channel Count", fontsize=11, fontweight="bold")
        ax1.set_title("Amplitude Distribution\n(Bad channels = outliers)", fontsize=11, fontweight="bold")
        ax1.legend(fontsize=9)
        ax1.grid(alpha=0.3, axis="y")

    # Per-channel scatter (top right) — CLEARLY MARK BAD CHANNELS
    ax2 = plt.subplot(2, 2, 2)
    if len(std_after) > 0:
        channel_indices = np.arange(len(std_after))
        good_mask = ~bad_mask_after
        
        # Plot good channels (large, green circles)
        ax2.scatter(channel_indices[good_mask], std_after[good_mask], 
                  s=150, c="#2ca02c", alpha=0.7, edgecolors="black", linewidth=2, label="✓ GOOD", zorder=3)
        
        # Plot bad channels (large red X with halo)
        if np.any(bad_mask_after):
            ax2.scatter(channel_indices[bad_mask_after], std_after[bad_mask_after], 
                       s=300, c="#d62728", marker="X", linewidth=3, label="✗ BAD", zorder=4, edgecolors="darkred")
            # Add red circle halos around bad channels
            for idx in np.where(bad_mask_after)[0]:
                circle = plt.Circle((idx, std_after[idx]), 3, fill=False, edgecolor="#d62728", linewidth=3, linestyle="--", alpha=0.5)
                ax2.add_patch(circle)
        
        ax2.set_xlabel("Channel Index", fontsize=11, fontweight="bold")
        ax2.set_ylabel("Std (µV)", fontsize=11, fontweight="bold")
        ax2.set_title(f"Per-Channel Amplitude: {len(bads)} BAD channels identified", fontsize=11, fontweight="bold", color="#d62728")
        ax2.legend(fontsize=10, loc="upper left")
        ax2.grid(alpha=0.3)

    # Bad channel count summary (bottom left)
    ax3 = plt.subplot(2, 2, 3)
    n_bad = len(bads)
    n_good = len(eeg_picks_after) - n_bad
    
    labels = [f"✓ GOOD\n({n_good})", f"✗ BAD\n({n_bad})"]
    sizes = [n_good, n_bad]
    colors_pie = ["#2ca02c", "#d62728"]
    explode = (0, 0.15) if n_bad > 0 else (0, 0)
    
    wedges, texts, autotexts = ax3.pie(sizes, explode=explode, labels=labels, colors=colors_pie, 
                                        autopct="%1.1f%%", shadow=True, startangle=90, 
                                        textprops={"fontsize": 11, "fontweight": "bold"})
    
    # Enhance the pie chart
    for autotext in autotexts:
        autotext.set_color("white")
        autotext.set_fontsize(12)
        autotext.set_fontweight("bold")
    
    ax3.set_title(f"Channel Status Summary\nTotal: {len(eeg_picks_after)} channels", fontsize=11, fontweight="bold")

    # Bad channels list (bottom right) — PROMINENTLY DISPLAYED
    ax4 = plt.subplot(2, 2, 4)
    if len(bads) > 0:
        bad_list_text = "❌ IDENTIFIED BAD CHANNELS:\n\n"
        for i, ch in enumerate(sorted(bads), 1):
            bad_list_text += f"  {i}. {ch}\n"
        bg_color = "#ffe6e6"
        title_color = "#d62728"
    else:
        bad_list_text = "✓ NO BAD CHANNELS DETECTED\nAll channels are good!"
        bg_color = "#e6ffe6"
        title_color = "#2ca02c"
    
    ax4.text(0.05, 0.95, bad_list_text, transform=ax4.transAxes,
            fontsize=12, verticalalignment="top", family="monospace", fontweight="bold",
            bbox=dict(boxstyle="round,pad=1", facecolor=bg_color, edgecolor=title_color, linewidth=3, alpha=0.9))
    ax4.axis("off")

    fig.suptitle(f"sub-{subject_id} {person} - BAD CHANNEL DETECTION RESULTS", fontsize=14, fontweight="bold")
    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / f"sub-{subject_id}_{person}_bad_channels_amplitudes.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Bad channels amplitude comparison saved: {plot_path.name}")


def plot_qc_report_visualization(subject_id, person, output_dir):
    """Visualize the QC report data if available."""
    qc_report_path = config.BAD_CHANNELS_DIR / f"sub-{subject_id}_{person}_bad_channels_detect.tsv"
    
    if not qc_report_path.exists():
        return

    try:
        qc_data = pd.read_csv(qc_report_path, sep="\t")
        if len(qc_data) == 0:
            return

        required_cols = {"channel", "std", "robust_z", "suggested", "reason"}
        if not required_cols.issubset(set(qc_data.columns)):
            print(f"  WARNING: QC report missing required columns for plotting: {required_cols}")
            return

        qc_data = qc_data.copy()
        qc_data["std"] = pd.to_numeric(qc_data["std"], errors="coerce")
        qc_data["robust_z"] = pd.to_numeric(qc_data["robust_z"], errors="coerce")
        qc_data = qc_data.dropna(subset=["std", "robust_z"])
        if qc_data.empty:
            return

        qc_data["std_uv"] = qc_data["std"] * 1e6
        qc_data["is_suggested"] = qc_data["suggested"].astype(str).str.strip().str.lower() == "yes"
        qc_data["reason"] = qc_data["reason"].fillna("").astype(str)
        qc_data["has_manual"] = qc_data["reason"].str.contains("manual_tsv", regex=False)
        qc_data["has_auto"] = qc_data["reason"].str.contains("outlier_std|flat", regex=True)
        qc_data["is_auto_only"] = qc_data["is_suggested"] & qc_data["has_auto"] & (~qc_data["has_manual"])
        qc_data["is_manual_only"] = qc_data["is_suggested"] & qc_data["has_manual"] & (~qc_data["has_auto"])
        qc_data["is_both"] = qc_data["is_suggested"] & qc_data["has_auto"] & qc_data["has_manual"]
        channel_idx = np.arange(len(qc_data))
        threshold_z = float(getattr(config, "BAD_CHANNEL_ZSCORE_THRESHOLD", 4.0))
        flat_threshold_uv = float(getattr(config, "BAD_CHANNEL_FLAT_STD_THRESHOLD", 1e-12)) * 1e6
        marked_rows = qc_data[qc_data["is_suggested"]]

        def _plot_qc_zscore_panel(ax):
            """Render the robust z-score QC panel."""
            ax.scatter(channel_idx, qc_data["robust_z"], s=42, color="#7a7a7a", alpha=0.75, label="Channels")
            if qc_data["is_auto_only"].any():
                ax.scatter(
                    channel_idx[qc_data["is_auto_only"]],
                    qc_data.loc[qc_data["is_auto_only"], "robust_z"],
                    s=70,
                    color=QC_COLOR_AUTO,
                    marker="x",
                    linewidth=1.2,
                    label=LABEL_AUTO,
                    zorder=8,
                )
            if qc_data["is_manual_only"].any():
                ax.scatter(
                    channel_idx[qc_data["is_manual_only"]],
                    qc_data.loc[qc_data["is_manual_only"], "robust_z"],
                    s=76,
                    facecolors=QC_COLOR_MANUAL,
                    alpha=0.35,
                    edgecolors=QC_COLOR_MANUAL,
                    linewidth=2.0,
                    label=LABEL_MANUAL,
                    zorder=4,
                )
            if qc_data["is_both"].any():
                ax.scatter(
                    channel_idx[qc_data["is_both"]],
                    qc_data.loc[qc_data["is_both"], "robust_z"],
                    s=98,
                    facecolors=QC_COLOR_OVERLAP,
                    alpha=0.35,
                    edgecolors=QC_COLOR_OVERLAP,
                    linewidth=2.0,
                    label=LABEL_OVERLAP,
                    zorder=5,
                )
                ax.scatter(
                    channel_idx[qc_data["is_both"]],
                    qc_data.loc[qc_data["is_both"], "robust_z"],
                    s=70,
                    color=QC_COLOR_AUTO,
                    marker="x",
                    linewidth=1.2,
                    zorder=8,
                )
            ax.axhline(
                threshold_z,
                color=QC_COLOR_THRESHOLD,
                linestyle="--",
                linewidth=2,
                label=f"+z threshold ({threshold_z:.1f})",
            )
            ax.axhline(
                -threshold_z,
                color=QC_COLOR_THRESHOLD,
                linestyle="--",
                linewidth=2,
                label=f"-z threshold ({threshold_z:.1f})",
            )

            for row in marked_rows.itertuples(index=False):
                x_val = np.where(qc_data["channel"].values == row.channel)[0][0]
                y_val = float(row.robust_z)
                ax.text(
                    x_val + 0.25,
                    y_val,
                    str(row.channel),
                    fontsize=7.5,
                    fontweight="bold",
                    ha="left",
                    va="center",
                    bbox=dict(boxstyle="round,pad=0.16", facecolor="white", edgecolor="#555555", alpha=0.9),
                    zorder=6,
                )

            ax.set_xlabel("Channel index")
            ax.set_ylabel("Robust z-score")
            ax.set_title("Detection score per channel")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8.5, markerscale=0.8, handletextpad=0.35, borderpad=0.25)

        def _plot_qc_std_panel(ax):
            """Render the channel-STD QC panel."""
            ax.scatter(channel_idx, qc_data["std_uv"], s=42, color="#7a7a7a", alpha=0.75, label="Channels")
            if qc_data["is_auto_only"].any():
                ax.scatter(
                    channel_idx[qc_data["is_auto_only"]],
                    qc_data.loc[qc_data["is_auto_only"], "std_uv"],
                    s=70,
                    color=QC_COLOR_AUTO,
                    marker="x",
                    linewidth=1.2,
                    label=LABEL_AUTO,
                    zorder=8,
                )
            if qc_data["is_manual_only"].any():
                ax.scatter(
                    channel_idx[qc_data["is_manual_only"]],
                    qc_data.loc[qc_data["is_manual_only"], "std_uv"],
                    s=76,
                    facecolors=QC_COLOR_MANUAL,
                    alpha=0.35,
                    edgecolors=QC_COLOR_MANUAL,
                    linewidth=2.0,
                    label=LABEL_MANUAL,
                    zorder=4,
                )
            if qc_data["is_both"].any():
                ax.scatter(
                    channel_idx[qc_data["is_both"]],
                    qc_data.loc[qc_data["is_both"], "std_uv"],
                    s=98,
                    facecolors=QC_COLOR_OVERLAP,
                    alpha=0.35,
                    edgecolors=QC_COLOR_OVERLAP,
                    linewidth=2.0,
                    label=LABEL_OVERLAP,
                    zorder=5,
                )
                ax.scatter(
                    channel_idx[qc_data["is_both"]],
                    qc_data.loc[qc_data["is_both"], "std_uv"],
                    s=70,
                    color=QC_COLOR_AUTO,
                    marker="x",
                    linewidth=1.2,
                    zorder=8,
                )
            ax.axhline(
                flat_threshold_uv,
                color=QC_COLOR_THRESHOLD,
                linestyle="--",
                linewidth=2,
                label=f"flat threshold ({flat_threshold_uv:.2e} µV)",
            )

            for row in marked_rows.itertuples(index=False):
                x_val = np.where(qc_data["channel"].values == row.channel)[0][0]
                y_val = float(row.std_uv)
                ax.text(
                    x_val + 0.25,
                    y_val,
                    str(row.channel),
                    fontsize=7.5,
                    fontweight="bold",
                    ha="left",
                    va="center",
                    bbox=dict(boxstyle="round,pad=0.16", facecolor="white", edgecolor="#555555", alpha=0.9),
                    zorder=6,
                )

            ax.set_xlabel("Channel index")
            ax.set_ylabel("STD (µV)")
            ax.set_title("Channel STD used by detector")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8.5, markerscale=0.8, handletextpad=0.35, borderpad=0.25)

        fig, axes = plt.subplots(1, 2, figsize=(16, 7.2))
        _plot_qc_zscore_panel(axes[0])
        _plot_qc_std_panel(axes[1])

        bad_rows = qc_data[qc_data["is_suggested"]]
        if len(bad_rows) > 0:
            reason_lines = [f"{row.channel}: {row.reason}" for row in bad_rows.itertuples(index=False)]
            fig.text(
                0.5,
                0.01,
                "Suggested bad channels -> " + " | ".join(reason_lines),
                ha="center",
                va="bottom",
                fontsize=9,
            )

        fig.suptitle(f"sub-{subject_id} {person} - QC Metrics from Bad Channel Detection")
        plt.tight_layout(rect=[0, 0.04, 1, 0.96])

        output_dir.mkdir(parents=True, exist_ok=True)
        plot_path = output_dir / f"sub-{subject_id}_{person}_bad_channels_qc_metrics.png"
        fig.savefig(plot_path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        print(f"  ✓ QC metrics visualization saved: {plot_path.name}")

        fig_z, ax_z = plt.subplots(1, 1, figsize=(9.2, 6.8))
        _plot_qc_zscore_panel(ax_z)
        fig_z.suptitle(f"sub-{subject_id} {person} - QC Detection Scores", fontsize=13, fontweight="bold")
        fig_z.tight_layout(rect=[0, 0, 1, 0.96])
        zscore_path = output_dir / f"sub-{subject_id}_{person}_bad_channels_qc_zscore.png"
        fig_z.savefig(zscore_path, dpi=130, bbox_inches="tight")
        plt.close(fig_z)
        print(f"  ✓ QC z-score panel saved: {zscore_path.name}")

        fig_std, ax_std = plt.subplots(1, 1, figsize=(9.2, 6.8))
        _plot_qc_std_panel(ax_std)
        fig_std.suptitle(f"sub-{subject_id} {person} - QC Channel STD", fontsize=13, fontweight="bold")
        fig_std.tight_layout(rect=[0, 0, 1, 0.96])
        std_path = output_dir / f"sub-{subject_id}_{person}_bad_channels_qc_std.png"
        fig_std.savefig(std_path, dpi=130, bbox_inches="tight")
        plt.close(fig_std)
        print(f"  ✓ QC STD panel saved: {std_path.name}")

    except Exception as e:
        print(f"  WARNING: Could not visualize QC report: {e}")


def main():
    args = parse_args()
    subjects = get_subjects(args.subjects)
    output_dir = config.QC_DIR

    print("\n" + "=" * 80)
    print("VISUALIZATION: Step 03 - Bad Channels Detection")
    print("=" * 80)
    print(f"Subjects: {', '.join(subjects)}")
    print(f"Output: {output_dir}")
    print("=" * 80 + "\n")

    for subject_id in subjects:
        print(f"\n--- Subject {subject_id} ---")

        for person in ["P1", "P2"]:
            before_path = config.OUTPUT_DIR / f"sub-{subject_id}_{person}_renamed_montaged.fif"
            after_path = config.OUTPUT_DIR / f"sub-{subject_id}_{person}_badchannels_detected.fif"

            if not before_path.exists():
                print(f"  {person}: Before file (renamed_montaged) not found")
                continue
            if not after_path.exists():
                print(f"  {person}: After file (badchannels_detected) not found")
                continue

            try:
                raw_before = mne.io.read_raw_fif(str(before_path), preload=False, verbose=False)
                raw_after = mne.io.read_raw_fif(str(after_path), preload=False, verbose=False)

                print(f"\n  {person}:")
                
                # Generate plots
                plot_bad_channels_topomap(raw_after, subject_id, person, output_dir)
                plot_all_channels_timeseries(raw_after, subject_id, person, output_dir)
                plot_qc_report_visualization(subject_id, person, output_dir)

            except Exception as e:
                print(f"  {person}: Error: {e}")

    print("\n" + "=" * 80)
    print(f"✓ All visualizations saved to: {output_dir}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
