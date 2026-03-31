from __future__ import annotations

import argparse
import csv
import json
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
from mne.filter import filter_data
from rich.console import Console
from rich.live import Live
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from scipy.signal import hilbert

CURRENT_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = CURRENT_DIR.parent
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from preprocessing import config

console = Console()
progress = Progress(
    TextColumn("[progress.description]{task.description}"),
    BarColumn(),
    MofNCompleteColumn(),
    TextColumn("•"),
    TimeElapsedColumn(),
    TextColumn("•"),
    TimeRemainingColumn(),
    console=console,
    transient=False,
    refresh_per_second=10,
)

DEFAULT_BANDS: dict[str, tuple[float, float]] = {
    "theta": (4.0, 7.0),
    "alpha": (8.0, 12.0),
    "beta": (13.0, 30.0),
}


def _resolve_subjects(subjects_arg: str | None) -> list[str]:
    """
    Resolves and normalizes subject IDs from CLI input.

    Args:
        subjects_arg (str | None): Comma-separated subject list or None.

    Returns:
        list[str]: Normalized subject IDs (zero-padded when numeric).
    """
    def _normalize(value: str) -> str:
        value = value.strip()
        return value.zfill(2) if value.isdigit() else value

    if subjects_arg:
        return [_normalize(part) for part in subjects_arg.split(",") if part.strip()]
    return [_normalize(str(subject)) for subject in config.SUBJECTS]


def _parse_bands(bands_arg: str | None) -> dict[str, tuple[float, float]]:
    """
    Parses frequency-band configuration from CLI string.

    Expected format: name:fmin-fmax,name2:fmin-fmax.

    Args:
        bands_arg (str | None): Raw --bands argument.

    Returns:
        dict[str, tuple[float, float]]: Mapping band name -> (fmin, fmax).

    Raises:
        ValueError: If band syntax or boundaries are invalid.
    """
    if not bands_arg:
        return dict(DEFAULT_BANDS)

    bands: dict[str, tuple[float, float]] = {}
    parts = [part.strip() for part in bands_arg.split(",") if part.strip()]
    for part in parts:
        if ":" not in part:
            raise ValueError(
                f"Invalid band specification '{part}'. Use name:fmin-fmax, e.g. alpha:8-12"
            )
        name, range_str = part.split(":", maxsplit=1)
        name = name.strip().lower()
        if "-" not in range_str:
            raise ValueError(
                f"Invalid range in '{part}'. Use name:fmin-fmax, e.g. alpha:8-12"
            )
        fmin_str, fmax_str = range_str.split("-", maxsplit=1)
        fmin = float(fmin_str)
        fmax = float(fmax_str)
        if fmin <= 0 or fmax <= fmin:
            raise ValueError(f"Invalid band edges for '{name}': {fmin}-{fmax}")
        bands[name] = (fmin, fmax)

    if not bands:
        raise ValueError("No valid bands parsed from --bands argument.")
    return bands


def _fdr_bh(p_values: np.ndarray) -> np.ndarray:
    """
    Applies Benjamini-Hochberg FDR correction to p-values.

    Args:
        p_values (np.ndarray): Raw p-values.

    Returns:
        np.ndarray: FDR-adjusted p-values (NaN preserved for non-finite inputs).
    """
    p = np.asarray(p_values, dtype=float)
    out = np.full_like(p, np.nan, dtype=float)
    finite_mask = np.isfinite(p)
    p_finite = p[finite_mask]

    if p_finite.size == 0:
        return out

    order = np.argsort(p_finite)
    ranks = np.arange(1, p_finite.size + 1, dtype=float)
    sorted_p = p_finite[order]

    adjusted_sorted = sorted_p * p_finite.size / ranks
    adjusted_sorted = np.minimum.accumulate(adjusted_sorted[::-1])[::-1]
    adjusted_sorted = np.clip(adjusted_sorted, 0.0, 1.0)

    adjusted = np.empty_like(p_finite)
    adjusted[order] = adjusted_sorted
    out[finite_mask] = adjusted
    return out


def _epoch_path(subject_id: str, person: str) -> Path:
    """Builds the expected epoch-file path for one subject/player."""
    return Path(config.OUTPUT_DIR) / f"sub-{subject_id}_{person}_epoch.fif"


def _load_pair_decision_epochs(subject_id: str, tmin: float, tmax: float) -> tuple[mne.Epochs, mne.Epochs]:
    """
    Loads aligned decision-phase EEG epochs for P1 and P2.

    Args:
        subject_id (str): Subject identifier.
        tmin (float): Start of analysis window in seconds.
        tmax (float): End of analysis window in seconds.

    Returns:
        tuple[mne.Epochs, mne.Epochs]: Trial-aligned EEG epochs for P1 and P2.
        Only shared EEG channels with names starting with A or B are retained.

    Raises:
        FileNotFoundError: If epoch files are missing.
        RuntimeError: If channel overlap or aligned trial count is insufficient.
    """
    p1_path = _epoch_path(subject_id, "P1")
    p2_path = _epoch_path(subject_id, "P2")
    if not p1_path.exists() or not p2_path.exists():
        raise FileNotFoundError(f"Missing epoch file(s) for sub-{subject_id}: {p1_path} / {p2_path}")

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"This filename .* does not conform to MNE naming conventions\.",
            category=RuntimeWarning,
        )
        p1_epochs = mne.read_epochs(str(p1_path), preload=True).crop(tmin=tmin, tmax=tmax)
        p2_epochs = mne.read_epochs(str(p2_path), preload=True).crop(tmin=tmin, tmax=tmax)

    p1_epochs.pick("eeg")
    p2_epochs.pick("eeg")

    p2_channel_set = set(p2_epochs.ch_names)
    common_channels = [
        name
        for name in p1_epochs.ch_names
        if name in p2_channel_set and str(name).upper().startswith(("A", "B"))
    ]
    if not common_channels:
        raise RuntimeError(f"No common EEG channels with A/B prefix for sub-{subject_id}")

    p1_epochs = p1_epochs.copy().pick(common_channels)
    p2_epochs = p2_epochs.copy().pick(common_channels)

    n_trials = min(len(p1_epochs), len(p2_epochs))
    if n_trials < 10:
        raise RuntimeError(f"Too few aligned trials for sub-{subject_id}: {n_trials}")

    return p1_epochs[:n_trials], p2_epochs[:n_trials]


def _compute_phase_data(data: np.ndarray, sfreq: float, fmin: float, fmax: float) -> np.ndarray:
    """
    Converts time-series EEG data to unit phasors in a narrow frequency band.

    Args:
        data (np.ndarray): EEG data (trials, channels, times).
        sfreq (float): Sampling frequency in Hz.
        fmin (float): Band-pass lower edge in Hz.
        fmax (float): Band-pass upper edge in Hz.

    Returns:
        np.ndarray: Complex unit phasors for phase-locking computation.
    """
    # Narrowband filtering followed by analytic normalization gives unit phasors in the chosen band.
    filtered = filter_data(data, sfreq=sfreq, l_freq=fmin, h_freq=fmax)
    analytic = hilbert(filtered, axis=-1)
    amplitude = np.abs(analytic)
    amplitude[amplitude == 0.0] = 1.0
    return (analytic / amplitude).astype(np.complex64, copy=False)


def _trial_channel_plv(phase_a: np.ndarray, phase_b: np.ndarray) -> np.ndarray:
    """
    Computes trial-wise PLV per channel between two phase tensors.

    Args:
        phase_a (np.ndarray): Unit phasors for participant A.
        phase_b (np.ndarray): Unit phasors for participant B.

    Returns:
        np.ndarray: PLV values with shape (trials, channels).
    """
    return np.abs(np.mean(phase_a * np.conj(phase_b), axis=-1))


def _compute_shuffle_stats(
    phase_1: np.ndarray,
    phase_2: np.ndarray,
    n_shuffles: int,
    random_state: int,
    batch_size: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes shuffled-trial PLV baseline statistics.

    Args:
        phase_1 (np.ndarray): Unit phasors for participant 1.
        phase_2 (np.ndarray): Unit phasors for participant 2.
        n_shuffles (int): Number of trial-shuffle permutations.
        random_state (int): Random seed.
        batch_size (int): Number of permutations processed per batch.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - Mean shuffled PLV per channel.
            - Global shuffled PLV distribution.
    """
    rng = np.random.default_rng(seed=random_state)
    n_trials = phase_2.shape[0]
    n_channels = phase_2.shape[1]

    shuffle_global = np.empty(n_shuffles, dtype=float)
    shuffle_channel_sum = np.zeros(n_channels, dtype=float)
    phase_1_batched = phase_1[None, ...]

    for start in range(0, n_shuffles, batch_size):
        stop = min(start + batch_size, n_shuffles)
        current_batch = stop - start
        perms = np.asarray([rng.permutation(n_trials) for _ in range(current_batch)], dtype=int)
        shuffled_phase_2 = phase_2[perms]
        shuffled_trial_channel_plv = np.abs(
            np.mean(phase_1_batched * np.conj(shuffled_phase_2), axis=-1)
        )
        shuffled_channel_plv = shuffled_trial_channel_plv.mean(axis=1)
        shuffle_channel_sum += shuffled_channel_plv.sum(axis=0)
        shuffle_global[start:stop] = shuffled_channel_plv.mean(axis=1)

    shuffle_channel_mean = shuffle_channel_sum / max(1, n_shuffles)
    return shuffle_channel_mean, shuffle_global


def _summarize_subject(
    subject_id: str,
    tmin: float,
    tmax: float,
    fmin: float,
    fmax: float,
    n_shuffles: int,
    random_state: int,
) -> tuple[list[dict], dict, mne.Info]:
    """
    Computes real and shuffled interbrain PLV summary for one subject pair.

    Args:
        subject_id (str): Subject identifier.
        tmin (float): Analysis-window start in seconds.
        tmax (float): Analysis-window end in seconds.
        fmin (float): Band-pass lower edge in Hz.
        fmax (float): Band-pass upper edge in Hz.
        n_shuffles (int): Number of shuffled controls.
        random_state (int): Random seed.

    Returns:
        tuple[list[dict], dict, mne.Info]:
            - Per-channel PLV rows.
            - Subject-level global summary.
            - Info object for plotting.
    """
    p1_epochs, p2_epochs = _load_pair_decision_epochs(subject_id=subject_id, tmin=tmin, tmax=tmax)
    x1 = p1_epochs.get_data(copy=True)
    x2 = p2_epochs.get_data(copy=True)

    sfreq = float(p1_epochs.info["sfreq"])
    phase_1 = _compute_phase_data(data=x1, sfreq=sfreq, fmin=fmin, fmax=fmax)
    phase_2 = _compute_phase_data(data=x2, sfreq=sfreq, fmin=fmin, fmax=fmax)

    real_trial_channel_plv = _trial_channel_plv(phase_1, phase_2)
    real_channel_plv = real_trial_channel_plv.mean(axis=0)
    real_global_plv = float(real_channel_plv.mean())

    shuffle_channel_mean, shuffle_global_arr = _compute_shuffle_stats(
        phase_1=phase_1,
        phase_2=phase_2,
        n_shuffles=n_shuffles,
        random_state=random_state,
    )

    if np.std(shuffle_global_arr) > 0:
        z_score = float((real_global_plv - np.mean(shuffle_global_arr)) / np.std(shuffle_global_arr))
    else:
        z_score = float("nan")

    p_value = float((1.0 + np.sum(shuffle_global_arr >= real_global_plv)) / (1.0 + len(shuffle_global_arr)))

    channel_rows: list[dict] = []
    for idx, channel_name in enumerate(p1_epochs.ch_names):
        channel_rows.append(
            {
                "subject": subject_id,
                "channel": channel_name,
                "plv_real": float(real_channel_plv[idx]),
                "plv_shuffle_mean": float(shuffle_channel_mean[idx]),
                "plv_delta": float(real_channel_plv[idx] - shuffle_channel_mean[idx]),
            }
        )

    subject_summary = {
        "subject": subject_id,
        "n_trials_used": int(phase_1.shape[0]),
        "n_channels": int(phase_1.shape[1]),
        "n_times": int(phase_1.shape[2]),
        "sfreq": sfreq,
        "fmin": float(fmin),
        "fmax": float(fmax),
        "tmin": float(tmin),
        "tmax": float(tmax),
        "n_shuffles": int(n_shuffles),
        "global_plv_real": real_global_plv,
        "global_plv_shuffle_mean": float(np.mean(shuffle_global_arr)),
        "global_plv_shuffle_std": float(np.std(shuffle_global_arr)),
        "global_plv_z": z_score,
        "global_plv_perm_p": p_value,
    }

    return channel_rows, subject_summary, p1_epochs.info.copy()


def _group_channel_rows(channel_rows: list[dict]) -> list[dict]:
    """
    Aggregates channel-level PLV metrics across subjects.

    Args:
        channel_rows (list[dict]): Per-subject channel rows.

    Returns:
        list[dict]: Group-mean channel metrics.
    """
    grouped: dict[str, dict] = {}
    for row in channel_rows:
        channel = str(row["channel"])
        entry = grouped.setdefault(
            channel,
            {"channel": channel, "plv_real_sum": 0.0, "plv_shuffle_sum": 0.0, "plv_delta_sum": 0.0, "n": 0},
        )
        entry["plv_real_sum"] += float(row["plv_real"])
        entry["plv_shuffle_sum"] += float(row["plv_shuffle_mean"])
        entry["plv_delta_sum"] += float(row["plv_delta"])
        entry["n"] += 1

    out_rows: list[dict] = []
    for channel in sorted(grouped):
        item = grouped[channel]
        n = max(1, int(item["n"]))
        out_rows.append(
            {
                "channel": channel,
                "plv_real": float(item["plv_real_sum"] / n),
                "plv_shuffle_mean": float(item["plv_shuffle_sum"] / n),
                "plv_delta": float(item["plv_delta_sum"] / n),
            }
        )

    return out_rows


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    """Writes a list of dictionaries to CSV with fixed field order."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def _save_subject_global_plot(subject_rows: list[dict], out_dir: Path, band_name: str) -> Path:
    """Saves per-subject global PLV plot comparing real and shuffled values."""
    rows = sorted(subject_rows, key=lambda row: str(row["subject"]))
    x = np.arange(len(rows), dtype=float)
    real = np.asarray([float(row["global_plv_real"]) for row in rows], dtype=float)
    shuf = np.asarray([float(row["global_plv_shuffle_mean"]) for row in rows], dtype=float)

    fig, ax = plt.subplots(figsize=(9.0, 5.5))
    for idx in range(len(rows)):
        ax.plot([x[idx], x[idx]], [shuf[idx], real[idx]], color="#b0bec5", linewidth=1.2, zorder=1)

    ax.scatter(x, shuf, label="Shuffled mean", color="#546e7a", s=55, zorder=2)
    ax.scatter(x, real, label="Real", color="#d32f2f", s=55, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels([f"sub-{row['subject']}" for row in rows], rotation=45, ha="right")
    ax.set_ylabel("Global interbrain PLV")
    ax.set_title(f"Interbrain Synchrony per Pair ({band_name}, real vs shuffled)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()

    out_path = out_dir / f"interbrain_synchrony_subject_global_real_vs_shuffled_{band_name}.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _plot_topomap_panel(
    values: np.ndarray,
    plot_info: mne.Info,
    *,
    ax: plt.Axes,
    cmap: str,
    vlim: tuple[float, float],
):
    """Plots one topomap panel with robust sphere fallback."""
    try:
        image, _ = mne.viz.plot_topomap(
            values,
            plot_info,
            axes=ax,
            show=False,
            contours=0,
            cmap=cmap,
            vlim=vlim,
            sphere="eeglab",
        )
    except Exception:
        image, _ = mne.viz.plot_topomap(
            values,
            plot_info,
            axes=ax,
            show=False,
            contours=0,
            cmap=cmap,
            vlim=vlim,
            sphere="auto",
        )
    return image


def _save_topomap_triplet(rows: list[dict], plot_info: mne.Info, title: str, out_path: Path) -> Path:
    """Saves a three-panel topomap figure: real, shuffled, and delta PLV."""
    rows = sorted(rows, key=lambda row: str(row["channel"]))
    real = np.asarray([float(row["plv_real"]) for row in rows], dtype=float)
    shuf = np.asarray([float(row["plv_shuffle_mean"]) for row in rows], dtype=float)
    delta = np.asarray([float(row["plv_delta"]) for row in rows], dtype=float)

    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.2))

    vmin_abs = float(min(real.min(), shuf.min()))
    vmax_abs = float(max(real.max(), shuf.max()))
    img_real = _plot_topomap_panel(real, plot_info, ax=axes[0], cmap="viridis", vlim=(vmin_abs, vmax_abs))
    axes[0].set_title("Real PLV")

    img_shuf = _plot_topomap_panel(shuf, plot_info, ax=axes[1], cmap="viridis", vlim=(vmin_abs, vmax_abs))
    axes[1].set_title("Shuffled PLV")

    max_abs = max(float(np.max(np.abs(delta))), np.finfo(float).eps)
    img_delta = _plot_topomap_panel(delta, plot_info, ax=axes[2], cmap="RdBu_r", vlim=(-max_abs, max_abs))
    axes[2].set_title("Real - Shuffled")

    fig.suptitle(title)
    fig.subplots_adjust(left=0.04, right=0.90, top=0.88, bottom=0.08, wspace=0.24)

    # Place each colorbar next to its corresponding plot region.
    cb_width = 0.015
    cb_pad = 0.010

    shuf_pos = axes[1].get_position()
    delta_pos = axes[2].get_position()

    cax_abs = fig.add_axes([shuf_pos.x1 + cb_pad, shuf_pos.y0, cb_width, shuf_pos.height])
    cbar_abs = fig.colorbar(img_shuf, cax=cax_abs)
    cbar_abs.set_label("PLV")

    cax_delta = fig.add_axes([delta_pos.x1 + cb_pad, delta_pos.y0, cb_width, delta_pos.height])
    cbar_delta = fig.colorbar(img_delta, cax=cax_delta)
    cbar_delta.set_label("PLV delta")

    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _save_group_topomap(group_channel_rows: list[dict], plot_info: mne.Info, out_dir: Path, band_name: str) -> Path:
    """Saves group-level interbrain synchrony topomap triplet."""
    out_path = out_dir / f"interbrain_synchrony_group_topomaps_{band_name}.png"
    return _save_topomap_triplet(
        rows=group_channel_rows,
        plot_info=plot_info,
        title=f"Group Interbrain Synchrony Topomaps ({band_name})",
        out_path=out_path,
    )


def _save_subject_topomap(
    subject_id: str,
    subject_channel_rows: list[dict],
    plot_info: mne.Info,
    out_dir: Path,
    band_name: str,
) -> Path:
    """Saves subject-level interbrain synchrony topomap triplet."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"sub-{subject_id}_interbrain_synchrony_topomaps_{band_name}.png"
    return _save_topomap_triplet(
        rows=subject_channel_rows,
        plot_info=plot_info,
        title=f"Interbrain Synchrony Topomaps sub-{subject_id} ({band_name})",
        out_path=out_path,
    )


def _save_zscore_plot(subject_rows: list[dict], out_dir: Path, band_name: str) -> Path:
    """Saves per-subject global PLV z-score bar chart for one band."""
    rows = sorted(subject_rows, key=lambda row: str(row["subject"]))
    z = np.asarray([float(row["global_plv_z"]) for row in rows], dtype=float)

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--")
    ax.bar(np.arange(len(z)), z, color="#1976d2", edgecolor="black", linewidth=0.6)
    ax.set_xticks(np.arange(len(z)))
    ax.set_xticklabels([f"sub-{row['subject']}" for row in rows], rotation=45, ha="right")
    ax.set_ylabel("Global PLV z-score vs shuffled")
    ax.set_title(f"Interbrain Synchrony Strength by Pair ({band_name})")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()

    out_path = out_dir / f"interbrain_synchrony_subject_zscores_{band_name}.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _save_combined_zscore_plot(all_subject_rows: list[dict], out_dir: Path) -> Path:
    """
    Saves a multi-panel z-score overview across all frequency bands.

    Gold borders highlight subjects with permutation p < 0.05.
    """
    band_colors = {"theta": "#FF6B6B", "alpha": "#4ECDC4", "beta": "#45B7D1"}
    
    bands_present = sorted(set(row.get("band", "unknown") for row in all_subject_rows))
    subject_ids = sorted(set(str(row["subject"]) for row in all_subject_rows), key=lambda s: int(s) if s.isdigit() else 999)
    
    n_subjects = len(subject_ids)
    n_bands = len(bands_present)
    fig, axes = plt.subplots(1, n_bands, figsize=(5.5 * n_bands, 5.2), sharey=True)
    if n_bands == 1:
        axes = [axes]
    
    for ax_idx, band_name in enumerate(bands_present):
        band_rows = sorted(
            [row for row in all_subject_rows if row.get("band") == band_name],
            key=lambda row: str(row["subject"])
        )
        
        subject_labels = [f"sub-{row['subject']}" for row in band_rows]
        z_scores = np.asarray([float(row["global_plv_z"]) for row in band_rows], dtype=float)
        p_values = np.asarray([float(row["global_plv_perm_p"]) for row in band_rows], dtype=float)
        
        colors = [band_colors.get(band_name, "#1976d2") for _ in range(len(z_scores))]
        
        ax = axes[ax_idx]
        ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--", label="No effect")
        ax.axhline(1.96, color="green", linewidth=1.5, linestyle=":", alpha=0.7, label="Strong (p<0.05)")
        ax.axhline(-1.96, color="orange", linewidth=1.5, linestyle=":", alpha=0.7)
        
        bars = ax.bar(
            np.arange(len(z_scores)),
            z_scores,
            color=colors,
            edgecolor="black",
            linewidth=0.6,
            alpha=0.8
        )
        
        for idx, (bar, p_val) in enumerate(zip(bars, p_values)):
            if p_val < 0.05:
                bar.set_edgecolor("gold")
                bar.set_linewidth(2.5)
        
        ax.set_xticks(np.arange(len(subject_labels)))
        ax.set_xticklabels(subject_labels, rotation=45, ha="right", fontsize=9)
        ax.set_title(f"{band_name.capitalize()} Band", fontweight="bold")
        ax.grid(axis="y", alpha=0.25)
        
        if ax_idx == 0:
            ax.set_ylabel("Global PLV z-score vs shuffled", fontweight="bold")
            ax.legend(loc="upper left", fontsize=9)
    
    fig.suptitle("Interbrain Synchrony Strength Across Frequency Bands (gold border=p<0.05)", 
                 fontsize=12, fontweight="bold", y=1.00)
    fig.tight_layout()
    
    out_path = out_dir / "interbrain_synchrony_subject_zscores_combined.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _save_band_aggregate_plot(group_summaries: list[dict], out_dir: Path) -> Path:
    """Saves one aggregated z-score bar per frequency band."""
    preferred_order = {"theta": 0, "alpha": 1, "beta": 2}
    ordered = sorted(
        group_summaries,
        key=lambda row: (preferred_order.get(str(row.get("band", "")).lower(), 999), str(row.get("band", ""))),
    )

    band_labels = [str(row["band"]).capitalize() for row in ordered]
    agg_z = np.asarray([float(row["mean_global_z"]) for row in ordered], dtype=float)

    fig, ax = plt.subplots(figsize=(7.8, 5.2))
    x = np.arange(len(agg_z))
    bars = ax.bar(x, agg_z, color=["#FF6B6B", "#4ECDC4", "#45B7D1"][: len(agg_z)], edgecolor="black", linewidth=0.8)
    ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--")
    ax.axhline(1.96, color="green", linewidth=1.4, linestyle=":", alpha=0.8, label="Threshold +1.96")
    ax.axhline(-1.96, color="orange", linewidth=1.4, linestyle=":", alpha=0.8, label="Threshold -1.96")
    ax.set_xticks(x)
    ax.set_xticklabels(band_labels)
    ax.set_ylabel("Aggregated global PLV z-score (mean across subjects)")
    ax.set_title("Interbrain Synchrony Aggregated by Frequency Band")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper right", fontsize=9)

    for bar, value in zip(bars, agg_z):
        y = value + 0.05 if value >= 0 else value - 0.10
        ax.text(bar.get_x() + bar.get_width() / 2.0, y, f"{value:.2f}", ha="center", va="bottom" if value >= 0 else "top")

    fig.tight_layout()
    out_path = out_dir / "interbrain_synchrony_band_aggregated_zscores.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _prepare_topomap_info(info: mne.Info) -> mne.Info:
    """Prepares EEG-only info and ensures valid sensor geometry for topomaps."""
    info_plot = info.copy()
    eeg_picks = mne.pick_types(info_plot, eeg=True, exclude=[])
    if len(eeg_picks) == 0:
        raise RuntimeError("No EEG channels available for interbrain topomap plotting.")

    info_plot = mne.pick_info(info_plot, eeg_picks, copy=True)
    has_dig = info_plot.get("dig") is not None and len(info_plot["dig"]) > 0
    if not has_dig:
        montage = mne.channels.make_standard_montage("biosemi64", head_size=0.105)
        info_plot.set_montage(montage, match_case=False, on_missing="ignore")
    return info_plot


def run_interbrain_synchrony(
    subjects: list[str],
    tmin: float,
    tmax: float,
    fmin: float,
    fmax: float,
    n_shuffles: int,
    random_state: int,
    band_name: str,
    subject_topomap_dir: Path | None = None,
) -> tuple[list[dict], list[dict], dict, list[dict], mne.Info, list[Path]]:
    """
    Runs interbrain synchrony analysis for one frequency band.

    Args:
        subjects (list[str]): Subject IDs to process.
        tmin (float): Analysis-window start in seconds.
        tmax (float): Analysis-window end in seconds.
        fmin (float): Band-pass lower edge in Hz.
        fmax (float): Band-pass upper edge in Hz.
        n_shuffles (int): Number of shuffled controls.
        random_state (int): Random seed.
        band_name (str): Frequency-band label for reporting.
        subject_topomap_dir (Path | None): Optional output dir for subject topomaps.

    Returns:
        tuple[list[dict], list[dict], dict, list[dict], mne.Info, list[Path]]:
            - All per-subject channel rows.
            - Subject summaries.
            - Group summary for this band.
            - Group channel averages.
            - Plot info for topomaps.
            - Saved subject topomap paths.
    """
    all_channel_rows: list[dict] = []
    subject_summaries: list[dict] = []
    subject_topomap_paths: list[Path] = []
    representative_info: mne.Info | None = None

    with Live(progress, console=console, refresh_per_second=1) as live:
        task_id = progress.add_task(f"Interbrain synchrony ({band_name})", total=len(subjects))

        for subject_id in subjects:
            try:
                progress.update(task_id, description=f"{band_name}: sub-{subject_id}")
                live.refresh()

                channel_rows, summary, info = _summarize_subject(
                    subject_id=subject_id,
                    tmin=tmin,
                    tmax=tmax,
                    fmin=fmin,
                    fmax=fmax,
                    n_shuffles=n_shuffles,
                    random_state=random_state,
                )
                all_channel_rows.extend(channel_rows)
                subject_summaries.append(summary)
                if representative_info is None:
                    representative_info = info

                if subject_topomap_dir is not None:
                    ordered_channels = [str(row["channel"]) for row in channel_rows]
                    picks = mne.pick_channels(info["ch_names"], include=ordered_channels, ordered=True)
                    subject_plot_info = mne.pick_info(info.copy(), picks)
                    subject_plot_info = _prepare_topomap_info(subject_plot_info)
                    subject_topomap_paths.append(
                        _save_subject_topomap(
                            subject_id=subject_id,
                            subject_channel_rows=channel_rows,
                            plot_info=subject_plot_info,
                            out_dir=subject_topomap_dir,
                            band_name=band_name,
                        )
                    )

                console.print(
                    f"[green]✓[/green] sub-{subject_id} [{band_name}]: "
                    f"real={summary['global_plv_real']:.4f}, "
                    f"shuffle={summary['global_plv_shuffle_mean']:.4f}, "
                    f"z={summary['global_plv_z']:.3f}, p={summary['global_plv_perm_p']:.4f}"
                )
            except Exception as exc:
                console.print(f"[yellow]⚠[/yellow] sub-{subject_id} [{band_name}]: skipped ({exc})")

            progress.advance(task_id)
            live.refresh()

    if not all_channel_rows or representative_info is None:
        raise RuntimeError("No valid interbrain synchrony results were produced.")

    subject_summaries = sorted(subject_summaries, key=lambda row: str(row["subject"]))
    group_channel_rows = _group_channel_rows(all_channel_rows)

    p = np.asarray([float(row["global_plv_perm_p"]) for row in subject_summaries], dtype=float)
    p_fdr = _fdr_bh(p)

    for row, p_corr in zip(subject_summaries, p_fdr):
        row["band"] = band_name
        row["global_plv_perm_p_fdr"] = float(p_corr)

    for row in all_channel_rows:
        row["band"] = band_name

    for row in group_channel_rows:
        row["band"] = band_name

    real = np.asarray([float(row["global_plv_real"]) for row in subject_summaries], dtype=float)
    shuf = np.asarray([float(row["global_plv_shuffle_mean"]) for row in subject_summaries], dtype=float)
    z = np.asarray([float(row["global_plv_z"]) for row in subject_summaries], dtype=float)

    group_summary = {
        "band": band_name,
        "n_subjects": int(len(subject_summaries)),
        "fmin": float(fmin),
        "fmax": float(fmax),
        "tmin": float(tmin),
        "tmax": float(tmax),
        "n_shuffles": int(n_shuffles),
        "mean_global_plv_real": float(np.mean(real)),
        "mean_global_plv_shuffle": float(np.mean(shuf)),
        "mean_global_plv_delta": float(np.mean(real - shuf)),
        "mean_global_z": float(np.nanmean(z)),
        "subjects_with_perm_p_lt_0_05": int(np.sum(p < 0.05)),
        "subjects_with_perm_p_fdr_lt_0_05": int(np.sum(p_fdr < 0.05)),
    }

    ordered_channels = [str(row["channel"]) for row in group_channel_rows]
    picks = mne.pick_channels(representative_info["ch_names"], include=ordered_channels, ordered=True)
    plot_info = mne.pick_info(representative_info.copy(), picks)
    plot_info = _prepare_topomap_info(plot_info)

    return all_channel_rows, subject_summaries, group_summary, group_channel_rows, plot_info, subject_topomap_paths


def main() -> None:
    """
    CLI entry point for interbrain synchrony analysis and plotting.

    Returns:
        None
    """
    mne.set_config("MNE_LOGGING_LEVEL", "ERROR")

    parser = argparse.ArgumentParser(
        description=(
            "Compute interbrain phase-locking value (PLV) between P1 and P2 during decision phase, "
            "with shuffled-trial control and verification plots."
        )
    )
    parser.add_argument("--subjects", type=str, default=None, help="Comma-separated subject IDs, e.g. 01,02,03")
    parser.add_argument("--tmin", type=float, default=0.0, help="Start time of decision window in seconds")
    parser.add_argument("--tmax", type=float, default=2.0, help="End time of decision window in seconds")
    parser.add_argument(
        "--bands",
        type=str,
        default=None,
        help=(
            "Comma-separated bands as name:fmin-fmax, e.g. theta:4-7,alpha:8-12,beta:13-30. "
            "If omitted, defaults to theta/alpha/beta."
        ),
    )
    parser.add_argument("--fmin", type=float, default=None, help="Legacy single-band lower edge in Hz")
    parser.add_argument("--fmax", type=float, default=None, help="Legacy single-band upper edge in Hz")
    parser.add_argument("--n-shuffles", type=int, default=200, help="Number of shuffled controls per subject")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    subjects = _resolve_subjects(args.subjects)

    if args.bands:
        bands = _parse_bands(args.bands)
    elif args.fmin is not None or args.fmax is not None:
        fmin = float(config.IBS_FMIN) if args.fmin is None else float(args.fmin)
        fmax = float(config.IBS_FMAX) if args.fmax is None else float(args.fmax)
        if fmax <= fmin:
            raise ValueError("--fmax must be greater than --fmin.")
        bands = {"custom": (fmin, fmax)}
    else:
        bands = dict(DEFAULT_BANDS)

    out_dir = Path(config.OUTPUT_DIR).parent / "decoding"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_channel_rows: list[dict] = []
    all_subject_rows: list[dict] = []
    group_summaries: list[dict] = []
    subject_plot_paths: list[Path] = []
    topomap_plot_paths: list[Path] = []
    zscore_plot_paths: list[Path] = []
    subject_topomap_paths: list[Path] = []
    subject_topomap_dir = out_dir / "interbrain_synchrony_subject_topomaps"

    for band_name, (fmin, fmax) in bands.items():
        channel_rows, subject_rows, group_summary, group_channel_rows, plot_info, band_subject_topomaps = run_interbrain_synchrony(
            subjects=subjects,
            tmin=args.tmin,
            tmax=args.tmax,
            fmin=fmin,
            fmax=fmax,
            n_shuffles=args.n_shuffles,
            random_state=args.random_state,
            band_name=band_name,
            subject_topomap_dir=subject_topomap_dir,
        )

        all_channel_rows.extend(channel_rows)
        all_subject_rows.extend(subject_rows)
        group_summaries.append(group_summary)
        subject_topomap_paths.extend(band_subject_topomaps)

        subject_plot_paths.append(_save_subject_global_plot(subject_rows=subject_rows, out_dir=out_dir, band_name=band_name))
        topomap_plot_paths.append(
            _save_group_topomap(
                group_channel_rows=group_channel_rows,
                plot_info=plot_info,
                out_dir=out_dir,
                band_name=band_name,
            )
        )
        zscore_plot_paths.append(_save_zscore_plot(subject_rows=subject_rows, out_dir=out_dir, band_name=band_name))

    combined_zscore_plot_path = _save_combined_zscore_plot(all_subject_rows, out_dir)
    aggregate_band_plot_path = _save_band_aggregate_plot(group_summaries, out_dir)

    channel_path = out_dir / "interbrain_synchrony_channel_values.csv"
    subject_path = out_dir / "interbrain_synchrony_subject_summary.csv"
    group_path = out_dir / "interbrain_synchrony_group_summary.json"
    band_aggregate_path = out_dir / "interbrain_synchrony_band_aggregates.csv"

    _write_csv(
        channel_path,
        all_channel_rows,
        fieldnames=["subject", "band", "channel", "plv_real", "plv_shuffle_mean", "plv_delta"],
    )
    _write_csv(
        subject_path,
        all_subject_rows,
        fieldnames=[
            "subject",
            "band",
            "n_trials_used",
            "n_channels",
            "n_times",
            "sfreq",
            "fmin",
            "fmax",
            "tmin",
            "tmax",
            "n_shuffles",
            "global_plv_real",
            "global_plv_shuffle_mean",
            "global_plv_shuffle_std",
            "global_plv_z",
            "global_plv_perm_p",
            "global_plv_perm_p_fdr",
        ],
    )

    group_payload = {
        "n_bands": int(len(group_summaries)),
        "n_subjects_requested": int(len(subjects)),
        "tmin": float(args.tmin),
        "tmax": float(args.tmax),
        "n_shuffles": int(args.n_shuffles),
        "bands": group_summaries,
    }
    group_path.write_text(json.dumps(group_payload, indent=2), encoding="utf-8")
    _write_csv(
        band_aggregate_path,
        group_summaries,
        fieldnames=[
            "band",
            "n_subjects",
            "fmin",
            "fmax",
            "tmin",
            "tmax",
            "n_shuffles",
            "mean_global_plv_real",
            "mean_global_plv_shuffle",
            "mean_global_plv_delta",
            "mean_global_z",
            "subjects_with_perm_p_lt_0_05",
            "subjects_with_perm_p_fdr_lt_0_05",
        ],
    )

    console.print("\n=== Interbrain Synchrony Summary ===")
    for summary in group_summaries:
        console.print(
            f"[{summary['band']}] N={summary['n_subjects']} | "
            f"PLV real={summary['mean_global_plv_real']:.4f}, "
            f"shuffle={summary['mean_global_plv_shuffle']:.4f}, "
            f"delta={summary['mean_global_plv_delta']:.4f}, "
            f"perm p<0.05={summary['subjects_with_perm_p_lt_0_05']}, "
            f"FDR p<0.05={summary['subjects_with_perm_p_fdr_lt_0_05']}"
        )

    console.print(f"Saved channel values: {channel_path}")
    console.print(f"Saved subject summary: {subject_path}")
    console.print(f"Saved group summary: {group_path}")
    console.print(f"Saved band aggregates: {band_aggregate_path}")
    console.print("Saved subject plots:")
    for path in subject_plot_paths:
        console.print(f"  - {path}")
    console.print("Saved topomap plots:")
    for path in topomap_plot_paths:
        console.print(f"  - {path}")
    console.print("Saved subject topomap plots:")
    for path in subject_topomap_paths:
        console.print(f"  - {path}")
    console.print("Saved z-score plots:")
    for path in zscore_plot_paths:
        console.print(f"  - {path}")
    console.print(f"Saved combined z-score plot: {combined_zscore_plot_path}")
    console.print(f"Saved band aggregate plot: {aggregate_band_plot_path}")


if __name__ == "__main__":
    main()
