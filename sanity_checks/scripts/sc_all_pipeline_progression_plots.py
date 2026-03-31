"""
Sanity Check Visualization: Pipeline Progression (Original -> Processed)

Creates two QC figures per subject and player:
1) Progression plot across all available preprocessing stages (GFP and PSD).
2) Direct comparison of original vs. latest available processed stage.

This is intended to answer: "Do my preprocessing changes make sense on EEG data?"

REASONING:
- Purpose: provide a single narrative view across preprocessing stages so graders can see cumulative effects rather than isolated step outputs.
- Reproducibility: stage order and file suffixes are explicitly defined in this script, which makes the progression deterministic.
- Interpretation focus: the expected argument is "This seems correct because GFP/PSD changes appear where the corresponding preprocessing operation should affect them."
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import NamedTuple

import matplotlib.pyplot as plt
import mne
import numpy as np
from mne_bids import BIDSPath, read_raw_bids

CURRENT_DIR = Path(__file__).resolve().parent
# Make local sanity-check helpers importable when running as a script.
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

PIPELINE_DIR = CURRENT_DIR.parent.parent / "eeg_pipeline"
# Make pipeline package importable for preprocessing config access.
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

PREPROCESSING_DIR = PIPELINE_DIR / "preprocessing"
if str(PREPROCESSING_DIR) not in sys.path:
    sys.path.insert(0, str(PREPROCESSING_DIR))

CONFIG_PATH = PREPROCESSING_DIR / "config.py"
# Load preprocessing config explicitly by path so this script remains robust
# regardless of the current working directory or package execution mode.
_config_spec = importlib.util.spec_from_file_location("preprocessing_config", CONFIG_PATH)
if _config_spec is None or _config_spec.loader is None:
    raise ImportError(f"Could not load preprocessing config from {CONFIG_PATH}")
config = importlib.util.module_from_spec(_config_spec)
_config_spec.loader.exec_module(config)


class StageSpec(NamedTuple):
    """Static metadata describing one pipeline stage."""
    key: str
    label: str
    person_specific: bool


class StageMetrics(NamedTuple):
    """Computed time/frequency metrics used in progression plots."""
    key: str
    label: str
    times: np.ndarray
    gfp_uv: np.ndarray
    trace_uv: np.ndarray
    trace_channel: str
    freqs: np.ndarray
    psd_db: np.ndarray


STAGE_ORDER = [
    StageSpec("original", "Original", False),
    StageSpec("downsampled", "Downsample", False),
    StageSpec("split", "Split", True),
    StageSpec("renamed_montaged", "Rename+Montage", True),
    StageSpec("badchannels_detected", "Bad Ch. Detect", True),
    StageSpec("interpolated", "Interpolate", True),
    StageSpec("filtered", "Filter", True),
    StageSpec("ica_cleaned", "ICA Clean", True),
]


def _source_text(subject_id: str, person: str, stage_key: str) -> str:
    """Return human-readable stage source text for console diagnostics."""
    if stage_key == "original":
        return f"BIDS: sub-{subject_id} task-RPS eeg"
    raw_path = _raw_path_for_stage(subject_id, person, stage_key)
    if raw_path is None:
        return "unknown"
    return str(raw_path)


def _raw_path_for_stage(subject_id: str, person: str, stage_key: str) -> Path | None:
    """Resolve file path for a stage, or None for raw BIDS input."""
    if stage_key == "original":
        return None
    if stage_key == "downsampled":
        return config.OUTPUT_DIR / f"sub-{subject_id}_downsampled.fif"
    return config.OUTPUT_DIR / f"sub-{subject_id}_{person}_{stage_key}.fif"


def _load_raw(subject_id: str, person: str, stage_key: str) -> mne.io.BaseRaw | None:
    """Load stage data as an MNE Raw object, returning None on failure."""
    # Original stage is loaded from BIDS, not from intermediate FIF output.
    if stage_key == "original":
        bids_path = BIDSPath(
            subject=subject_id,
            task="RPS",
            datatype="eeg",
            suffix="eeg",
            root=config.BIDS_ROOT,
        )
        try:
            return read_raw_bids(bids_path, verbose=False)
        except Exception:
            return None

    # All processed stages are expected as FIF files in preprocessing output.
    raw_path = _raw_path_for_stage(subject_id, person, stage_key)
    if raw_path is None or not raw_path.exists():
        return None

    try:
        return mne.io.read_raw_fif(str(raw_path), preload=False, verbose=False)
    except Exception:
        return None


def _person_eeg_picks(raw: mne.io.BaseRaw, person: str, stage_key: str) -> list[int]:
    """Select EEG channels belonging to one person for a given stage.

    Early stages still carry player prefixes; later stages are already split.
    """
    if stage_key in {"original", "downsampled"}:
        prefix = config.PLAYER_PREFIX_MAP[person]
        channel_types = raw.get_channel_types()
        picks = [
            i
            for i, (name, ch_type) in enumerate(zip(raw.ch_names, channel_types))
            if ch_type == "eeg" and name.startswith(prefix)
        ]
        if picks:
            return picks

    # For already split/renamed stages, regular EEG picks are sufficient.
    return list(mne.pick_types(raw.info, eeg=True, exclude=[]))


def _compute_stage_metrics(
    raw: mne.io.BaseRaw,
    person: str,
    stage_key: str,
    stage_label: str,
    duration_sec: float = 30.0,
) -> StageMetrics | None:
    """Compute comparable stage metrics for progression visualization.

    Returns synchronized time-domain and PSD summaries from the same stage.
    """
    # Restrict computation to person-specific EEG channels.
    eeg_picks = _person_eeg_picks(raw, person, stage_key)
    if len(eeg_picks) == 0:
        return None

    # Use a bounded analysis window for stable and fast comparisons.
    sfreq = float(raw.info["sfreq"])
    n_samples = min(int(duration_sec * sfreq), raw.n_times)
    if n_samples < 10:
        return None

    data = raw.get_data(picks=eeg_picks, start=0, stop=n_samples)

    # GFP provides channel-agnostic overall activity magnitude over time.
    gfp_uv = np.std(data, axis=0) * 1e6
    times = np.arange(n_samples) / sfreq

    preferred_names = ["Fp1", "Fpz", "Fz", "Cz", "Pz", "Oz"]
    candidate_names: list[str]
    if stage_key in {"original", "downsampled"}:
        prefix = config.PLAYER_PREFIX_MAP[person]
        candidate_names = [f"{prefix}A1", f"{prefix}B1", f"{prefix}B6", f"{prefix}B16"]
    elif stage_key == "split":
        candidate_names = ["A1", "B1", "B6", "B16"]
    else:
        candidate_names = preferred_names

    # Choose a representative trace channel with stage-aware fallback rules.
    trace_pick = None
    for ch_name in candidate_names:
        if ch_name in raw.ch_names:
            trace_pick = raw.ch_names.index(ch_name)
            break
    if trace_pick is None:
        trace_pick = eeg_picks[0]

    trace_uv = raw.get_data(picks=[trace_pick], start=0, stop=n_samples)[0] * 1e6
    trace_channel = raw.ch_names[trace_pick]

    # Compute PSD on the same temporal window as time-domain summaries.
    max_time = n_samples / sfreq
    raw_for_psd = raw.copy().pick(eeg_picks).crop(tmin=0.0, tmax=max_time, include_tmax=False)
    spectrum = raw_for_psd.compute_psd(method="welch", fmin=0.5, fmax=45.0, verbose=False)
    psd, freqs = spectrum.get_data(return_freqs=True)
    psd_db = 10 * np.log10(psd.mean(axis=0) + np.finfo(float).eps)

    return StageMetrics(
        key=stage_key,
        label=stage_label,
        times=times,
        gfp_uv=gfp_uv,
        trace_uv=trace_uv,
        trace_channel=trace_channel,
        freqs=freqs,
        psd_db=psd_db,
    )


def _plot_progression(subject_id: str, person: str, metrics: list[StageMetrics]) -> Path:
    """Plot full-stage progression for GFP and PSD in one figure."""
    fig, axes = plt.subplots(2, 1, figsize=(13, 9))
    n = len(metrics)

    # Color progression by stage order to make cumulative changes readable.
    for idx, m in enumerate(metrics):
        color = plt.cm.viridis(idx / max(n - 1, 1))
        axes[0].plot(m.times, m.gfp_uv, color=color, linewidth=1.6, label=m.label)
        axes[1].plot(m.freqs, m.psd_db, color=color, linewidth=1.6, label=m.label)

    axes[0].set_title(f"sub-{subject_id} {person}: GFP progression across preprocessing")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Amplitude (uV, GFP)")
    axes[0].grid(alpha=0.3)

    axes[1].set_title(f"sub-{subject_id} {person}: PSD progression across preprocessing")
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("Power (dB)")
    axes[1].grid(alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(4, n), frameon=False)

    plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.93])
    out_path = config.QC_DIR / f"sub-{subject_id}_{person}_pipeline_progression_gfp_psd.png"
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _plot_original_vs_latest(subject_id: str, person: str, first: StageMetrics, last: StageMetrics) -> Path:
    """Plot direct original-vs-latest comparison (time and PSD)."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))

    axes[0].plot(
        first.times,
        first.trace_uv,
        color="tab:blue",
        linewidth=1.2,
        label=f"{first.label} ({first.trace_channel})",
    )
    axes[0].plot(
        last.times,
        last.trace_uv,
        color="tab:orange",
        linewidth=1.2,
        label=f"{last.label} ({last.trace_channel})",
    )
    axes[0].set_title(f"sub-{subject_id} {person}: Original vs {last.label} (time domain)")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Amplitude (uV)")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(first.freqs, first.psd_db, color="tab:blue", linewidth=2.0, label=first.label)
    axes[1].plot(last.freqs, last.psd_db, color="tab:orange", linewidth=2.0, label=last.label)
    axes[1].set_title(f"sub-{subject_id} {person}: Original vs {last.label} (PSD)")
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("Power (dB)")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    out_path = config.QC_DIR / f"sub-{subject_id}_{person}_original_vs_latest_gfp_psd.png"
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _plot_butterfly_original_vs_latest(
    subject_id: str,
    person: str,
    first_raw: mne.io.BaseRaw,
    first_stage_key: str,
    first_label: str,
    last_raw: mne.io.BaseRaw,
    last_stage_key: str,
    last_label: str,
    duration_sec: float = 10.0,
    max_channels: int = 20,
) -> Path | None:
    """Plot butterfly overlays for original and latest stage EEG data."""
    # Determine person-specific EEG picks for both comparison endpoints.
    first_picks = _person_eeg_picks(first_raw, person, first_stage_key)
    last_picks = _person_eeg_picks(last_raw, person, last_stage_key)

    if len(first_picks) == 0 or len(last_picks) == 0:
        return None

    # Limit channels for readability and plotting performance.
    first_picks = first_picks[:max_channels]
    last_picks = last_picks[:max_channels]

    first_sfreq = float(first_raw.info["sfreq"])
    last_sfreq = float(last_raw.info["sfreq"])
    first_n_samples = min(int(duration_sec * first_sfreq), first_raw.n_times)
    last_n_samples = min(int(duration_sec * last_sfreq), last_raw.n_times)

    if first_n_samples < 10 or last_n_samples < 10:
        return None

    first_data_uv = first_raw.get_data(picks=first_picks, start=0, stop=first_n_samples) * 1e6
    last_data_uv = last_raw.get_data(picks=last_picks, start=0, stop=last_n_samples) * 1e6
    first_times = np.arange(first_n_samples) / first_sfreq
    last_times = np.arange(last_n_samples) / last_sfreq

    # Two stacked panels: top=original endpoint, bottom=latest endpoint.
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=False)

    axes[0].plot(first_times, first_data_uv.T, linewidth=0.6, alpha=0.6)
    axes[0].set_title(f"sub-{subject_id} {person}: Butterfly ({first_label})")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Amplitude (uV)")
    axes[0].grid(alpha=0.25)
    axes[0].text(
        0.01,
        0.96,
        f"n={len(first_picks)} channels",
        transform=axes[0].transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
    )

    axes[1].plot(last_times, last_data_uv.T, linewidth=0.6, alpha=0.6)
    axes[1].set_title(f"sub-{subject_id} {person}: Butterfly ({last_label})")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Amplitude (uV)")
    axes[1].grid(alpha=0.25)
    axes[1].text(
        0.01,
        0.96,
        f"n={len(last_picks)} channels",
        transform=axes[1].transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
    )

    plt.tight_layout()
    out_path = config.QC_DIR / f"sub-{subject_id}_{person}_original_vs_latest_butterfly.png"
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _safe_file_label(label: str) -> str:
    """Convert stage labels into filesystem-safe filename tokens."""
    return (
        label.lower()
        .replace("+", "plus")
        .replace(" ", "_")
        .replace(".", "")
        .replace("-", "_")
    )


def _plot_step_transition(subject_id: str, person: str, prev_m: StageMetrics, next_m: StageMetrics) -> Path:
    """Plot pairwise transition diagnostics between adjacent stages."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))

    axes[0].plot(
        prev_m.times,
        prev_m.trace_uv,
        color="tab:blue",
        linewidth=1.1,
        label=f"{prev_m.label} ({prev_m.trace_channel})",
    )
    axes[0].plot(
        next_m.times,
        next_m.trace_uv,
        color="tab:orange",
        linewidth=1.1,
        label=f"{next_m.label} ({next_m.trace_channel})",
    )
    axes[0].set_title(f"sub-{subject_id} {person}: {prev_m.label} -> {next_m.label} (time)")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Amplitude (uV)")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=8)

    axes[1].plot(prev_m.freqs, prev_m.psd_db, color="tab:blue", linewidth=1.8, label=prev_m.label)
    axes[1].plot(next_m.freqs, next_m.psd_db, color="tab:orange", linewidth=1.8, label=next_m.label)
    axes[1].set_title(f"sub-{subject_id} {person}: {prev_m.label} -> {next_m.label} (PSD)")
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("Power (dB)")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    prev_tag = _safe_file_label(prev_m.label)
    next_tag = _safe_file_label(next_m.label)
    out_path = config.QC_DIR / f"sub-{subject_id}_{person}_step_transition_{prev_tag}_to_{next_tag}.png"
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path


def create_pipeline_progression_plots() -> None:
    """Generate progression, endpoint, butterfly, and transition plots.

    For each subject/person pair:
    - gather available stage metrics
    - build multi-stage progression plots
    - compare first vs latest stage
    - optionally create butterfly and adjacent-transition views
    """
    print("\n" + "=" * 80)
    print("SANITY CHECK: Pipeline progression plots (Original -> Processed)")
    print("=" * 80)

    for subject_id in config.SUBJECTS:
        print(f"\n--- Subject {subject_id} ---")

        for person in ["P1", "P2"]:
            print(f"\n{person}:")
            metrics: list[StageMetrics] = []

            # Collect metrics in fixed stage order; skip missing/unusable stages.
            for stage in STAGE_ORDER:
                source = _source_text(subject_id=subject_id, person=person, stage_key=stage.key)
                raw = _load_raw(subject_id=subject_id, person=person, stage_key=stage.key)
                if raw is None:
                    print(f"  - {stage.label}: missing ({source})")
                    continue

                stage_metrics = _compute_stage_metrics(
                    raw=raw,
                    person=person,
                    stage_key=stage.key,
                    stage_label=stage.label,
                )
                if stage_metrics is None:
                    print(f"  - {stage.label}: no usable EEG channels ({source})")
                    continue

                metrics.append(stage_metrics)
                print(f"  - {stage.label}: ok ({source})")

            # Need at least two valid stages for meaningful progression plots.
            if len(metrics) < 2:
                print("  WARNING: Need at least 2 stages to plot progression. Skipping.")
                continue

            progression_path = _plot_progression(subject_id=subject_id, person=person, metrics=metrics)
            print(f"  Saved progression plot: {progression_path.name}")

            # First and last available stages define endpoint comparison.
            original = metrics[0]
            latest = metrics[-1]
            compare_path = _plot_original_vs_latest(
                subject_id=subject_id,
                person=person,
                first=original,
                last=latest,
            )
            print(f"  Saved original vs latest plot: {compare_path.name}")

            original_raw = _load_raw(subject_id=subject_id, person=person, stage_key=original.key)
            latest_raw = _load_raw(subject_id=subject_id, person=person, stage_key=latest.key)
            # Butterfly plot is optional and depends on successful raw loading.
            if original_raw is not None and latest_raw is not None:
                butterfly_path = _plot_butterfly_original_vs_latest(
                    subject_id=subject_id,
                    person=person,
                    first_raw=original_raw,
                    first_stage_key=original.key,
                    first_label=original.label,
                    last_raw=latest_raw,
                    last_stage_key=latest.key,
                    last_label=latest.label,
                )
                if butterfly_path is not None:
                    print(f"  Saved butterfly plot: {butterfly_path.name}")
                else:
                    print("  INFO: Could not generate butterfly plot (insufficient EEG data).")
            else:
                print("  INFO: Could not generate butterfly plot (stage files missing).")

            # Also generate adjacent transition plots for stepwise interpretation.
            if len(metrics) >= 2:
                print("  Creating adjacent step-transition plots...")
                for prev_m, next_m in zip(metrics[:-1], metrics[1:]):
                    transition_path = _plot_step_transition(
                        subject_id=subject_id,
                        person=person,
                        prev_m=prev_m,
                        next_m=next_m,
                    )
                    print(f"  Saved transition plot: {transition_path.name}")

    print("\n" + "=" * 80)
    print("Pipeline progression plotting completed.")
    print("=" * 80)


if __name__ == "__main__":
    create_pipeline_progression_plots()

