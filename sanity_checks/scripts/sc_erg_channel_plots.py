"""
Sanity Check: Plot ERG channels for reporting.

Creates report-friendly plots that show what ERG channels contain:
- Time series (first N seconds)
- Power spectral density (PSD)
- Simple high-amplitude event markers (quantile-based)

Outputs are written to:
  output/preprocessing/erg_eog_check/plots/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

PIPELINE_DIR = CURRENT_DIR.parent.parent / "eeg_pipeline"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from preprocessing import config


FILE_SUFFIX_PRIORITY = [
    "filtered",
    "interpolated",
    "badchannels_detected",
    "renamed_montaged",
]

COMPARISON_STAGE_ORDER = [
    "renamed_montaged",
    "badchannels_detected",
    "interpolated",
    "filtered",
]

FRONTAL_CANDIDATES = ["Fp1", "Fpz", "Fp2", "AF7", "AF8", "AF3", "AF4", "AFz", "Fz"]
BLINK_MIN_MS = 80.0
BLINK_MAX_MS = 600.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create ERG channel plots for report/QC.")
    parser.add_argument(
        "--subjects",
        nargs="*",
        default=None,
        help="Optional subject IDs without 'sub-' prefix, e.g. 01 02 03",
    )
    parser.add_argument(
        "--persons",
        nargs="*",
        default=["P1", "P2"],
        choices=["P1", "P2"],
        help="Player streams to inspect (default: P1 P2)",
    )
    parser.add_argument(
        "--duration-seconds",
        type=float,
        default=60.0,
        help="Seconds shown in time-series panel (default: 60)",
    )
    parser.add_argument(
        "--peak-quantile",
        type=float,
        default=0.995,
        help="Quantile for event markers on abs(ERG) (default: 0.995)",
    )
    parser.add_argument(
        "--max-recordings",
        type=int,
        default=0,
        help="Optional limit for quick runs, 0 means no limit (default: 0)",
    )
    parser.add_argument(
        "--skip-comparison",
        action="store_true",
        help="Skip before/after comparison plots (enabled by default).",
    )
    return parser.parse_args()


def resolve_subjects(subject_args: list[str] | None) -> list[str]:
    if subject_args:
        return [value.replace("sub-", "") for value in subject_args]
    return list(config.SUBJECTS)


def find_input_file(subject_id: str, person: str) -> tuple[Path, str] | None:
    for suffix in FILE_SUFFIX_PRIORITY:
        path = config.OUTPUT_DIR / f"sub-{subject_id}_{person}_{suffix}.fif"
        if path.exists():
            return path, suffix
    return None


def _file_for_suffix(subject_id: str, person: str, suffix: str) -> Path:
    return config.OUTPUT_DIR / f"sub-{subject_id}_{person}_{suffix}.fif"


def find_comparison_pair(subject_id: str, person: str) -> tuple[Path, str, Path, str] | None:
    existing = []
    for suffix in COMPARISON_STAGE_ORDER:
        path = _file_for_suffix(subject_id, person, suffix)
        if path.exists():
            existing.append((suffix, path))

    if len(existing) < 2:
        return None

    # Use the latest and the directly preceding available stage.
    target_suffix, target_path = existing[-1]
    base_suffix, base_path = existing[-2]
    if base_suffix == target_suffix:
        return None
    if base_path.resolve() == target_path.resolve():
        return None
    return base_path, base_suffix, target_path, target_suffix


def count_peak_events(mask: np.ndarray) -> int:
    if mask.size == 0:
        return 0
    starts = np.sum((mask[1:] == 1) & (mask[:-1] == 0))
    return int(starts + (1 if mask[0] == 1 else 0))


def extract_peak_events(
    envelope_uv: np.ndarray,
    signal_uv: np.ndarray,
    threshold_uv: float,
    sfreq: float,
) -> list[dict[str, float]]:
    if envelope_uv.size < 3:
        return []

    events: list[dict[str, float]] = []

    candidate_peaks = np.where(
        (envelope_uv[1:-1] > envelope_uv[:-2])
        & (envelope_uv[1:-1] >= envelope_uv[2:])
        & (envelope_uv[1:-1] >= threshold_uv)
    )[0] + 1

    refractory_samples = max(1, int(round(0.20 * sfreq)))
    accepted_peaks: list[int] = []
    for peak in candidate_peaks:
        if not accepted_peaks or (peak - accepted_peaks[-1]) >= refractory_samples:
            accepted_peaks.append(int(peak))
        elif envelope_uv[peak] > envelope_uv[accepted_peaks[-1]]:
            accepted_peaks[-1] = int(peak)

    for peak_idx in accepted_peaks:
        half_level = max(threshold_uv, 0.5 * envelope_uv[peak_idx])

        left = peak_idx
        while left > 0 and envelope_uv[left] >= half_level:
            left -= 1
        right = peak_idx
        while right < (envelope_uv.size - 1) and envelope_uv[right] >= half_level:
            right += 1

        duration_ms = (right - left) / sfreq * 1000.0
        events.append(
            {
                "start_idx": float(left),
                "stop_idx": float(right),
                "peak_idx": float(peak_idx),
                "peak_uv": float(signal_uv[peak_idx]),
                "duration_ms": float(duration_ms),
                "blink_like": float(BLINK_MIN_MS <= duration_ms <= BLINK_MAX_MS),
            }
        )

    return events


def moving_average(signal: np.ndarray, window_samples: int) -> np.ndarray:
    if window_samples <= 1:
        return signal
    kernel = np.ones(window_samples, dtype=float) / float(window_samples)
    return np.convolve(signal, kernel, mode="same")


def get_erg_channels(raw: mne.io.BaseRaw) -> list[str]:
    bads = set(raw.info.get("bads", []))
    return [name for name in raw.ch_names if "erg" in name.lower() and name not in bads]


def get_frontal_channel(raw: mne.io.BaseRaw) -> str | None:
    bads = set(raw.info.get("bads", []))
    for channel in FRONTAL_CANDIDATES:
        if channel in raw.ch_names and channel not in bads:
            return channel
    return None


def _nice_unit_scale(signal_uv: np.ndarray) -> tuple[np.ndarray, str]:
    max_abs_uv = np.nanmax(np.abs(signal_uv)) if signal_uv.size else 0.0
    if max_abs_uv >= 1000.0:
        return signal_uv / 1000.0, "mV"
    return signal_uv, "uV"


def plot_erg_channel(
    raw: mne.io.BaseRaw,
    channel_name: str,
    *,
    duration_seconds: float,
    peak_quantile: float,
    title_prefix: str,
    output_path: Path,
) -> dict[str, float]:
    sfreq = float(raw.info["sfreq"])
    n_show = int(max(1, duration_seconds * sfreq))
    n_show = min(n_show, raw.n_times)

    data_volts = raw.get_data(picks=[channel_name], start=0, stop=n_show)[0]
    data_uv = data_volts * 1e6
    data_uv = data_uv - np.median(data_uv)

    abs_uv = np.abs(data_uv)
    smooth_window = max(3, int(round(0.12 * sfreq)))
    envelope_uv = moving_average(abs_uv, smooth_window)
    threshold = float(np.quantile(envelope_uv, peak_quantile))
    peak_mask = envelope_uv >= threshold
    peak_sample_count = int(np.sum(peak_mask))
    events = extract_peak_events(envelope_uv, data_uv, threshold, sfreq)
    peak_count = len(events)
    blink_like_events = [event for event in events if bool(event["blink_like"])]

    times = np.arange(n_show) / sfreq
    scaled, unit = _nice_unit_scale(data_uv)

    psd = mne.time_frequency.psd_array_welch(
        data_volts[np.newaxis, :],
        sfreq=sfreq,
        fmin=0.5,
        fmax=min(40.0, sfreq / 2.0 - 0.1),
        n_fft=min(2048, n_show),
        verbose="ERROR",
    )
    psd_values, freqs = psd
    psd_values = psd_values[0]
    psd_db = 10 * np.log10(np.maximum(psd_values, 1e-30))

    fig, axes = plt.subplots(4, 1, figsize=(12, 11), sharex=False)

    axes[0].plot(times, scaled, color="tab:blue", linewidth=1.0)
    for event in events:
        start_t = float(event["start_idx"]) / sfreq
        stop_t = float(event["stop_idx"]) / sfreq
        peak_t = float(event["peak_idx"]) / sfreq
        color = "tab:red" if bool(event["blink_like"]) else "tab:gray"
        alpha = 0.18 if bool(event["blink_like"]) else 0.08
        axes[0].axvspan(start_t, stop_t, color=color, alpha=alpha)
        axes[0].axvline(peak_t, color=color, alpha=0.45, linewidth=0.7)

    shown = min(10, len(events))
    if shown > 0:
        peak_indices = np.asarray([int(event["peak_idx"]) for event in events[:shown]], dtype=int)
        axes[0].scatter(
            times[peak_indices],
            scaled[peak_indices],
            color="tab:red",
            s=16,
            alpha=0.8,
            label=f"Event peaks (first {shown})",
        )
    axes[0].set_title(f"{title_prefix} | {channel_name} | Time series")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel(f"Amplitude ({unit})")
    axes[0].grid(alpha=0.3)
    axes[0].legend(loc="upper right")

    blink_ratio = (len(blink_like_events) / len(events)) if events else 0.0
    axes[0].text(
        0.01,
        0.97,
        f"event candidates={len(events)}, blink-like={len(blink_like_events)} ({blink_ratio:.0%})",
        transform=axes[0].transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
    )

    axes[1].plot(freqs, psd_db, color="tab:green", linewidth=1.5)
    axes[1].axvline(4.0, color="tab:gray", linestyle="--", linewidth=0.8, alpha=0.7)
    axes[1].axvline(8.0, color="tab:gray", linestyle="--", linewidth=0.8, alpha=0.7)
    axes[1].set_title(f"{channel_name} | PSD (Welch)")
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("Power (dB)")
    axes[1].grid(alpha=0.3)

    durations_ms = np.asarray([float(event["duration_ms"]) for event in events], dtype=float)
    if durations_ms.size > 0:
        axes[2].hist(durations_ms, bins=20, color="tab:cyan", edgecolor="black", alpha=0.8)
        axes[2].axvspan(BLINK_MIN_MS, BLINK_MAX_MS, color="tab:green", alpha=0.15, label="blink-like range")
        axes[2].axvline(BLINK_MIN_MS, color="tab:green", linestyle="--", linewidth=0.8)
        axes[2].axvline(BLINK_MAX_MS, color="tab:green", linestyle="--", linewidth=0.8)
        axes[2].set_title(f"{channel_name} | Event duration distribution")
        axes[2].set_xlabel("Event duration (ms)")
        axes[2].set_ylabel("Count")
        axes[2].grid(alpha=0.3)
        axes[2].legend(loc="upper right")
    else:
        axes[2].text(
            0.5,
            0.5,
            "No event candidates found",
            transform=axes[2].transAxes,
            ha="center",
            va="center",
        )
        axes[2].set_title(f"{channel_name} | Event duration distribution")
        axes[2].set_xlabel("Event duration (ms)")
        axes[2].set_ylabel("Count")
        axes[2].grid(alpha=0.3)

    win_pre = int(0.30 * sfreq)
    win_post = int(0.40 * sfreq)
    template_segments = []
    for event in blink_like_events[:20]:
        peak_idx = int(event["peak_idx"])
        start = peak_idx - win_pre
        stop = peak_idx + win_post
        if start < 0 or stop > data_uv.size:
            continue
        template_segments.append(data_uv[start:stop])

    if template_segments:
        template = np.median(np.stack(template_segments, axis=0), axis=0)
        template_time = (np.arange(template.size) - win_pre) / sfreq * 1000.0
        template_scaled, template_unit = _nice_unit_scale(template)
        axes[3].plot(template_time, template_scaled, color="tab:purple", linewidth=2)
        axes[3].axvline(0.0, color="k", linestyle="--", linewidth=0.8)
        axes[3].set_title(f"{channel_name} | Median blink-like waveform (n={len(template_segments)})")
        axes[3].set_xlabel("Time around event peak (ms)")
        axes[3].set_ylabel(f"Amplitude ({template_unit})")
        axes[3].grid(alpha=0.3)
    else:
        axes[3].text(
            0.5,
            0.5,
            "No blink-like events for waveform template",
            transform=axes[3].transAxes,
            ha="center",
            va="center",
        )
        axes[3].set_title(f"{channel_name} | Blink-like waveform")
        axes[3].set_xlabel("Time around event peak (ms)")
        axes[3].set_ylabel("Amplitude")
        axes[3].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)

    return {
        "peak_threshold_uv": threshold,
        "peak_sample_count": float(peak_sample_count),
        "peak_event_count": float(peak_count),
        "blink_like_event_count": float(len(blink_like_events)),
        "blink_like_ratio": float(blink_ratio),
        "std_uv": float(np.std(data_uv)),
    }


def plot_comparison(
    raw_base: mne.io.BaseRaw,
    raw_target: mne.io.BaseRaw,
    erg_channel: str,
    *,
    frontal_channel: str | None,
    duration_seconds: float,
    title_prefix: str,
    output_path: Path,
) -> None:
    sfreq = float(raw_base.info["sfreq"])
    n_show = int(max(1, duration_seconds * sfreq))
    n_show = min(n_show, raw_base.n_times, raw_target.n_times)

    base_erg_uv = raw_base.get_data(picks=[erg_channel], start=0, stop=n_show)[0] * 1e6
    target_erg_uv = raw_target.get_data(picks=[erg_channel], start=0, stop=n_show)[0] * 1e6

    base_erg_uv = base_erg_uv - np.median(base_erg_uv)
    target_erg_uv = target_erg_uv - np.median(target_erg_uv)
    erg_delta_uv = base_erg_uv - target_erg_uv

    times = np.arange(n_show) / sfreq
    base_scaled, base_unit = _nice_unit_scale(base_erg_uv)
    target_scaled, _ = _nice_unit_scale(target_erg_uv)
    delta_scaled, _ = _nice_unit_scale(erg_delta_uv)

    n_rows = 4 if frontal_channel is not None else 3
    fig, axes = plt.subplots(n_rows, 1, figsize=(12, 9 if n_rows == 4 else 7), sharex=True)
    if n_rows == 3:
        axes = np.asarray(axes)

    axes[0].plot(times, base_scaled, color="tab:red", linewidth=1)
    axes[0].set_title(f"{title_prefix} | {erg_channel} | Before")
    axes[0].set_ylabel(f"Amplitude ({base_unit})")
    axes[0].grid(alpha=0.3)

    axes[1].plot(times, target_scaled, color="tab:green", linewidth=1)
    axes[1].set_title(f"{erg_channel} | After")
    axes[1].set_ylabel(f"Amplitude ({base_unit})")
    axes[1].grid(alpha=0.3)

    axes[2].plot(times, delta_scaled, color="tab:blue", linewidth=1)
    axes[2].axhline(0.0, color="k", linestyle="--", linewidth=0.8)
    axes[2].set_title(f"{erg_channel} | Delta (Before - After)")
    axes[2].set_ylabel(f"Delta ({base_unit})")
    axes[2].grid(alpha=0.3)

    if frontal_channel is not None:
        base_frontal_uv = raw_base.get_data(picks=[frontal_channel], start=0, stop=n_show)[0] * 1e6
        target_frontal_uv = raw_target.get_data(picks=[frontal_channel], start=0, stop=n_show)[0] * 1e6
        base_frontal_uv = base_frontal_uv - np.median(base_frontal_uv)
        target_frontal_uv = target_frontal_uv - np.median(target_frontal_uv)

        frontal_mean = (np.std(base_frontal_uv) + np.std(target_frontal_uv)) / 2.0
        erg_mean = (np.std(base_erg_uv) + np.std(target_erg_uv)) / 2.0
        scale = frontal_mean / erg_mean if erg_mean > 0 else 1.0
        scale = 1.0 if not np.isfinite(scale) or scale <= 0 else scale

        axes[3].plot(times, target_erg_uv * scale, color="tab:purple", linewidth=1, alpha=0.8, label=f"{erg_channel} (scaled)")
        axes[3].plot(times, target_frontal_uv, color="tab:orange", linewidth=1, alpha=0.85, label=f"{frontal_channel}")
        axes[3].set_title(f"After-stage overlay | {erg_channel} vs {frontal_channel}")
        axes[3].set_ylabel("Amplitude (uV)")
        axes[3].grid(alpha=0.3)
        axes[3].legend(loc="upper right")

    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    subjects = resolve_subjects(args.subjects)
    output_dir = config.OUTPUT_DIR / "erg_eog_check" / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    created = 0
    comparison_created = 0
    seen = 0

    print("=" * 80)
    print("ERG Channel Plot Export")
    print("=" * 80)

    for subject_id in subjects:
        for person in args.persons:
            if args.max_recordings > 0 and seen >= args.max_recordings:
                break

            found = find_input_file(subject_id, person)
            if found is None:
                print(f"sub-{subject_id} {person}: skipped (no matching input file)")
                continue

            input_path, suffix = found
            seen += 1
            raw = mne.io.read_raw_fif(input_path, preload=False, verbose="ERROR")
            erg_channels = get_erg_channels(raw)

            if not erg_channels:
                print(f"sub-{subject_id} {person}: no ERG channels found")
                continue

            print(
                f"sub-{subject_id} {person}: plotting {len(erg_channels)} ERG channel(s) "
                f"from {input_path.name}"
            )
            for ch_name in erg_channels:
                file_name = f"sub-{subject_id}_{person}_{suffix}_{ch_name}_erg_report.png"
                out_path = output_dir / file_name
                metrics = plot_erg_channel(
                    raw,
                    ch_name,
                    duration_seconds=args.duration_seconds,
                    peak_quantile=args.peak_quantile,
                    title_prefix=f"sub-{subject_id} {person} ({suffix})",
                    output_path=out_path,
                )
                created += 1
                print(
                    f"  - {ch_name}: saved {out_path.name} | "
                    f"std={metrics['std_uv']:.2f} uV, "
                    f"events={int(metrics['peak_event_count'])}, "
                    f"blink_like={int(metrics['blink_like_event_count'])} ({metrics['blink_like_ratio']:.0%}), "
                    f"thr={metrics['peak_threshold_uv']:.2f} uV"
                )

            if not args.skip_comparison:
                pair = find_comparison_pair(subject_id, person)
                if pair is None:
                    print("  - comparison: skipped (requires both base-step file and filtered file)")
                else:
                    base_path, base_suffix, target_path, target_suffix = pair
                    raw_base = mne.io.read_raw_fif(base_path, preload=False, verbose="ERROR")
                    raw_target = mne.io.read_raw_fif(target_path, preload=False, verbose="ERROR")
                    frontal_channel = get_frontal_channel(raw_target)

                    common_erg = [ch for ch in get_erg_channels(raw_target) if ch in raw_base.ch_names]
                    if not common_erg:
                        print("  - comparison: skipped (no common ERG channels between stages)")
                    else:
                        for ch_name in common_erg:
                            comp_name = (
                                f"sub-{subject_id}_{person}_{base_suffix}_vs_{target_suffix}_{ch_name}_comparison.png"
                            )
                            comp_path = output_dir / comp_name
                            plot_comparison(
                                raw_base,
                                raw_target,
                                ch_name,
                                frontal_channel=frontal_channel,
                                duration_seconds=args.duration_seconds,
                                title_prefix=f"sub-{subject_id} {person} ({base_suffix} -> {target_suffix})",
                                output_path=comp_path,
                            )
                            comparison_created += 1
                            if frontal_channel is not None:
                                print(f"  - {ch_name}: saved {comp_name} (with {frontal_channel} overlay)")
                            else:
                                print(f"  - {ch_name}: saved {comp_name}")

        if args.max_recordings > 0 and seen >= args.max_recordings:
            break

    print("-" * 80)
    print(f"Saved {created} ERG plot(s) and {comparison_created} comparison plot(s) to: {output_dir}")


if __name__ == "__main__":
    main()
