"""
Sanity Check: Can ERG channels be used as EOG proxies?

The script scans available preprocessing outputs, looks for ERG channels,
measures how strongly large ERG deflections co-occur with frontal EEG activity,
and writes a clear YES/NO recommendation.

Outputs:
- TSV per recording with channel-level metrics
- JSON summary with overall recommendation
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import mne
import numpy as np

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

PIPELINE_DIR = CURRENT_DIR.parent.parent / "eeg_pipeline"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from preprocessing import config


MAX_SECONDS_DEFAULT = 120.0
PEAK_QUANTILE_DEFAULT = 0.995
ERG_PEAK_MIN_EVENTS = 5
MIN_WINDOW_SECONDS = 30.0
CHANNEL_COUPLING_THRESHOLD = 1.30
RECORDING_DECISION_THRESHOLD = 1.30
GLOBAL_YES_RATE_THRESHOLD = 0.60
WINDOW_PASS_RATE_THRESHOLD = 0.50

FILE_SUFFIX_PRIORITY = [
    "filtered",
    "interpolated",
]

FRONTAL_CANDIDATES = ["Fp1", "Fpz", "Fp2", "AF7", "AF8", "AF3", "AF4", "AFz"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check whether ERG channels are usable as EOG proxies.")
    parser.add_argument(
        "--subjects",
        nargs="*",
        default=None,
        help="Optional subject IDs without 'sub-' prefix, e.g. 01 02 03",
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=MAX_SECONDS_DEFAULT,
        help=f"Maximum duration per recording to inspect (default: {MAX_SECONDS_DEFAULT}).",
    )
    parser.add_argument(
        "--peak-quantile",
        type=float,
        default=PEAK_QUANTILE_DEFAULT,
        help=f"Quantile used to define ERG peak events (default: {PEAK_QUANTILE_DEFAULT}).",
    )
    return parser.parse_args()


def resolve_subjects(subject_args: list[str] | None) -> list[str]:
    if subject_args:
        return [subject.replace("sub-", "") for subject in subject_args]
    return list(config.SUBJECTS)


def find_input_file(subject_id: str, person: str) -> Path | None:
    for suffix in FILE_SUFFIX_PRIORITY:
        path = config.OUTPUT_DIR / f"sub-{subject_id}_{person}_{suffix}.fif"
        if path.exists():
            return path
    return None


def get_erg_channels(raw: mne.io.BaseRaw) -> list[str]:
    bads = set(raw.info.get("bads", []))
    return [channel for channel in raw.ch_names if "erg" in channel.lower() and channel not in bads]


def get_frontal_channels(raw: mne.io.BaseRaw) -> list[str]:
    bads = set(raw.info.get("bads", []))
    return [channel for channel in FRONTAL_CANDIDATES if channel in raw.ch_names and channel not in bads]


def _corrcoef(signal_a: np.ndarray, signal_b: np.ndarray) -> float:
    denom = np.std(signal_a) * np.std(signal_b)
    if denom == 0:
        return float("nan")
    return float(np.dot(signal_a, signal_b) / (signal_a.size * denom))


def _count_peak_events(mask: np.ndarray) -> int:
    if mask.size == 0:
        return 0
    mask_int = mask.astype(np.int8)
    starts = np.sum((mask_int[1:] == 1) & (mask_int[:-1] == 0))
    return int(starts + (1 if mask_int[0] == 1 else 0))


def iter_windows(raw: mne.io.BaseRaw, max_seconds: float) -> list[tuple[int, int]]:
    window_size = int(raw.info["sfreq"] * max_seconds)
    min_window_size = int(raw.info["sfreq"] * MIN_WINDOW_SECONDS)
    if window_size <= 0:
        return []

    windows = []
    start = 0
    while start < raw.n_times:
        stop = min(start + window_size, raw.n_times)
        if stop - start >= min_window_size:
            windows.append((start, stop))
        start = stop
    return windows


def compute_window_metrics(
    raw: mne.io.BaseRaw,
    erg_channel: str,
    frontal_channels: list[str],
    *,
    start: int,
    stop: int,
    peak_quantile: float,
) -> dict[str, object]:
    picks = [erg_channel] + frontal_channels
    data = raw.get_data(picks=picks, start=start, stop=stop)
    data = data - np.median(data, axis=1, keepdims=True)

    erg_data = data[0]
    erg_abs = np.abs(erg_data)
    threshold = float(np.quantile(erg_abs, peak_quantile))
    peak_mask = erg_abs >= threshold
    peak_sample_count = int(np.sum(peak_mask))
    peak_event_count = _count_peak_events(peak_mask)

    frontal_metrics = []
    for idx, frontal_channel in enumerate(frontal_channels, start=1):
        frontal_data = data[idx]
        frontal_abs = np.abs(frontal_data)
        baseline_abs = float(np.mean(frontal_abs))
        peak_abs = float(np.mean(frontal_abs[peak_mask])) if peak_sample_count else float("nan")
        coupling_ratio = peak_abs / baseline_abs if baseline_abs > 0 and peak_sample_count else float("nan")
        correlation = _corrcoef(erg_data, frontal_data)
        frontal_metrics.append(
            {
                "channel": frontal_channel,
                "coupling_ratio": coupling_ratio,
                "correlation": correlation,
            }
        )

    best_metric = max(
        frontal_metrics,
        key=lambda item: float(item["coupling_ratio"]) if np.isfinite(item["coupling_ratio"]) else float("-inf"),
        default=None,
    )
    return {
        "peak_threshold": threshold,
        "peak_sample_count": peak_sample_count,
        "peak_event_count": peak_event_count,
        "frontal_metrics": frontal_metrics,
        "best_metric": best_metric,
    }


def compute_erg_metrics(
    raw: mne.io.BaseRaw,
    erg_channel: str,
    frontal_channels: list[str],
    *,
    max_seconds: float,
    peak_quantile: float,
) -> dict[str, object]:
    windows = iter_windows(raw, max_seconds=max_seconds)
    if not windows:
        return {
            "erg_channel": erg_channel,
            "erg_type": raw.get_channel_types(picks=[erg_channel])[0],
            "erg_std": float("nan"),
            "peak_threshold": float("nan"),
            "peak_sample_count": 0,
            "peak_count": 0,
            "n_windows": 0,
            "window_pass_rate": 0.0,
            "best_frontal_channel": "",
            "best_coupling_ratio": float("nan"),
            "mean_coupling_ratio": float("nan"),
            "best_correlation": float("nan"),
            "is_eog_like": False,
        }

    erg_std_values = []
    peak_thresholds = []
    total_peak_sample_count = 0
    total_peak_event_count = 0
    window_best_ratios = []
    frontal_ratio_map = {channel: [] for channel in frontal_channels}
    frontal_corr_map = {channel: [] for channel in frontal_channels}

    for start, stop in windows:
        picks = [erg_channel]
        erg_segment = raw.get_data(picks=picks, start=start, stop=stop)[0]
        erg_segment = erg_segment - np.median(erg_segment)
        erg_std_values.append(float(np.std(erg_segment)))

        window_result = compute_window_metrics(
            raw,
            erg_channel,
            frontal_channels,
            start=start,
            stop=stop,
            peak_quantile=peak_quantile,
        )
        peak_thresholds.append(float(window_result["peak_threshold"]))
        total_peak_sample_count += int(window_result["peak_sample_count"])
        total_peak_event_count += int(window_result["peak_event_count"])

        best_metric = window_result["best_metric"]
        if best_metric is not None and np.isfinite(best_metric["coupling_ratio"]):
            window_best_ratios.append(float(best_metric["coupling_ratio"]))

        for metric in window_result["frontal_metrics"]:
            if np.isfinite(metric["coupling_ratio"]):
                frontal_ratio_map[metric["channel"]].append(float(metric["coupling_ratio"]))
            if np.isfinite(metric["correlation"]):
                frontal_corr_map[metric["channel"]].append(float(metric["correlation"]))

    frontal_summary = []
    for frontal_channel in frontal_channels:
        ratios = frontal_ratio_map[frontal_channel]
        corrs = frontal_corr_map[frontal_channel]
        frontal_summary.append(
            {
                "channel": frontal_channel,
                "median_ratio": float(np.median(ratios)) if ratios else float("nan"),
                "median_correlation": float(np.median(corrs)) if corrs else float("nan"),
            }
        )

    frontal_summary.sort(
        key=lambda item: float(item["median_ratio"]) if np.isfinite(item["median_ratio"]) else float("-inf"),
        reverse=True,
    )
    best = frontal_summary[0] if frontal_summary else None
    best_ratio = float(best["median_ratio"]) if best is not None else float("nan")
    mean_ratio = float(
        np.nanmean([item["median_ratio"] for item in frontal_summary])
    ) if frontal_summary else float("nan")
    window_pass_rate = float(np.mean(np.asarray(window_best_ratios) >= CHANNEL_COUPLING_THRESHOLD)) if window_best_ratios else 0.0
    is_eog_like = (
        total_peak_event_count >= ERG_PEAK_MIN_EVENTS
        and best_ratio >= CHANNEL_COUPLING_THRESHOLD
        and window_pass_rate >= WINDOW_PASS_RATE_THRESHOLD
    )

    return {
        "erg_channel": erg_channel,
        "erg_type": raw.get_channel_types(picks=[erg_channel])[0],
        "erg_std": float(np.median(erg_std_values)) if erg_std_values else float("nan"),
        "peak_threshold": float(np.median(peak_thresholds)) if peak_thresholds else float("nan"),
        "peak_sample_count": total_peak_sample_count,
        "peak_count": total_peak_event_count,
        "n_windows": len(windows),
        "window_pass_rate": window_pass_rate,
        "best_frontal_channel": best["channel"] if best is not None else "",
        "best_coupling_ratio": best_ratio,
        "mean_coupling_ratio": mean_ratio,
        "best_correlation": float(best["median_correlation"]) if best is not None else float("nan"),
        "is_eog_like": is_eog_like,
    }


def assess_recording(
    subject_id: str,
    person: str,
    path: Path,
    *,
    max_seconds: float,
    peak_quantile: float,
) -> dict[str, object]:
    raw = mne.io.read_raw_fif(path, preload=False, verbose="ERROR")
    erg_channels = get_erg_channels(raw)
    frontal_channels = get_frontal_channels(raw)

    if not erg_channels:
        return {
            "subject_id": subject_id,
            "person": person,
            "file": path.name,
            "status": "no_erg",
            "recommendation": "NO",
            "reason": "No ERG channels present in recording.",
            "channels": [],
        }

    if len(frontal_channels) < 3:
        return {
            "subject_id": subject_id,
            "person": person,
            "file": path.name,
            "status": "insufficient_frontals",
            "recommendation": "NO",
            "reason": "Too few frontal channels available for comparison.",
            "channels": [],
        }

    channel_results = [
        compute_erg_metrics(
            raw,
            erg_channel,
            frontal_channels,
            max_seconds=max_seconds,
            peak_quantile=peak_quantile,
        )
        for erg_channel in erg_channels
    ]
    channel_results.sort(key=lambda item: float(item["best_coupling_ratio"]), reverse=True)
    best_result = channel_results[0]
    best_ratio = float(best_result["best_coupling_ratio"])
    recording_yes = best_ratio >= RECORDING_DECISION_THRESHOLD
    recommendation = "YES" if recording_yes else "NO"
    reason = (
        f"Best ERG channel {best_result['erg_channel']} reached coupling ratio {best_ratio:.2f} "
        f"with frontal channel {best_result['best_frontal_channel']}."
    )

    return {
        "subject_id": subject_id,
        "person": person,
        "file": path.name,
        "status": "ok",
        "recommendation": recommendation,
        "reason": reason,
        "channels": channel_results,
    }


def write_channel_tsv(output_path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "subject_id",
        "person",
        "file",
        "recording_recommendation",
        "erg_channel",
        "erg_type",
        "erg_std",
        "peak_threshold",
        "peak_sample_count",
        "peak_count",
        "n_windows",
        "window_pass_rate",
        "best_frontal_channel",
        "best_coupling_ratio",
        "mean_coupling_ratio",
        "best_correlation",
        "is_eog_like",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def summarise(results: list[dict[str, object]]) -> dict[str, object]:
    usable = [result for result in results if result["status"] == "ok"]
    yes_recordings = [result for result in usable if result["recommendation"] == "YES"]
    yes_rate = len(yes_recordings) / len(usable) if usable else 0.0
    overall_recommendation = "YES" if usable and yes_rate >= GLOBAL_YES_RATE_THRESHOLD else "NO"

    return {
        "total_recordings_checked": len(results),
        "recordings_with_metrics": len(usable),
        "recordings_recommended_yes": len(yes_recordings),
        "recording_yes_rate": yes_rate,
        "global_yes_rate_threshold": GLOBAL_YES_RATE_THRESHOLD,
        "overall_recommendation": overall_recommendation,
        "recommendation_reason": (
            f"{len(yes_recordings)} of {len(usable)} recordings with ERG data passed the EOG-like coupling test."
            if usable
            else "No usable recordings with ERG channels were found."
        ),
        "interpretation": (
            "YES means ERG channels are usable as an EOG fallback in this dataset."
            if overall_recommendation == "YES"
            else "NO means ERG channels should not be treated as a reliable EOG source by default."
        ),
    }


def main() -> None:
    args = parse_args()
    subjects = resolve_subjects(args.subjects)

    output_dir = config.OUTPUT_DIR / "erg_eog_check"
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    channel_rows = []

    print("=" * 80)
    print("ERG as EOG Viability Check")
    print("=" * 80)

    for subject_id in subjects:
        for person in ["P1", "P2"]:
            input_path = find_input_file(subject_id, person)
            if input_path is None:
                print(f"sub-{subject_id} {person}: skipped (no suitable preprocessing file found)")
                results.append(
                    {
                        "subject_id": subject_id,
                        "person": person,
                        "file": "",
                        "status": "missing_file",
                        "recommendation": "NO",
                        "reason": "No suitable preprocessing file found.",
                        "channels": [],
                    }
                )
                continue

            result = assess_recording(
                subject_id,
                person,
                input_path,
                max_seconds=args.max_seconds,
                peak_quantile=args.peak_quantile,
            )
            results.append(result)

            print(
                f"sub-{subject_id} {person}: {result['recommendation']} | "
                f"{result['status']} | {result['reason']}"
            )

            for channel_result in result["channels"]:
                channel_rows.append(
                    {
                        "subject_id": subject_id,
                        "person": person,
                        "file": result["file"],
                        "recording_recommendation": result["recommendation"],
                        **channel_result,
                    }
                )

    summary = summarise(results)
    summary["results"] = results

    tsv_path = output_dir / "erg_eog_viability.tsv"
    json_path = output_dir / "erg_eog_viability_summary.json"

    write_channel_tsv(tsv_path, channel_rows)
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("-" * 80)
    print(f"Overall recommendation: {summary['overall_recommendation']}")
    print(summary["recommendation_reason"])
    print(f"Saved channel metrics: {tsv_path}")
    print(f"Saved summary: {json_path}")


if __name__ == "__main__":
    main()