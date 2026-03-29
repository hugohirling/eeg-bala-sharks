from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
from mne.filter import filter_data
from scipy.signal import hilbert

CURRENT_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = CURRENT_DIR.parent
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from preprocessing import config


def _resolve_subjects(subjects_arg: str | None) -> list[str]:
    def _normalize(value: str) -> str:
        value = value.strip()
        return value.zfill(2) if value.isdigit() else value

    if subjects_arg:
        return [_normalize(part) for part in subjects_arg.split(",") if part.strip()]
    return [_normalize(str(subject)) for subject in config.SUBJECTS]


def _epoch_path(subject_id: str, person: str) -> Path:
    return Path(config.OUTPUT_DIR) / f"sub-{subject_id}_{person}_epoch.fif"


def _load_pair_decision_epochs(subject_id: str, tmin: float, tmax: float) -> tuple[mne.Epochs, mne.Epochs]:
    p1_path = _epoch_path(subject_id, "P1")
    p2_path = _epoch_path(subject_id, "P2")
    if not p1_path.exists() or not p2_path.exists():
        raise FileNotFoundError(f"Missing epoch file(s) for sub-{subject_id}: {p1_path} / {p2_path}")

    p1_epochs = mne.read_epochs(str(p1_path), preload=True, verbose=False).crop(tmin=tmin, tmax=tmax)
    p2_epochs = mne.read_epochs(str(p2_path), preload=True, verbose=False).crop(tmin=tmin, tmax=tmax)

    p1_epochs.pick("eeg")
    p2_epochs.pick("eeg")

    common_channels = [name for name in p1_epochs.ch_names if name in set(p2_epochs.ch_names)]
    if not common_channels:
        raise RuntimeError(f"No common EEG channels for sub-{subject_id}")

    p1_epochs = p1_epochs.copy().pick(common_channels)
    p2_epochs = p2_epochs.copy().pick(common_channels)

    n_trials = min(len(p1_epochs), len(p2_epochs))
    if n_trials < 10:
        raise RuntimeError(f"Too few aligned trials for sub-{subject_id}: {n_trials}")

    return p1_epochs[:n_trials], p2_epochs[:n_trials]


def _compute_phase_data(data: np.ndarray, sfreq: float, fmin: float, fmax: float) -> np.ndarray:
    # Narrowband filtering before Hilbert transform gives phase in the chosen band.
    filtered = filter_data(data, sfreq=sfreq, l_freq=fmin, h_freq=fmax, verbose=False)
    analytic = hilbert(filtered, axis=-1)
    return np.angle(analytic)


def _trial_channel_plv(phase_a: np.ndarray, phase_b: np.ndarray) -> np.ndarray:
    phase_diff = phase_a - phase_b
    return np.abs(np.mean(np.exp(1j * phase_diff), axis=-1))


def _summarize_subject(
    subject_id: str,
    tmin: float,
    tmax: float,
    fmin: float,
    fmax: float,
    n_shuffles: int,
    random_state: int,
) -> tuple[list[dict], dict, mne.Info]:
    p1_epochs, p2_epochs = _load_pair_decision_epochs(subject_id=subject_id, tmin=tmin, tmax=tmax)
    x1 = p1_epochs.get_data(copy=True)
    x2 = p2_epochs.get_data(copy=True)

    sfreq = float(p1_epochs.info["sfreq"])
    phase_1 = _compute_phase_data(data=x1, sfreq=sfreq, fmin=fmin, fmax=fmax)
    phase_2 = _compute_phase_data(data=x2, sfreq=sfreq, fmin=fmin, fmax=fmax)

    real_trial_channel_plv = _trial_channel_plv(phase_1, phase_2)
    real_channel_plv = real_trial_channel_plv.mean(axis=0)
    real_global_plv = float(real_channel_plv.mean())

    rng = np.random.default_rng(seed=random_state)
    shuffle_global: list[float] = []
    shuffle_channel: list[np.ndarray] = []
    for _ in range(n_shuffles):
        perm = rng.permutation(phase_2.shape[0])
        shuffled_trial_channel_plv = _trial_channel_plv(phase_1, phase_2[perm])
        shuffled_channel_plv = shuffled_trial_channel_plv.mean(axis=0)
        shuffle_channel.append(shuffled_channel_plv)
        shuffle_global.append(float(shuffled_channel_plv.mean()))

    shuffle_channel_mean = np.mean(np.stack(shuffle_channel, axis=0), axis=0)
    shuffle_global_arr = np.asarray(shuffle_global, dtype=float)

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
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def _save_subject_global_plot(subject_rows: list[dict], out_dir: Path) -> Path:
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
    ax.set_title("Interbrain Synchrony per Pair (Decision phase, real vs shuffled)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()

    out_path = out_dir / "interbrain_synchrony_subject_global_real_vs_shuffled.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _save_group_topomap(group_channel_rows: list[dict], plot_info: mne.Info, out_dir: Path) -> Path:
    rows = sorted(group_channel_rows, key=lambda row: str(row["channel"]))
    real = np.asarray([float(row["plv_real"]) for row in rows], dtype=float)
    shuf = np.asarray([float(row["plv_shuffle_mean"]) for row in rows], dtype=float)
    delta = np.asarray([float(row["plv_delta"]) for row in rows], dtype=float)

    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.2))

    vmin_abs = float(min(real.min(), shuf.min()))
    vmax_abs = float(max(real.max(), shuf.max()))

    img_real, _ = mne.viz.plot_topomap(
        real,
        plot_info,
        axes=axes[0],
        show=False,
        contours=0,
        cmap="viridis",
        vlim=(vmin_abs, vmax_abs),
    )
    axes[0].set_title("Real PLV")

    img_shuf, _ = mne.viz.plot_topomap(
        shuf,
        plot_info,
        axes=axes[1],
        show=False,
        contours=0,
        cmap="viridis",
        vlim=(vmin_abs, vmax_abs),
    )
    axes[1].set_title("Shuffled PLV")

    max_abs = float(np.max(np.abs(delta)))
    img_delta, _ = mne.viz.plot_topomap(
        delta,
        plot_info,
        axes=axes[2],
        show=False,
        contours=0,
        cmap="RdBu_r",
        vlim=(-max_abs, max_abs),
    )
    axes[2].set_title("Real - Shuffled")

    fig.suptitle("Group Interbrain Synchrony Topomaps (alpha band)")
    fig.colorbar(img_shuf, ax=[axes[0], axes[1]], shrink=0.8)
    fig.colorbar(img_delta, ax=[axes[2]], shrink=0.8)
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    out_path = out_dir / "interbrain_synchrony_group_topomaps.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _save_zscore_plot(subject_rows: list[dict], out_dir: Path) -> Path:
    rows = sorted(subject_rows, key=lambda row: str(row["subject"]))
    z = np.asarray([float(row["global_plv_z"]) for row in rows], dtype=float)

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--")
    ax.bar(np.arange(len(z)), z, color="#1976d2", edgecolor="black", linewidth=0.6)
    ax.set_xticks(np.arange(len(z)))
    ax.set_xticklabels([f"sub-{row['subject']}" for row in rows], rotation=45, ha="right")
    ax.set_ylabel("Global PLV z-score vs shuffled")
    ax.set_title("Interbrain Synchrony Strength by Pair")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()

    out_path = out_dir / "interbrain_synchrony_subject_zscores.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def run_interbrain_synchrony(
    subjects: list[str],
    tmin: float,
    tmax: float,
    fmin: float,
    fmax: float,
    n_shuffles: int,
    random_state: int,
) -> tuple[list[dict], list[dict], dict, list[dict], mne.Info]:
    all_channel_rows: list[dict] = []
    subject_summaries: list[dict] = []
    representative_info: mne.Info | None = None

    for subject_id in subjects:
        try:
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
            print(
                f"sub-{subject_id}: global PLV real={summary['global_plv_real']:.4f}, "
                f"shuffle={summary['global_plv_shuffle_mean']:.4f}, "
                f"z={summary['global_plv_z']:.3f}, p={summary['global_plv_perm_p']:.4f}"
            )
        except Exception as exc:
            print(f"sub-{subject_id}: skipped ({exc})")

    if not all_channel_rows or representative_info is None:
        raise RuntimeError("No valid interbrain synchrony results were produced.")

    subject_summaries = sorted(subject_summaries, key=lambda row: str(row["subject"]))
    group_channel_rows = _group_channel_rows(all_channel_rows)

    real = np.asarray([float(row["global_plv_real"]) for row in subject_summaries], dtype=float)
    shuf = np.asarray([float(row["global_plv_shuffle_mean"]) for row in subject_summaries], dtype=float)
    z = np.asarray([float(row["global_plv_z"]) for row in subject_summaries], dtype=float)
    p = np.asarray([float(row["global_plv_perm_p"]) for row in subject_summaries], dtype=float)

    group_summary = {
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
    }

    ordered_channels = [str(row["channel"]) for row in group_channel_rows]
    picks = mne.pick_channels(representative_info["ch_names"], include=ordered_channels, ordered=True)
    plot_info = mne.pick_info(representative_info.copy(), picks)

    return all_channel_rows, subject_summaries, group_summary, group_channel_rows, plot_info


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute interbrain phase-locking value (PLV) between P1 and P2 during decision phase, "
            "with shuffled-trial control and verification plots."
        )
    )
    parser.add_argument("--subjects", type=str, default=None, help="Comma-separated subject IDs, e.g. 01,02,03")
    parser.add_argument("--tmin", type=float, default=0.0, help="Start time of decision window in seconds")
    parser.add_argument("--tmax", type=float, default=2.0, help="End time of decision window in seconds")
    parser.add_argument("--fmin", type=float, default=float(config.IBS_FMIN), help="Lower band edge in Hz")
    parser.add_argument("--fmax", type=float, default=float(config.IBS_FMAX), help="Upper band edge in Hz")
    parser.add_argument("--n-shuffles", type=int, default=200, help="Number of shuffled controls per subject")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    subjects = _resolve_subjects(args.subjects)

    out_dir = Path(config.OUTPUT_DIR).parent / "decoding"
    out_dir.mkdir(parents=True, exist_ok=True)

    channel_rows, subject_rows, group_summary, group_channel_rows, plot_info = run_interbrain_synchrony(
        subjects=subjects,
        tmin=args.tmin,
        tmax=args.tmax,
        fmin=args.fmin,
        fmax=args.fmax,
        n_shuffles=args.n_shuffles,
        random_state=args.random_state,
    )

    channel_path = out_dir / "interbrain_synchrony_channel_values.csv"
    subject_path = out_dir / "interbrain_synchrony_subject_summary.csv"
    group_path = out_dir / "interbrain_synchrony_group_summary.json"

    _write_csv(
        channel_path,
        channel_rows,
        fieldnames=["subject", "channel", "plv_real", "plv_shuffle_mean", "plv_delta"],
    )
    _write_csv(
        subject_path,
        subject_rows,
        fieldnames=[
            "subject",
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
        ],
    )
    group_path.write_text(json.dumps(group_summary, indent=2), encoding="utf-8")

    subject_plot_path = _save_subject_global_plot(subject_rows=subject_rows, out_dir=out_dir)
    topomap_plot_path = _save_group_topomap(group_channel_rows=group_channel_rows, plot_info=plot_info, out_dir=out_dir)
    zscore_plot_path = _save_zscore_plot(subject_rows=subject_rows, out_dir=out_dir)

    print("\n=== Interbrain Synchrony Summary ===")
    print(f"N subjects: {group_summary['n_subjects']}")
    print(f"Band: {group_summary['fmin']:.1f}-{group_summary['fmax']:.1f} Hz")
    print(f"Window: {group_summary['tmin']:.2f}-{group_summary['tmax']:.2f} s")
    print(f"Mean global PLV (real): {group_summary['mean_global_plv_real']:.4f}")
    print(f"Mean global PLV (shuffle): {group_summary['mean_global_plv_shuffle']:.4f}")
    print(f"Mean global PLV delta: {group_summary['mean_global_plv_delta']:.4f}")
    print(f"Subjects with perm p < 0.05: {group_summary['subjects_with_perm_p_lt_0_05']}")
    print(f"Saved channel values: {channel_path}")
    print(f"Saved subject summary: {subject_path}")
    print(f"Saved group summary: {group_path}")
    print(f"Saved subject plot: {subject_plot_path}")
    print(f"Saved topomap plot: {topomap_plot_path}")
    print(f"Saved z-score plot: {zscore_plot_path}")


if __name__ == "__main__":
    main()
