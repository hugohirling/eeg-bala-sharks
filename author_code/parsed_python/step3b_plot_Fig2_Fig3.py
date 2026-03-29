"""
Plot decoding results (Python version of step3b_plot_Fig2_Fig3.m):
   - Plot decoding accuracy across time
   - Split winner vs loser for self-current decoding
   - Compute Bayes-factor approximation without random fallbacks
"""

import os
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat

warnings.filterwarnings("ignore")

# Central path configuration
import sys
BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
import paths as _project_paths  # noqa: E402

path_to_data = str(_project_paths.INPUT_DIR)
OUTPUT_ROOT = _project_paths.OUTPUT_DIR / "author_code"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
derivatives_path = str(OUTPUT_ROOT)
plot_dir = str(OUTPUT_ROOT / "plots")
os.makedirs(plot_dir, exist_ok=True)

# Set parameters
pair_ids = np.concatenate([np.arange(1, 10), np.arange(11, 23), np.arange(25, 35)])
num_pairs = len(pair_ids)
pair_idx = np.arange(1, num_pairs * 2 + 1).reshape(-1, 2)
pair_idx0 = pair_idx - 1
num_tests = 4
num_time_bins = 20
test_names = ["Self Current", "Other Current", "Self Previous", "Other Previous"]

time_bin_info = {
    "part_a": {"name": "Decision", "start": -0.2, "end": 2.0, "n_bins": 8, "color": "#3498db"},
    "part_b": {"name": "Response", "start": 1.8, "end": 4.0, "n_bins": 8, "color": "#e74c3c"},
    "part_c": {"name": "Feedback", "start": 3.8, "end": 5.0, "n_bins": 4, "color": "#2ecc71"},
}


def calculate_time_points(info):
    points = []
    for part in ["part_a", "part_b", "part_c"]:
        part_info = info[part]
        duration = part_info["end"] - part_info["start"]
        width = duration / part_info["n_bins"]
        for i in range(part_info["n_bins"]):
            points.append(part_info["start"] + (i + 0.5) * width)
    return np.asarray(points)


def nansem(data, axis=0):
    n = np.sum(np.isfinite(data), axis=axis)
    std = np.nanstd(data, axis=axis)
    return std / np.sqrt(np.maximum(n, 1))


def bic_bayes_factor_onesample(data, mu0=0.0):
    """Approximate BF10 with a BIC difference for one-sample mean model.

    Returns BF10 (>=0), with 1 meaning equal evidence.
    """
    x = np.asarray(data, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 3:
        return np.nan

    # H0: mean fixed to mu0.
    rss0 = np.sum((x - mu0) ** 2)
    # H1: mean estimated from data.
    mu1 = np.mean(x)
    rss1 = np.sum((x - mu1) ** 2)

    eps = 1e-12
    rss0 = max(rss0, eps)
    rss1 = max(rss1, eps)

    # k0=0 free mean parameters, k1=1 free mean parameter.
    bic0 = n * np.log(rss0 / n) + 0 * np.log(n)
    bic1 = n * np.log(rss1 / n) + 1 * np.log(n)

    bf10 = np.exp((bic0 - bic1) / 2.0)
    return float(bf10)


def load_winner_idx(pair):
    events_path = os.path.join(path_to_data, f"sub-{pair:02d}", "eeg", f"sub-{pair:02d}_task-RPS_events.tsv")
    if not os.path.exists(events_path):
        print(f"Warning: missing events file for sub-{pair:02d}: {events_path}")
        return None
    events = pd.read_csv(events_path, sep="\t")
    p1_wins = int((events["outcome"] == 2).sum())
    p2_wins = int((events["outcome"] == 3).sum())
    if p1_wins > p2_wins:
        return 0
    if p2_wins > p1_wins:
        return 1
    return None


def load_decoding_accuracy():
    all_acc = np.full((num_pairs * 2, num_time_bins, num_tests), np.nan)
    missing_files = []

    for p, pair in enumerate(pair_ids):
        for ppt in (1, 2):
            mat_path = os.path.join(derivatives_path, f"pair-{pair:02d}_player-{ppt}_task-RPS_decoding.mat")
            row_idx = pair_idx0[p, ppt - 1]
            if not os.path.exists(mat_path):
                missing_files.append(mat_path)
                continue

            try:
                mat_data = loadmat(mat_path)
                for test in range(num_tests):
                    key = f"decoding_acc_test{test}"
                    if key not in mat_data:
                        continue
                    arr = np.asarray(mat_data[key]).squeeze()
                    if arr.size == num_time_bins:
                        all_acc[row_idx, :, test] = arr.astype(float)
            except Exception as exc:
                print(f"Error loading {mat_path}: {exc}")

    if missing_files:
        print(f"Warning: {len(missing_files)} decoding files missing. No random fallback was used.")
    return all_acc


def make_winner_loser_split(all_acc):
    wl = np.full((num_pairs, num_time_bins, 2, num_tests), np.nan)
    tie_pairs = []
    missing_pairs = []
    no_decode_pairs = []
    for p, pair in enumerate(pair_ids):
        this_pair_idx = pair_idx0[p]
        pair_data = all_acc[this_pair_idx, :, :]
        if not np.any(np.isfinite(pair_data)):
            no_decode_pairs.append(pair)
            continue

        winner = load_winner_idx(pair)
        if winner is None:
            events_path = os.path.join(path_to_data, f"sub-{pair:02d}", "eeg", f"sub-{pair:02d}_task-RPS_events.tsv")
            if os.path.exists(events_path):
                tie_pairs.append(pair)
            else:
                missing_pairs.append(pair)
            continue

        for test in range(num_tests):
            wl[p, :, 0, test] = all_acc[this_pair_idx[winner], :, test]
            wl[p, :, 1, test] = all_acc[this_pair_idx[1 - winner], :, test]

    if tie_pairs:
        print(f"Info: tie pairs excluded from winner/loser split: {tie_pairs}")
    if missing_pairs:
        print(f"Info: missing events skipped in winner/loser split: {missing_pairs}")
    if no_decode_pairs:
        print(f"Info: no decoding data for pairs: {no_decode_pairs}")
    return wl


def compute_bayes_factors(all_acc, wl):
    bf = np.full((num_tests, num_time_bins), np.nan)
    bf_wl = np.full((3, num_time_bins, num_tests), np.nan)

    for test in range(num_tests):
        for t in range(num_time_bins):
            # All participants vs chance.
            bf[test, t] = bic_bayes_factor_onesample(all_acc[:, t, test] - (1.0 / 3.0), mu0=0.0)

            # Winners and losers separately vs chance.
            bf_wl[0, t, test] = bic_bayes_factor_onesample(wl[:, t, 0, test] - (1.0 / 3.0), mu0=0.0)
            bf_wl[1, t, test] = bic_bayes_factor_onesample(wl[:, t, 1, test] - (1.0 / 3.0), mu0=0.0)

            # Winner-loser difference vs 0.
            diff = wl[:, t, 0, test] - wl[:, t, 1, test]
            bf_wl[2, t, test] = bic_bayes_factor_onesample(diff, mu0=0.0)

    return bf, bf_wl


def add_phase_backgrounds(ax):
    for part in ["part_a", "part_b", "part_c"]:
        info = time_bin_info[part]
        ax.axvspan(info["start"], info["end"], alpha=0.1, color=info["color"])
    ax.axvline(x=2.0, color="gray", linestyle=":", linewidth=1, alpha=0.5)
    ax.axvline(x=4.0, color="gray", linestyle=":", linewidth=1, alpha=0.5)


def main():
    time_points_seconds = calculate_time_points(time_bin_info)
    colors = ["blue", "orange", "green", "red"]

    print("Loading decoding results...")
    all_acc = load_decoding_accuracy()
    wl = make_winner_loser_split(all_acc)
    bf, bf_wl = compute_bayes_factors(all_acc, wl)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Decoding Analysis Results", fontsize=14, fontweight="bold")

    # Plot 1: all participants.
    ax = axes[0, 0]
    for test in range(num_tests):
        mean_acc = np.nanmean(all_acc[:, :, test], axis=0)
        sem = nansem(all_acc[:, :, test], axis=0)
        ax.errorbar(
            time_points_seconds,
            mean_acc,
            yerr=sem,
            marker="o",
            label=test_names[test],
            color=colors[test],
            alpha=0.7,
            linewidth=2,
        )

    ax.axhline(y=1.0 / 3.0, color="k", linestyle="--", label="Chance (1/3)", linewidth=2)
    add_phase_backgrounds(ax)
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Time (seconds)")
    ax.set_title("Decoding Accuracy - All Subjects")
    ax.legend(fontsize=9, loc="best")
    ax.set_ylim([0.25, 0.55])
    ax.grid(True, alpha=0.3)

    # Plot 2: winner vs loser for self-current.
    ax = axes[0, 1]
    test_to_plot = 0
    mean_winner = np.nanmean(wl[:, :, 0, test_to_plot], axis=0)
    mean_loser = np.nanmean(wl[:, :, 1, test_to_plot], axis=0)
    sem_winner = nansem(wl[:, :, 0, test_to_plot], axis=0)
    sem_loser = nansem(wl[:, :, 1, test_to_plot], axis=0)

    ax.errorbar(time_points_seconds, mean_winner, yerr=sem_winner, marker="o", label="Winner", color="green", alpha=0.7, linewidth=2)
    ax.errorbar(time_points_seconds, mean_loser, yerr=sem_loser, marker="s", label="Loser", color="red", alpha=0.7, linewidth=2)
    ax.axhline(y=1.0 / 3.0, color="k", linestyle="--", linewidth=2)
    add_phase_backgrounds(ax)
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Time (seconds)")
    ax.set_title("Self Current - Winner vs Loser")
    ax.legend()
    ax.set_ylim([0.25, 0.55])
    ax.grid(True, alpha=0.3)

    # Plot 3: Bayes factors (log10 scale).
    ax = axes[1, 0]
    for test in range(num_tests):
        ax.plot(time_points_seconds, np.log10(bf[test, :]), marker="o", label=test_names[test], color=colors[test], alpha=0.7, linewidth=2)
    ax.axhline(y=0.0, color="k", linestyle="-", linewidth=1)
    ax.axhline(y=np.log10(3.0), color="k", linestyle="--", label="BF = 3", linewidth=1, alpha=0.5)
    ax.axhline(y=np.log10(1.0 / 3.0), color="k", linestyle="--", linewidth=1, alpha=0.5)
    add_phase_backgrounds(ax)
    ax.set_ylabel("log10(BF)")
    ax.set_xlabel("Time (seconds)")
    ax.set_title("Bayes Factors Across Time (BIC Approx.)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Plot 4: summary.
    ax = axes[1, 1]
    ax.axis("off")

    mean_acc_all = np.nanmean(all_acc)
    std_acc_all = np.nanstd(all_acc)
    max_flat = np.nanargmax(all_acc) if np.any(np.isfinite(all_acc)) else None
    if max_flat is not None:
        max_idx = np.unravel_index(max_flat, all_acc.shape)
        max_acc = all_acc[max_idx]
        max_label = test_names[max_idx[2]]
    else:
        max_acc = np.nan
        max_label = "n/a"

    summary_text = "Summary Statistics:\n\n"
    summary_text += f"Overall Accuracy:\n- Mean: {mean_acc_all:.3f} +- {std_acc_all:.3f}\n"
    summary_text += f"- Max: {max_acc:.3f} (Test {max_label})\n"
    summary_text += "- Chance: 0.333\n\nPerformance by Test:\n"

    for test in range(num_tests):
        mean_test = np.nanmean(all_acc[:, :, test])
        max_test = np.nanmax(all_acc[:, :, test]) if np.any(np.isfinite(all_acc[:, :, test])) else np.nan
        summary_text += f"\n  {test_names[test]}:\n    Mean: {mean_test:.3f}, Max: {max_test:.3f}"

    summary_text += "\n\nWinner vs Loser (Self Current):"
    summary_text += f"\n  Winner mean: {np.nanmean(wl[:, :, 0, 0]):.3f}"
    summary_text += f"\n  Loser mean: {np.nanmean(wl[:, :, 1, 0]):.3f}"
    summary_text += f"\n\nBF(W-L diff) median (self current): {np.nanmedian(bf_wl[2, :, 0]):.3f}"

    ax.text(
        0.05,
        0.95,
        summary_text,
        fontsize=10,
        verticalalignment="top",
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()
    out_summary = os.path.join(plot_dir, "decoding_results.png")
    plt.savefig(out_summary, dpi=300, bbox_inches="tight")
    print(f"Saved: {out_summary}")
    plt.close()

    # Per-test plots.
    for test in range(num_tests):
        fig, ax = plt.subplots(figsize=(12, 6))
        mean_acc = np.nanmean(all_acc[:, :, test], axis=0)
        sem = nansem(all_acc[:, :, test], axis=0)

        ax.bar(time_points_seconds - 0.1, mean_acc, width=0.2, alpha=0.7, color=colors[test])
        ax.errorbar(time_points_seconds, mean_acc, yerr=sem, fmt="none", ecolor="black", capsize=5)
        ax.axhline(y=1.0 / 3.0, color="k", linestyle="--", linewidth=2, label="Chance")
        add_phase_backgrounds(ax)

        ax.set_ylabel("Accuracy", fontsize=12)
        ax.set_xlabel("Time (seconds)", fontsize=12)
        ax.set_title(f"{test_names[test]} - Decoding Accuracy", fontsize=14, fontweight="bold")
        ax.set_ylim([0.25, 0.55])
        ax.grid(True, alpha=0.3, axis="y")
        ax.legend()

        plt.tight_layout()
        out_test = os.path.join(plot_dir, f"decoding_test{test}_{test_names[test].replace(' ', '_')}.png")
        plt.savefig(out_test, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved: {out_test}")

    print("Decoding plots completed successfully!")


if __name__ == "__main__":
    main()