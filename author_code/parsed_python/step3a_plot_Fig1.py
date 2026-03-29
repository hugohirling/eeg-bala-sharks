"""
Plot behavioral responses (Python version of step3a_plot_Fig1.m):
   - Outcome summary (winner/loser/draw)
   - Response rank usage
   - Change-after-outcome rates
   - Markov predictability vs window size
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
plot_dir = str(OUTPUT_ROOT / "plots")
os.makedirs(plot_dir, exist_ok=True)

# Set parameters
pair_ids = np.concatenate([np.arange(1, 10), np.arange(11, 23), np.arange(25, 35)])
num_pairs = len(pair_ids)
num_blocks = 12
num_trials_per_block = 40
num_trials = num_blocks * num_trials_per_block
pair_idx0 = np.arange(num_pairs * 2).reshape(num_pairs, 2)

# Pre-allocate outputs
outcome_summary = np.full((num_pairs, 3), np.nan)
all_played_rank = np.full((3, num_pairs * 2), np.nan)
ranked_resp = np.full((3, num_pairs * 2), np.nan)
prop_stay = np.full((3, num_pairs * 2), np.nan)  # row0=after win, row1=after lose, row2=after draw


def _winner_idx(events):
    p1_win = int((events["outcome"] == 2).sum())
    p2_win = int((events["outcome"] == 3).sum())
    if p1_win > p2_win:
        return 1  # MATLAB winner_idx=1
    if p1_win < p2_win:
        return 2  # MATLAB winner_idx=2
    return 3      # tie


def _percent_per_rps(values):
    values = np.asarray(values)
    out = np.zeros((3,), dtype=float)
    if values.size == 0:
        return out
    for r in (1, 2, 3):
        out[r - 1] = 100.0 * np.mean(values == r)
    return out


for p, pair in enumerate(pair_ids):
    print(f"Analyzing pair {p + 1} of {num_pairs}: sub-{pair:02d}")

    events_filename = os.path.join(path_to_data, f"sub-{pair:02d}", "eeg", f"sub-{pair:02d}_task-RPS_events.tsv")
    if not os.path.exists(events_filename):
        print(f"Warning: missing events file for sub-{pair:02d}: {events_filename}")
        continue

    try:
        events = pd.read_csv(events_filename, sep="\t")
    except Exception as exc:
        print(f"Warning: could not read events for sub-{pair:02d}: {exc}")
        continue

    if len(events) != num_trials:
        print(f"Warning: sub-{pair:02d} has {len(events)} trials (expected {num_trials}); skipping pair.")
        continue

    winner_idx = _winner_idx(events)
    if winner_idx == 3:
        print(f"Warning: no winner for pair sub-{pair:02d}")

    # Remove no-response trials for outcome + played-response summaries.
    events_r = events[(events["player1_resp"] > 0) & (events["player2_resp"] > 0)].copy()
    if events_r.empty:
        print(f"Warning: no valid response trials for sub-{pair:02d}; skipping pair.")
        continue

    # Base order in MATLAB before sorting: [draw, p1_wins, p2_wins]
    base_outcome = np.array([
        np.mean(events_r["outcome"] == 1),
        np.mean(events_r["outcome"] == 2),
        np.mean(events_r["outcome"] == 3),
    ]) * 100.0

    if winner_idx in (1, 2):
        loser_idx = 1 if winner_idx == 2 else 2
        # MATLAB equivalent: [draw, winner_wins, loser_wins]
        order = [0, winner_idx, loser_idx]
        outcome_summary[p, :] = base_outcome[order]
    else:
        outcome_summary[p, :] = base_outcome

    # Response usage per player on valid-response trials.
    played = np.column_stack([events_r["player1_resp"].to_numpy(), events_r["player2_resp"].to_numpy()])
    all_played = np.zeros((3, 2), dtype=float)

    for ppt in (0, 1):
        percentages = _percent_per_rps(played[:, ppt])
        all_played[:, ppt] = percentages

        # Rank response IDs by usage (1=R,2=P,3=S)
        rank_idx = np.argsort(-percentages) + 1
        all_played_rank[:, pair_idx0[p, ppt]] = rank_idx

    ranked_sorted = np.sort(all_played, axis=0)[::-1, :]
    ranked_resp[:, pair_idx0[p, 0]] = ranked_sorted[:, 0]
    ranked_resp[:, pair_idx0[p, 1]] = ranked_sorted[:, 1]

    # Build behavior tables per player for stay/change analysis (without removing null responses).
    p1 = events[["player1_resp", "player2_resp", "outcome"]].to_numpy().copy()
    p2 = events[["player2_resp", "player1_resp", "outcome"]].to_numpy().copy()
    p2[p1[:, 2] == 1, 2] = 1
    p2[p1[:, 2] == 2, 2] = 3
    p2[p1[:, 2] == 3, 2] = 2

    # MATLAB uses blocks of 40 sequential trials.
    p1_blocks = p1.reshape(num_blocks, num_trials_per_block, 3)
    p2_blocks = p2.reshape(num_blocks, num_trials_per_block, 3)

    # Collect stay arrays as in MATLAB naming.
    p1_draw, p1_win, p1_lose = [], [], []
    p2_draw, p2_win, p2_lose = [], [], []

    for block_num in range(num_blocks):
        for trial_num in range(1, num_trials_per_block):
            prev_idx = trial_num - 1

            # MATLAB gate: current and previous responses of both players must be >0.
            valid_transition = (
                p1_blocks[block_num, prev_idx, 0] > 0
                and p1_blocks[block_num, trial_num, 0] > 0
                and p2_blocks[block_num, prev_idx, 0] > 0
                and p2_blocks[block_num, trial_num, 0] > 0
            )
            if not valid_transition:
                continue

            # Player 1
            p1_prev_outcome = int(p1_blocks[block_num, prev_idx, 2])
            p1_stay = int(p1_blocks[block_num, trial_num, 0] == p1_blocks[block_num, prev_idx, 0])
            if p1_prev_outcome == 1:
                p1_draw.append(p1_stay)
            elif p1_prev_outcome == 2:
                p1_win.append(p1_stay)
            elif p1_prev_outcome == 3:
                p1_lose.append(p1_stay)

            # Player 2
            p2_prev_outcome = int(p2_blocks[block_num, prev_idx, 2])
            p2_stay = int(p2_blocks[block_num, trial_num, 0] == p2_blocks[block_num, prev_idx, 0])
            if p2_prev_outcome == 1:
                p2_draw.append(p2_stay)
            elif p2_prev_outcome == 2:
                p2_win.append(p2_stay)
            elif p2_prev_outcome == 3:
                p2_lose.append(p2_stay)

    # MATLAB row mapping:
    # row 1 -> after win, row 2 -> after lose, row 3 -> after draw
    def _pct_or_nan(arr):
        return 100.0 * np.mean(arr) if len(arr) > 0 else np.nan

    prop_stay[0, pair_idx0[p, 0]] = _pct_or_nan(p1_win)
    prop_stay[1, pair_idx0[p, 0]] = _pct_or_nan(p1_lose)
    prop_stay[2, pair_idx0[p, 0]] = _pct_or_nan(p1_draw)

    prop_stay[0, pair_idx0[p, 1]] = _pct_or_nan(p2_win)
    prop_stay[1, pair_idx0[p, 1]] = _pct_or_nan(p2_lose)
    prop_stay[2, pair_idx0[p, 1]] = _pct_or_nan(p2_draw)

# Markov predictability panel data.
markov_file = str(OUTPUT_ROOT / "markov_chain_pred.mat")
pred_acc = None
if os.path.exists(markov_file):
    md = loadmat(markov_file)
    mean_acc = md.get("Mean_Accuracy", None)
    if mean_acc is not None and mean_acc.ndim == 3:
        # Stack players: [pairs x windows] -> [pairs*2 x windows]
        pred_acc = np.vstack([mean_acc[:, 0, :], mean_acc[:, 1, :]]) * 100.0
        pred_acc = pred_acc[:, 5:100]  # MATLAB keeps 5:100

# Build a MATLAB-inspired 2x2 summary figure.
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle("Behavioral Summary (MATLAB-aligned metrics)", fontsize=14, fontweight="bold")

# A) Outcomes (winner/loser/draw)
ax = axes[0, 0]
ax.boxplot(
    [outcome_summary[:, 1], outcome_summary[:, 2], outcome_summary[:, 0]],
    labels=["Winner wins", "Loser wins", "Draw"],
    showfliers=False,
)
ax.axhline(100.0 / 3.0, color="k", linestyle="--", linewidth=1)
ax.set_ylabel("Percentage")
ax.set_ylim([18, 48])
ax.set_title("Outcomes")

# B) Response rank usage (most/mid/least)
ax = axes[0, 1]
ax.boxplot(
    [ranked_resp[0, :], ranked_resp[1, :], ranked_resp[2, :]],
    labels=["Most played", "Mid played", "Least played"],
    showfliers=False,
)
ax.axhline(100.0 / 3.0, color="k", linestyle="--", linewidth=1)
ax.set_ylabel("Percentage")
ax.set_ylim([18, 48])
ax.set_title("Response Usage")

# C) Change after previous outcome
ax = axes[1, 0]
prop_change = 100.0 - prop_stay
ax.boxplot(
    [prop_change[0, :], prop_change[1, :], prop_change[2, :]],
    labels=["After win", "After loss", "After draw"],
    showfliers=False,
)
ax.axhline(200.0 / 3.0, color="k", linestyle="--", linewidth=1)
ax.set_ylabel("Percentage")
ax.set_ylim([20, 103])
ax.set_title("Change Strategy")

# D) Markov predictability
ax = axes[1, 1]
if pred_acc is not None:
    x = np.arange(5, 100)
    for row in pred_acc:
        ax.plot(x, row, color=(0.3, 0.3, 0.3, 0.15), linewidth=1)

    mean_curve = np.nanmean(pred_acc, axis=0)
    sem_curve = np.nanstd(pred_acc, axis=0) / np.sqrt(np.sum(np.isfinite(pred_acc), axis=0))
    ax.fill_between(x, mean_curve - sem_curve, mean_curve + sem_curve, color="C0", alpha=0.2)
    ax.plot(x, mean_curve, color="C0", linewidth=2)

    ax.axhline(100.0 / 3.0, color="k", linestyle="--", linewidth=1)
    ax.set_ylim([25, 65])
    ax.set_xlim([5, 99])
else:
    ax.text(0.5, 0.5, "markov_chain_pred.mat not found", ha="center", va="center")

ax.set_xlabel("N previous games")
ax.set_ylabel("Accuracy (%)")
ax.set_title("Predictability")

plt.tight_layout()
out_path = os.path.join(plot_dir, "Figure1.png")
plt.savefig(out_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"Saved: {out_path}")
print("Behavioral plots completed!")
