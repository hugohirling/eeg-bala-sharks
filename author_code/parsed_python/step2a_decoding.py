"""
Decoding script (Python version of step2a_decoding.m):
   - Decode own & opponent response for current & previous trial.

This implementation follows the MATLAB workflow more closely:
  - average re-reference
  - split into decision/response/feedback parts
  - baseline-correct each part on [-0.2, 0]
  - drop first trial of each 40-trial block
  - 250 ms binning using strict bin boundaries (start < t < end)
  - balanced sample averaging (count=4, repeats=20)
  - time decoding + channel searchlight decoding with nearest neighbors
"""

import os
import warnings
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from scipy.io import savemat
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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

# Set parameters
DEFAULT_PAIR_IDS = np.concatenate([np.arange(1, 10), np.arange(11, 23), np.arange(25, 35)])
num_trials = 480
num_chan = 64
num_time_bins = 20


def _is_truthy(value):
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


FORCE_REPROCESS = _is_truthy(os.environ.get("RPS_FORCE_REPROCESS", "0"))


def _resolve_pair_ids(default_ids):
    raw = os.environ.get("RPS_SUBJECTS") or os.environ.get("EEG_SUBJECTS")
    if not raw:
        return default_ids
    parsed = []
    for token in raw.split(","):
        token = token.strip().lower().replace("sub-", "")
        if not token:
            continue
        try:
            parsed.append(int(token))
        except ValueError:
            print(f"Warning: invalid subject token ignored: {token}")
    if not parsed:
        return default_ids
    return np.array([sid for sid in parsed if sid in set(default_ids)], dtype=int)


pair_ids = _resolve_pair_ids(DEFAULT_PAIR_IDS)
num_pairs = len(pair_ids)


def build_behav_table(events_df):
    """Build behavioral table equivalent to MATLAB columns self/other/result/selfp/otherp."""
    behav_data = events_df[["player1_resp", "player2_resp", "outcome"]].to_numpy()
    prev_cols = np.vstack([np.full((1, 2), np.nan), behav_data[:-1, :2]])

    player_1 = np.hstack([behav_data, prev_cols])
    player_2 = np.hstack([
        behav_data[:, [1, 0]],
        np.zeros((len(behav_data), 1)),
        np.vstack([np.full((1, 2), np.nan), behav_data[:-1, [1, 0]]]),
    ])

    # Recode outcome for player 2 to be relative to player 2.
    player_2[behav_data[:, 2] == 1, 2] = 1
    player_2[behav_data[:, 2] == 2, 2] = 3
    player_2[behav_data[:, 2] == 3, 2] = 2

    return np.dstack([player_1, player_2])


def average_reference(eeg):
    """Apply average reference across channels."""
    return eeg - eeg.mean(axis=1, keepdims=True)


def baseline_correct_part(eeg_part, part_times):
    """Baseline-correct one part using [-0.2, 0] on part-relative time."""
    mask = (part_times >= -0.2) & (part_times <= 0)
    if not np.any(mask):
        return eeg_part
    baseline = eeg_part[:, :, mask].mean(axis=2, keepdims=True)
    return eeg_part - baseline


def make_time_windows():
    """Match MATLAB bin edges.

    AB: 0:0.25:2 (8 bins)
    C:  0:0.25:1 (4 bins)
    """
    starts_ab = np.arange(0.0, 2.0, 0.25)
    ends_ab = np.arange(0.25, 2.25, 0.25)
    windows_ab = np.column_stack([starts_ab, ends_ab])

    starts_c = np.arange(0.0, 1.0, 0.25)
    ends_c = np.arange(0.25, 1.25, 0.25)
    windows_c = np.column_stack([starts_c, ends_c])
    return windows_ab, windows_c


def bin_part(eeg_part, part_times, windows):
    """Average part into bins using MATLAB-style strict boundaries (> start, < end)."""
    n_trials, n_ch, _ = eeg_part.shape
    out = np.full((n_trials, n_ch, windows.shape[0]), np.nan)
    for wi, (w_start, w_end) in enumerate(windows):
        mask = (part_times > w_start) & (part_times < w_end)
        if np.any(mask):
            out[:, :, wi] = eeg_part[:, :, mask].mean(axis=2)
    return out


def balanced_average_samples(X, y, count=4, repeats=20, seed=1):
    """Approximate cosmo_average_samples with class-balanced grouped averaging."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    classes = np.unique(y)

    X_out = []
    y_out = []
    chunks_out = []

    for rep in range(repeats):
        rep_samples = []
        rep_targets = []
        for cls in classes:
            idx = np.where(y == cls)[0]
            rng.shuffle(idx)
            n_groups = len(idx) // count
            if n_groups < 1:
                continue
            idx_use = idx[: n_groups * count].reshape(n_groups, count)
            cls_avg = X[idx_use].mean(axis=1)
            rep_samples.append(cls_avg)
            rep_targets.append(np.full(n_groups, cls))

        if not rep_samples:
            continue

        X_rep = np.vstack(rep_samples)
        y_rep = np.concatenate(rep_targets)

        rep_order = rng.permutation(len(y_rep))
        X_rep = X_rep[rep_order]
        y_rep = y_rep[rep_order]

        X_out.append(X_rep)
        y_out.append(y_rep)
        chunks_out.append(np.full(len(y_rep), rep, dtype=int))

    if not X_out:
        raise ValueError("Could not create averaged samples (too few trials per class).")

    return np.vstack(X_out), np.concatenate(y_out), np.concatenate(chunks_out)


def run_lda_cv(X, y, cv_folds=10):
    y = np.asarray(y)
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2 or np.min(counts) < 2:
        raise ValueError("Not enough samples per class for stratified CV")

    n_splits = min(cv_folds, np.min(counts))
    if n_splits < 2:
        raise ValueError("Not enough samples to perform cross-validation")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lda", LinearDiscriminantAnalysis()),
    ])

    scores = []
    for train_idx, test_idx in skf.split(X, y):
        pipe.fit(X[train_idx], y[train_idx])
        scores.append(pipe.score(X[test_idx], y[test_idx]))
    return float(np.mean(scores))


def get_channel_neighbors(epochs, max_neighbors=4):
    """Build channel neighborhoods (self + nearest channels)."""
    ch_names = epochs.ch_names[:num_chan]
    pos = []
    for ch in ch_names:
        idx = epochs.ch_names.index(ch)
        loc = epochs.info["chs"][idx]["loc"][:3]
        pos.append(loc)
    pos = np.asarray(pos)

    # Fallback-safe distance matrix.
    d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)

    neighborhoods = []
    for ci in range(len(ch_names)):
        nearest = np.argsort(d[ci])[:max_neighbors]
        neighborhood = np.unique(np.concatenate([[ci], nearest]))
        neighborhoods.append(neighborhood)
    return neighborhoods


def decode_time_and_searchlight(X_binned, y_targets, neighborhoods):
    """Run time decoding and channel searchlight decoding for one target variable."""
    valid_idx = y_targets > 0
    X = X_binned[valid_idx]
    y = y_targets[valid_idx].astype(int)

    if len(np.unique(y)) < 3:
        raise ValueError("Not enough classes after removing no-responses.")

    time_acc = np.full((num_time_bins,), np.nan)
    sl_acc = np.full((num_chan, num_time_bins), np.nan)

    for tb in range(num_time_bins):
        # Whole-head decoding at this time bin.
        X_tb = X[:, :, tb]
        X_avg, y_avg, _ = balanced_average_samples(X_tb, y, count=4, repeats=20, seed=1)
        time_acc[tb] = run_lda_cv(X_avg, y_avg, cv_folds=10)

        # Channel searchlight decoding at this time bin.
        for ch_idx, neigh in enumerate(neighborhoods):
            X_neigh = X_tb[:, neigh]
            X_neigh = X_neigh.reshape(X_neigh.shape[0], -1)
            Xn_avg, yn_avg, _ = balanced_average_samples(X_neigh, y, count=4, repeats=20, seed=1)
            sl_acc[ch_idx, tb] = run_lda_cv(Xn_avg, yn_avg, cv_folds=10)

    return time_acc, sl_acc


def process_player(pair, player, behav_data):
    """Process one player for one pair and save decoding output .mat."""
    print(f"   Player {player}")

    out_path = os.path.join(derivatives_path, f"pair-{pair:02d}_player-{player}_task-RPS_decoding.mat")
    if os.path.exists(out_path) and not FORCE_REPROCESS:
        print(f"   Decoding already exists, skipping: {out_path}")
        return

    epo_path = os.path.join(derivatives_path, f"pair-{pair:02d}_player-{player}_task-RPS_eeg-epo.fif")
    if not os.path.exists(epo_path):
        print(f"   File not found: {epo_path}. Skipping.")
        return

    epochs = mne.read_epochs(epo_path, preload=True)
    eeg = epochs.get_data()[:, :num_chan, :]
    eeg = average_reference(eeg)
    times = epochs.times

    # Match behavioral and EEG trial counts.
    n_trials_eeg = eeg.shape[0]
    behav = behav_data[:, :, player - 1]
    n_trials = min(n_trials_eeg, behav.shape[0], num_trials)
    eeg = eeg[:n_trials]
    behav = behav[:n_trials]

    # Split into 3 parts.
    idx_a = (times >= -0.2) & (times <= 2.0)
    idx_b = (times >= 1.8) & (times <= 4.0)
    idx_c = (times >= 3.8) & (times <= 5.0)

    eeg_a = eeg[:, :, idx_a]
    eeg_b = eeg[:, :, idx_b]
    eeg_c = eeg[:, :, idx_c]

    times_a = times[idx_a]
    # MATLAB resets part B/C time vectors to start at -0.2.
    times_b = times_a.copy()
    times_c = times_a[: eeg_c.shape[2]].copy()

    # Baseline correction for each part separately using [-0.2, 0].
    eeg_a = baseline_correct_part(eeg_a, times_a)
    eeg_b = baseline_correct_part(eeg_b, times_b)
    eeg_c = baseline_correct_part(eeg_c, times_c)

    # Remove first trial of each block (1:40:480 in MATLAB, 0-based here).
    rem_idx = np.arange(0, n_trials, 40)
    keep_idx = np.setdiff1d(np.arange(n_trials), rem_idx)
    if keep_idx.size == 0:
        print("   No trials left after block trimming. Skipping.")
        return

    eeg_a = eeg_a[keep_idx]
    eeg_b = eeg_b[keep_idx]
    eeg_c = eeg_c[keep_idx]
    behav = behav[keep_idx]

    # 250 ms binning and recombine A/B/C into 20 bins.
    windows_ab, windows_c = make_time_windows()
    binned_a = bin_part(eeg_a, times_a, windows_ab)
    binned_b = bin_part(eeg_b, times_b, windows_ab)
    binned_c = bin_part(eeg_c, times_c, windows_c)
    X_binned = np.concatenate([binned_a, binned_b, binned_c], axis=2)

    neighborhoods = get_channel_neighbors(epochs, max_neighbors=4)

    target_cols = {
        0: 0,  # self
        1: 1,  # other
        2: 3,  # self previous
        3: 4,  # other previous
    }

    mat_dict = {}
    for test in range(4):
        y = behav[:, target_cols[test]]
        try:
            time_acc, sl_acc = decode_time_and_searchlight(X_binned, y, neighborhoods)
            print(f"   Test {test}: mean accuracy = {np.nanmean(time_acc):.3f}")
        except Exception as exc:
            print(f"   Test {test}: failed ({exc})")
            time_acc = np.full((num_time_bins,), np.nan)
            sl_acc = np.full((num_chan, num_time_bins), np.nan)

        mat_dict[f"decoding_acc_test{test}"] = time_acc
        mat_dict[f"searchlight_test{test}"] = sl_acc

    savemat(out_path, mat_dict)
    print(f"   Saved: {out_path}")


def main():
    for p, pair in enumerate(pair_ids, start=1):
        pair_out_1 = os.path.join(derivatives_path, f"pair-{pair:02d}_player-1_task-RPS_decoding.mat")
        pair_out_2 = os.path.join(derivatives_path, f"pair-{pair:02d}_player-2_task-RPS_decoding.mat")
        if os.path.exists(pair_out_1) and os.path.exists(pair_out_2) and not FORCE_REPROCESS:
            print(f"Skipping sub-{pair:02d}: both player decoding files already exist.")
            continue

        print(f"Loading pair {p} of {num_pairs}: sub-{pair:02d}")
        events_file = os.path.join(path_to_data, f"sub-{pair:02d}", "eeg", f"sub-{pair:02d}_task-RPS_events.tsv")
        if not os.path.exists(events_file):
            print(f"Warning: skipping sub-{pair:02d}, events missing: {events_file}")
            continue
        try:
            events = pd.read_csv(events_file, sep="\t")
        except Exception as exc:
            print(f"Warning: skipping sub-{pair:02d}, events unreadable: {exc}")
            continue
        behav_data = build_behav_table(events)

        for player in (1, 2):
            process_player(pair, player, behav_data)

    print("Decoding analysis completed successfully!")


if __name__ == "__main__":
    main()