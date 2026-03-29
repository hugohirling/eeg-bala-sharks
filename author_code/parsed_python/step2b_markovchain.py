"""
Markov chain analysis (Python version of step2b_markovchain.m):
   - Predict response based on N previous trials
   - Calculate accuracy for different window sizes (5-100 trials)
"""

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

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

# Set parameters
DEFAULT_PAIR_IDS = np.concatenate([np.arange(1, 10), np.arange(11, 23), np.arange(25, 35)])
num_trials = 480
num_windows = 100


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

# Pre-allocate output (MATLAB-compatible shapes)
Mean_Accuracy = np.zeros((num_pairs, 2, num_windows))
M_pred = np.zeros((num_pairs, 2, num_windows, num_trials, 4))

for p, pair in enumerate(pair_ids):
    print(f"Loading pair {p + 1} of {num_pairs}: sub-{pair:02d}")

    events_filename = os.path.join(path_to_data, f"sub-{pair:02d}", "eeg", f"sub-{pair:02d}_task-RPS_events.tsv")
    if not os.path.exists(events_filename):
        print(f"Warning: skipping sub-{pair:02d}, events missing: {events_filename}")
        continue
    try:
        events = pd.read_csv(events_filename, sep="\t")
    except Exception as exc:
        print(f"Warning: skipping sub-{pair:02d}, events unreadable: {exc}")
        continue

    for ppt in (1, 2):
        resp = events["player1_resp"].to_numpy() if ppt == 1 else events["player2_resp"].to_numpy()

        # Column mapping:
        # 1 Trial, 2 N_Rock, 3 R_R, 4 R_P, 5 R_S,
        # 6 N_Paper, 7 P_R, 8 P_P, 9 P_S,
        # 10 N_Scissors, 11 S_R, 12 S_P, 13 S_S
        prob_data = np.full((num_trials, 13), np.nan)
        prob_data[0, :] = np.array([1, 3, 1, 1, 1, 3, 1, 1, 1, 3, 1, 1, 1], dtype=float)

        # Build cumulative transition counts.
        for i in range(1, num_trials):
            prob_data[i, :] = prob_data[i - 1, :]
            prob_data[i, 0] = i + 1

            prev_r = resp[i - 1]
            cur_r = resp[i]

            if prev_r == 1:
                prob_data[i, 1] = prob_data[i - 1, 1] + 1
                if cur_r == 1:
                    prob_data[i, 2] = prob_data[i - 1, 2] + 1
                elif cur_r == 2:
                    prob_data[i, 3] = prob_data[i - 1, 3] + 1
                else:
                    prob_data[i, 4] = prob_data[i - 1, 4] + 1
            elif prev_r == 2:
                prob_data[i, 5] = prob_data[i - 1, 5] + 1
                if cur_r == 1:
                    prob_data[i, 6] = prob_data[i - 1, 6] + 1
                elif cur_r == 2:
                    prob_data[i, 7] = prob_data[i - 1, 7] + 1
                else:
                    prob_data[i, 8] = prob_data[i - 1, 8] + 1
            elif prev_r == 3:
                prob_data[i, 9] = prob_data[i - 1, 9] + 1
                if cur_r == 1:
                    prob_data[i, 10] = prob_data[i - 1, 10] + 1
                elif cur_r == 2:
                    prob_data[i, 11] = prob_data[i - 1, 11] + 1
                else:
                    prob_data[i, 12] = prob_data[i - 1, 12] + 1

        # Windowed prediction.
        for window_size in range(5, num_windows + 1):
            prob_res = np.full((num_trials, 4), np.nan)
            m_prob = np.array([[1.0 / 3, 1.0 / 3, 1.0 / 3], [1.0 / 3, 1.0 / 3, 1.0 / 3], [1.0 / 3, 1.0 / 3, 1.0 / 3]])

            for i in range(2, num_trials):  # MATLAB i=3:num_trials
                if (i + 1) < (window_size + 1):
                    inter = prob_data[i - 1, :].copy()
                else:
                    # MATLAB: prob_data(i-1,:) - prob_data(i-window_size,:)
                    inter = prob_data[i - 1, :] - prob_data[i - window_size, :]

                inter[0] = i + 1

                if inter[1] > 0:
                    m_prob[0, :] = [inter[2] / inter[1], inter[3] / inter[1], inter[4] / inter[1]]
                else:
                    m_prob[0, :] = [1.0 / 3, 1.0 / 3, 1.0 / 3]

                if inter[5] > 0:
                    m_prob[1, :] = [inter[6] / inter[5], inter[7] / inter[5], inter[8] / inter[5]]
                else:
                    m_prob[1, :] = [1.0 / 3, 1.0 / 3, 1.0 / 3]

                if inter[9] > 0:
                    m_prob[2, :] = [inter[10] / inter[9], inter[11] / inter[9], inter[12] / inter[9]]
                else:
                    m_prob[2, :] = [1.0 / 3, 1.0 / 3, 1.0 / 3]

                # Column 1: actual response.
                prob_res[i, 0] = resp[i]

                # MATLAB-equivalent fallback indexing for missing responses.
                # Keep idx as 1-based to match original logic exactly.
                if resp[i - 1] > 0:
                    idx1 = i + 1
                elif resp[i - 2] > 0:
                    idx1 = i
                else:
                    idx1 = i - 1

                if idx1 > 1:
                    last_resp = resp[idx1 - 2]
                    if last_resp == 1:
                        idx_max = int(np.argmax(m_prob[0, :])) + 1
                        prob_res[i, 1] = idx_max
                        prob_res[i, 2] = np.max(m_prob[0, :])
                    elif last_resp == 2:
                        idx_max = int(np.argmax(m_prob[1, :])) + 1
                        prob_res[i, 1] = idx_max
                        prob_res[i, 2] = np.max(m_prob[1, :])
                    elif last_resp == 3:
                        idx_max = int(np.argmax(m_prob[2, :])) + 1
                        prob_res[i, 1] = idx_max
                        prob_res[i, 2] = np.max(m_prob[2, :])

                # Column 4: accuracy
                if np.isnan(prob_res[i, 2]):
                    prob_res[i, 3] = np.nan
                elif prob_res[i, 0] == prob_res[i, 1]:
                    prob_res[i, 3] = 1
                else:
                    prob_res[i, 3] = 0

                data_mean = prob_res[2:480, 3]
                data_mean = data_mean[np.isfinite(data_mean)]
                Mean_Accuracy[p, ppt - 1, window_size - 1] = np.mean(data_mean) if data_mean.size > 0 else np.nan

                M_pred[p, ppt - 1, window_size - 1, :, :] = prob_res

print("Markov chain analysis completed!")

save_path = str(OUTPUT_ROOT / "markov_chain_pred.mat")
os.makedirs(os.path.dirname(save_path), exist_ok=True)
sio.savemat(save_path, {"M_pred": M_pred, "Mean_Accuracy": Mean_Accuracy})

print(f"Results saved to {save_path}")
