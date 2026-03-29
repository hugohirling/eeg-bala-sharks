"""
Markov chain analysis (Python version of step2b_markovchain.m):
   - Predict response based on N previous trials
   - Calculate accuracy for different window sizes (5-100 trials)
"""

import os
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

# Set the path
path_to_data = 'MNE-sample-data/ds006761'

# Set parameters
pair_ids = np.concatenate([np.arange(1, 10), np.arange(11, 23), np.arange(25, 35)])
num_pairs = len(pair_ids)
num_trials = 480
num_windows = 100

# Pre-allocate output
# Mean_Accuracy: (num_pairs, 2 players, num_windows)
Mean_Accuracy = np.zeros((num_pairs, 2, num_windows))

# M_pred: (num_pairs, 2 players, num_windows, num_trials, 4 columns)
# Columns: actual, predicted, probability, accuracy
M_pred = np.full((num_pairs, 2, num_windows, num_trials, 4), np.nan)

# Load behavioral data
for p in range(num_pairs):
    pair = pair_ids[p]
    print(f'Loading pair {p + 1} of {num_pairs}: sub-{pair:02d}')

    # Load behavioral events
    events_filename = os.path.join(path_to_data, f'sub-{pair:02d}', 'eeg', f'sub-{pair:02d}_task-RPS_events.tsv')
    events = pd.read_csv(events_filename, sep='\t')

    # Loop over the 2 players in the pair
    for ppt in range(1, 3):
        # Get response for this player (Rock=1, Paper=2, Scissors=3)
        if ppt == 1:
            resp = events['player1_resp'].values
        else:
            resp = events['player2_resp'].values

        # Pre-allocate prob_data matrix
        # Columns: trial, N_Rock, R_R, R_P, R_S, N_Paper, P_R, P_P, P_S, N_Scissors, S_R, S_P, S_S
        prob_data = np.full((num_trials, 13), np.nan)
        
        # Initialize first trial with uniform priors
        prob_data[0, :] = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

        # Build transition probability matrix
        for i in range(1, num_trials):
            prob_data[i, :] = prob_data[i - 1, :]
            prob_data[i, 0] = i + 1  # Trial number

            # Update counts
            if resp[i - 1] == 1:  # Previous was Rock
                prob_data[i, 1] += 1
                if resp[i] == 1:
                    prob_data[i, 2] += 1
                elif resp[i] == 2:
                    prob_data[i, 3] += 1
                else:
                    prob_data[i, 4] += 1

            elif resp[i - 1] == 2:  # Previous was Paper
                prob_data[i, 5] += 1
                if resp[i] == 1:
                    prob_data[i, 6] += 1
                elif resp[i] == 2:
                    prob_data[i, 7] += 1
                else:
                    prob_data[i, 8] += 1

            elif resp[i - 1] == 3:  # Previous was Scissors
                prob_data[i, 9] += 1
                if resp[i] == 1:
                    prob_data[i, 10] += 1
                elif resp[i] == 2:
                    prob_data[i, 11] += 1
                else:
                    prob_data[i, 12] += 1

        # Markov chain prediction with different window sizes
        for window_size in range(5, num_windows + 1):
            prob_res = np.full((num_trials, 4), np.nan)
            m_Prob = np.array([[1/3, 1/3, 1/3], [1/3, 1/3, 1/3], [1/3, 1/3, 1/3]])

            for i in range(2, num_trials):
                # Get change in this window
                if i < window_size + 1:
                    inter_prob_data = prob_data[i - 1, :]
                else:
                    inter_prob_data = prob_data[i - 1, :] - prob_data[i - window_size - 1, :]
                
                inter_prob_data[0] = i + 1

                # Calculate transition probabilities
                # For Rock (came after)
                if inter_prob_data[1] > 0:
                    m_Prob[0, :] = [
                        inter_prob_data[2] / inter_prob_data[1],
                        inter_prob_data[3] / inter_prob_data[1],
                        inter_prob_data[4] / inter_prob_data[1]
                    ]
                else:
                    m_Prob[0, :] = [1/3, 1/3, 1/3]

                # For Paper (came after)
                if inter_prob_data[5] > 0:
                    m_Prob[1, :] = [
                        inter_prob_data[6] / inter_prob_data[5],
                        inter_prob_data[7] / inter_prob_data[5],
                        inter_prob_data[8] / inter_prob_data[5]
                    ]
                else:
                    m_Prob[1, :] = [1/3, 1/3, 1/3]

                # For Scissors (came after)
                if inter_prob_data[9] > 0:
                    m_Prob[2, :] = [
                        inter_prob_data[10] / inter_prob_data[9],
                        inter_prob_data[11] / inter_prob_data[9],
                        inter_prob_data[12] / inter_prob_data[9]
                    ]
                else:
                    m_Prob[2, :] = [1/3, 1/3, 1/3]

                # Make prediction
                prob_res[i, 0] = resp[i]  # Actual response

                # Find previous response (handle missing responses)
                if resp[i - 1] > 0:
                    idx = i - 1
                elif i > 1 and resp[i - 2] > 0:
                    idx = i - 2
                else:
                    idx = i - 1 if i > 0 else None

                if idx is not None and resp[idx] > 0:
                    last_resp = int(resp[idx])
                    idx_max = np.argmax(m_Prob[last_resp - 1, :])
                    prob_res[i, 1] = idx_max + 1  # Predicted response (1-3)
                    prob_res[i, 2] = m_Prob[last_resp - 1, idx_max]  # Probability

                    # Check accuracy
                    if np.isnan(prob_res[i, 2]):
                        prob_res[i, 3] = np.nan
                    elif prob_res[i, 0] == prob_res[i, 1]:
                        prob_res[i, 3] = 1
                    else:
                        prob_res[i, 3] = 0

                # Calculate window accuracy
                valid_data = prob_res[2:, 3]
                valid_data = valid_data[~np.isnan(valid_data)]
                if len(valid_data) > 0:
                    Mean_Accuracy[p, ppt - 1, window_size - 1] = np.mean(valid_data)

                M_pred[p, ppt - 1, window_size - 1, :, :] = prob_res

print('Markov chain analysis completed!')

# Save output
import scipy.io as sio
save_dict = {
    'M_pred': M_pred,
    'Mean_Accuracy': Mean_Accuracy
}
save_path = os.path.join(path_to_data, 'derivatives', 'markov_chain_pred.mat')
os.makedirs(os.path.dirname(save_path), exist_ok=True)
sio.savemat(save_path, save_dict)

print(f'Results saved to {save_path}')