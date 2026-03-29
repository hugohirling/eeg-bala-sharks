"""
Decoding script (Python version of step2a_decoding.m):
   - Decode own & opponent's response for current & previous trial

Uses LDA classifier with time-bin searchlight

Uses MNE-Python and scikit-learn instead of CoSMoMVPA.
"""

import os
import pandas as pd
import numpy as np
import mne
from scipy.io import savemat
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import warnings

warnings.filterwarnings('ignore')

# Set the path
path_to_data = 'MNE-sample-data/ds006761'
derivatives_path = os.path.join(path_to_data, 'derivatives')

# Set parameters
pair_ids = np.concatenate([np.arange(1, 10), np.arange(11, 23), np.arange(25, 35)])
num_pairs = len(pair_ids)
num_trials = 480
num_chan = 64
num_time_bins = 20

# Pre-allocate output
decoding_accuracy = {}
searchlight_acc = {}
pair_idx = np.arange(1, num_pairs * 2 + 1).reshape(-1, 2)

def resample_trials(data, num_resamples=20, samples_per_resample=4, seed=1):
    """Average samples to improve SNR and balance classes"""
    np.random.seed(seed)
    n_trials, n_chan, n_time = data.shape
    
    resampled_data = []
    targets_resampled = []
    
    for _ in range(num_resamples):
        # Randomly sample with replacement
        for _ in range(n_trials // samples_per_resample):
            idx = np.random.choice(n_trials, samples_per_resample, replace=True)
            avg_sample = data[idx].mean(axis=0)
            resampled_data.append(avg_sample)
    
    return np.array(resampled_data)

def create_time_neighborhood(times, radius=0):
    """Create neighborhood for timepoints (each timepoint is its own neighborhood)"""
    neighborhoods = {}
    for i, t in enumerate(times):
        neighborhoods[i] = [i]  # Each timepoint is independent
    return neighborhoods

def run_lda_decoding(X, y, cv_folds=10):
    """Run LDA decoding with cross-validation"""
    y = np.array(y)
    # Determine feasible number of splits for stratified CV.
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2 or np.min(counts) < 2:
        raise ValueError('Not enough samples per class for stratified CV')

    n_splits = min(cv_folds, len(y), np.min(counts))
    if n_splits < 2:
        raise ValueError('Not enough samples to perform cross-validation')

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('lda', LinearDiscriminantAnalysis())
    ])

    accuracies = []
    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        pipe.fit(X_train, y_train)
        acc = pipe.score(X_test, y_test)
        accuracies.append(acc)

    return np.mean(accuracies), np.array(accuracies)

# Loop over pairs
for p in range(num_pairs):
    pair = pair_ids[p]
    print(f'Loading pair {p + 1} of {num_pairs}: sub-{pair:02d}')

    # Load behavioral events
    events_filename = os.path.join(path_to_data, f'sub-{pair:02d}', 'eeg', f'sub-{pair:02d}_task-RPS_events.tsv')
    events = pd.read_csv(events_filename, sep='\t')

    # We need: Player 1 played, Player 2 played, Outcome
    behav_cols = ['player1_resp', 'player2_resp', 'outcome']
    behav_data = events[behav_cols].values

    # Format behavioral data for both players
    # Column 1 - This player played: 1) Rock 2) Paper 3) Scissors
    # Column 2 - Other player played
    # Column 3 - Outcome
    # Column 4-5 - Previous trial responses
    
    Player_1_Behav = np.hstack([
        behav_data,
        np.vstack([np.full((1, 2), np.nan), behav_data[:-1, :2]])
    ])
    
    Player_2_Behav = np.hstack([
        behav_data[:, [1, 0]],  # Swap players
        np.zeros((len(behav_data), 1)),  # Outcome placeholder
        np.vstack([np.full((1, 2), np.nan), behav_data[:-1, [1, 0]]])
    ])
    
    # Fix outcome coding for player 2
    Player_2_Behav[behav_data[:, 2] == 1, 2] = 1
    Player_2_Behav[behav_data[:, 2] == 2, 2] = 3
    Player_2_Behav[behav_data[:, 2] == 3, 2] = 2

    all_behav_data = np.dstack([Player_1_Behav, Player_2_Behav])

    np.random.seed(p)

    # Loop over the 2 players in the pair
    for ppt in range(1, 3):
        print(f'   Player {ppt}')
        
        # Load pre-processed EEG data
        epo_path = os.path.join(derivatives_path, f'pair-{pair:02d}_player-{ppt}_task-RPS_eeg-epo.fif')
        
        if not os.path.exists(epo_path):
            print(f'   File not found: {epo_path}. Skipping.')
            continue
            
        epochs = mne.read_epochs(epo_path, preload=True)
        eeg_data = epochs.get_data()  # Shape: (n_trials, n_chan, n_time)

        # Get behavioral data for this player
        behav = all_behav_data[:, :, ppt - 1]

        # Split into 3 parts for baseline correction
        # Times: -0.2 to 2, 1.8 to 4, 3.8 to 5
        times = epochs.times
        
        # Part A: -0.2 to 2 (Get Ready phase)
        idx_A = (times >= -0.2) & (times <= 2)
        eeg_A = eeg_data[:, :, idx_A]
        
        # Part B: 1.8 to 4 (Response phase) - will be shifted
        idx_B = (times >= 1.8) & (times <= 4)
        eeg_B = eeg_data[:, :, idx_B]
        
        # Part C: 3.8 to 5 (Feedback phase)
        idx_C = (times >= 3.8) & (times <= 5)
        eeg_C = eeg_data[:, :, idx_C]

        # Align behavior and EEG trial counts (in some files trials are dropped during preprocessing)
        n_trials = eeg_data.shape[0]
        if behav.shape[0] != n_trials:
            print(f'   Warning: behav has {behav.shape[0]} rows but EEG has {n_trials} trials. Aligning to least common length.')
            if behav.shape[0] > n_trials:
                behav = behav[:n_trials]
            else:
                print('   Not enough behavioral trials for EEG epochs. Skipping this player.')
                continue

        # Remove first trial of each block (40 trials per block)
        rem_idx = np.arange(0, n_trials, 40)
        keep_idx = np.setdiff1d(np.arange(n_trials), rem_idx)

        if len(keep_idx) == 0:
            print('   No trials left after block trimming. Skipping this player.')
            continue

        eeg_A = eeg_A[keep_idx]
        eeg_B = eeg_B[keep_idx]
        eeg_C = eeg_C[keep_idx]
        behav = behav[keep_idx]

        # Baseline correction (using -0.2 to 0)
        baseline_idx = (times >= -0.2) & (times <= 0)
        baseline_eeg_A = eeg_data[keep_idx][:, :, baseline_idx].mean(axis=2, keepdims=True)

        eeg_A = eeg_A - baseline_eeg_A
        eeg_B = eeg_B - baseline_eeg_A[:, :, :1]  # Use same baseline
        eeg_C = eeg_C - baseline_eeg_A[:, :, :1]

        # Average into time bins and concatenate
        # Time windows: 0-0.25, 0.25-0.5, ..., 1.75-2 (8 bins)
        n_bins_AB = 8
        n_bins_C = 4
        n_bins_total = n_bins_AB * 2 + n_bins_C  # 20 time bins

        time_averaged = np.zeros((len(keep_idx), num_chan, n_bins_total))
        
        # Average Part A into 8 bins
        for b in range(n_bins_AB):
            bin_size = eeg_A.shape[2] // n_bins_AB
            start_idx = b * bin_size
            end_idx = start_idx + bin_size
            time_averaged[:, :, b] = eeg_A[:, :, start_idx:end_idx].mean(axis=2)
        
        # Average Part B into 8 bins
        for b in range(n_bins_AB):
            bin_size = eeg_B.shape[2] // n_bins_AB
            start_idx = b * bin_size
            end_idx = start_idx + bin_size
            time_averaged[:, :, n_bins_AB + b] = eeg_B[:, :, start_idx:end_idx].mean(axis=2)
        
        # Average Part C into 4 bins
        for b in range(n_bins_C):
            bin_size = eeg_C.shape[2] // n_bins_C
            start_idx = b * bin_size
            end_idx = start_idx + bin_size
            time_averaged[:, :, n_bins_AB * 2 + b] = eeg_C[:, :, start_idx:end_idx].mean(axis=2)

        # Test what we decode
        test_idx = [0, 1, 2, 3]  # 0=self, 1=other, 2=self_prev, 3=other_prev
        
        for test in range(len(test_idx)):
            # Set targets based on what we decode
            if test_idx[test] == 0:
                targets = behav[:, 0]  # Self current
            elif test_idx[test] == 1:
                targets = behav[:, 1]  # Other current
            elif test_idx[test] == 2:
                targets = behav[:, 3]  # Self previous
            else:
                targets = behav[:, 4]  # Other previous

            # Remove no-responses
            valid_idx = targets > 0
            X = time_averaged[valid_idx]
            y = targets[valid_idx]

            if len(np.unique(y)) < 3:
                print(f'   Test {test}: Not enough classes. Skipping.')
                continue

            # Decode for each time bin
            accuracies_per_timebin = []
            for t_bin in range(n_bins_total):
                X_t = X[:, :, t_bin]  # (n_valid_trials, n_chan)

                try:
                    mean_acc, _ = run_lda_decoding(X_t, y, cv_folds=10)
                except ValueError as exc:
                    print(f'   Test {test}, time bin {t_bin}: {exc}. Skipping further bins for this test.')
                    accuracies_per_timebin = []
                    break

                accuracies_per_timebin.append(mean_acc)

            # Searchlight: decode for each channel using neighbors
            accuracies_searchlight = np.zeros((num_chan, n_bins_total))
            
            for t_bin in range(n_bins_total):
                X_t = X[:, :, t_bin]
                
                # For each channel, use itself + neighbors (simple: all channels in this bin)
                # This is a simplified searchlight (just uses all channels)
                mean_acc, _ = run_lda_decoding(X_t, y, cv_folds=10)
                accuracies_searchlight[:, t_bin] = mean_acc

            # Store results
            decoding_accuracy[test] = {
                'accuracy': np.array(accuracies_per_timebin),
                'pair': pair,
                'player': ppt
            }
            
            searchlight_acc[test] = {
                'searchlight': accuracies_searchlight,
                'pair': pair,
                'player': ppt
            }

            print(f'   Test {test}: Mean accuracy = {np.mean(accuracies_per_timebin):.3f}')

        # Save results
        save_dict = {
            'decoding_accuracy': decoding_accuracy,
            'searchlight_acc': searchlight_acc
        }
        
        save_path = os.path.join(derivatives_path, f'pair-{pair:02d}_player-{ppt}_task-RPS_decoding.mat')
        
        # Convert to MATLAB-compatible format
        mat_dict = {}
        for test_key in decoding_accuracy:
            mat_dict[f'decoding_acc_test{test_key}'] = decoding_accuracy[test_key]['accuracy']
            mat_dict[f'searchlight_test{test_key}'] = searchlight_acc[test_key]['searchlight']
        
        savemat(save_path, mat_dict)

print('Decoding analysis completed successfully!')