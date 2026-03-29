"""
Pre-processing script (Python version of step1_preprocessing.m):
   - Plot the data so we can identify noisy channels
   - Interpolate noisy channels
   - Down-sample to 256 Hz (to make the data easier to work with)
   - Save

Uses MNE-Python instead of FieldTrip.

Notes:
   - The 2 pairs are in a single file in the raw data. We have to select
   different channels based on the pair number.
"""

import os
import pandas as pd
import numpy as np
import mne
from scipy.io import loadmat

# Set the path
path_to_data = 'MNE-sample-data/ds006761'
path_to_code = 'author_code/parsed_python'  # Adjust if needed

# Set parameters
# If identify_bad_channels is true, we plot the data so we can identify bad
# channels. Skip if we've already identified bad channels and want to fix them.
# To fix bad channels, set interpolate_bad_channels to true. This assumes we
# have a tsv file called 'participants.tsv' with labels
# of the bad channels.
identify_bad_channels = False
interpolate_bad_channels = True

# Set parameters
num_trials = 480  # There were 480 games (trials) in the experiment
pair_ids = np.concatenate([np.arange(1, 10), np.arange(11, 23), np.arange(25, 35)])  # Pair IDs (Pair 10, 23, 24 excluded)
num_pairs = len(pair_ids)  # Number of pairs
FS = 2048  # Biosemi sampling frequency

# Load the demographics - this has information about which channels are bad
participants = pd.read_csv(os.path.join(path_to_data, 'participants.tsv'), sep='\t')

# Load biosemi64 montage
mat_data = loadmat(os.path.join(path_to_data, 'biosemi64.mat'))
biosemi64 = mat_data['biosemi64']

# Prepare montage
# Assuming the labels are standard BioSemi 64
ch_names_64 = [f'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 'Fz', 'Cz', 'Pz', 'Oz', 'FC1', 'FC2', 'CP1', 'CP2', 'FC5', 'FC6', 'CP5', 'CP6', 'TP9', 'TP10', 'POz', 'ECG', 'EOG', 'EMG', 'GSR', 'Respiration', 'Temp', 'Status', 'EXG1', 'EXG2', 'EXG3', 'EXG4', 'EXG5', 'EXG6', 'EXG7', 'EXG8', 'GSR1', 'GSR2', 'Erg1', 'Erg2', 'Resp', 'Plet', 'Temp', 'EXG1', 'EXG2', 'EXG3', 'EXG4', 'EXG5', 'EXG6', 'EXG7', 'EXG8', 'GSR1', 'GSR2', 'Erg1', 'Erg2']  # This is approximate, need to match
# Actually, for BioSemi 64, the labels are specific.
# From the code, it uses layout.label(1:64) from biosemi64.lay, but since we have biosemi64.mat, use that.

# The code uses ft_prepare_layout(struct('layout','biosemi64.lay')); then data_epoch.label(1:64) = layout.label(1:64);
# So, standard BioSemi 64 labels.

# MNE has 'biosemi64' montage
montage = mne.channels.make_standard_montage('biosemi64')

# But to match the 3D positions, use the mat file
# biosemi64 is 64x3 positions
ch_names_biosemi = ['Fp1', 'AF7', 'AF3', 'F1', 'F3', 'F5', 'F7', 'FT7', 'FC5', 'FC3', 'FC1', 'C1', 'C3', 'C5', 'T7', 'TP7', 'CP5', 'CP3', 'CP1', 'P1', 'P3', 'P5', 'P7', 'P9', 'PO7', 'PO3', 'O1', 'Iz', 'Oz', 'POz', 'Pz', 'CPz', 'Fpz', 'Fp2', 'AF8', 'AF4', 'AFz', 'Fz', 'F2', 'F4', 'F6', 'F8', 'FT8', 'FC6', 'FC4', 'FC2', 'FCz', 'C2', 'C4', 'C6', 'T8', 'TP8', 'CP6', 'CP4', 'CP2', 'Cz', 'P2', 'P4', 'P6', 'P8', 'P10', 'PO8', 'PO4', 'O2']  # Standard BioSemi 64
montage = mne.channels.make_dig_montage(ch_pos=dict(zip(ch_names_biosemi, biosemi64)), coord_frame='head')

# Run pre-processing
# Load the data. If we want to find bad channels, plot the data for visual
# inspection. Interpolate bad channels (found in demographics file), down-sample
# to 256 Hz and save.

# Loop over pairs
for p in range(num_pairs):
    pair = pair_ids[p]
    print(f'Loading pair {p+1} of {num_pairs}: {pair}')

    # Get the trigger times (for the start of each trial)
    events_filename = os.path.join(path_to_data, f'sub-{pair:02d}', 'eeg', f'sub-{pair:02d}_task-RPS_events.tsv')
    events = pd.read_csv(events_filename, sep='\t')
    stimonsample = events['onset_sample'].values

    # Specify the epoch: -0.2 to 5 sec (relative to onset of 'Decision' screen)
    prestim = 0.2  # epoch start
    poststim = 5  # epoch end

    # Make trial matrix
    trl = np.column_stack([stimonsample - int(prestim * FS), stimonsample + int(poststim * FS), stimonsample - stimonsample])  # TRL equivalent

    # Read the raw data
    raw_filename = os.path.join(path_to_data, f'sub-{pair:02d}', 'eeg', f'sub-{pair:02d}_task-RPS_eeg.bdf')
    raw = mne.io.read_raw_bdf(raw_filename, preload=True)

    # Loop over player 1 and 2 in the pair
    for ppt in [1, 2]:
        # Note: In the original, player 1 and 2 are swapped in EEG vs behavioral
        # Player 1: 2-A1 to 2-A32 & 2-B1 to 2-B32
        # Player 2: 1-A1 to 1-A32 & 1-B1 to 1-B32
        if ppt == 1:
            chan_pattern = ['2-A', '2-B']
        else:
            chan_pattern = ['1-A', '1-B']

        chan_idx = [any(pat in ch for pat in chan_pattern) for ch in raw.ch_names]
        orig_label = [ch for ch, idx in zip(raw.ch_names, chan_idx) if idx]

        # Select channels
        raw_ppt = raw.copy().pick_channels(orig_label)

        # Rename the first 64 EEG channels to standard BioSemi labels
        if len(raw_ppt.ch_names) >= 64:
            rename_dict = {raw_ppt.ch_names[i]: ch_names_biosemi[i] for i in range(64)}
            raw_ppt.rename_channels(rename_dict)
            # Keep only the 64 EEG channels
            raw_ppt.pick_channels(ch_names_biosemi)

        # Set the montage
        raw_ppt.set_montage(montage)

        raw_ppt = raw_ppt.resample(256, npad='auto')  # Down-sample to 256 Hz

        # Epoch the data
        events_array = np.column_stack([trl[:, 0], np.zeros(len(trl), dtype=int), np.ones(len(trl), dtype=int)])  # onset, prev, event_id
        epochs = mne.Epochs(raw_ppt, events_array, event_id=1, tmin=-prestim, tmax=poststim, baseline=None, preload=True)

        # Do we want to plot the data to identify bad channels?
        if identify_bad_channels:
            # Highpass and lowpass filter for plotting
            epochs_f = epochs.copy().filter(l_freq=0.1, h_freq=100, method='iir', iir_params=dict(order=4, ftype='butter'))
            epochs_f.plot(n_epochs=10, n_channels=64, scalings='auto', title=f'Pair {pair}, Player {ppt}')

        # Do we want to interpolate the bad channels?
        if interpolate_bad_channels:
            # Get the channels to fix for this ppt
            row = participants[participants['participant_id'] == f'sub-{pair:02d}']
            if ppt == 1:
                bad_chans_str = row['player1_pre_processing_channels_fixed'].values[0]
            else:
                bad_chans_str = row['player2_pre_processing_channels_fixed'].values[0]

            if pd.notna(bad_chans_str) and bad_chans_str != '':
                bad_chans = [ch.strip() for ch in bad_chans_str.split(',')]
                # Interpolate bad channels
                epochs.info['bads'] = bad_chans
                epochs.interpolate_bads()
                print(f'Pair {pair}, Player {ppt}: fixed {bad_chans_str}')

            # Down-sample the data to 256 Hz
            epochs_resampled = epochs.copy().resample(256)

            # Save the epoched data
            save_path = os.path.join(path_to_data, 'derivatives', f'pair-{pair:02d}_player-{ppt}_task-RPS_eeg-epo.fif')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            epochs_resampled.save(save_path, overwrite=True)

print('Preprocessing completed.')