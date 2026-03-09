"""
EEG Decoding Pipeline: Time-resolved decoding of self choice during RPS task.

This script implements two approaches for training time-resolved LDA decoders:
1. Option 1: Train individual models per subject, then average
2. Option 2: Train a single combined model across all subjects

The script supports RAM-efficient chunked processing and provides a Rich terminal UI
with live progress bars and log panels.

Data saving/loading feature:
- After processing subjects, data is automatically saved to pickle files
- Use --data-file flag to load pre-processed data instead of re-processing subjects
- Use --option to select which analysis to run (1, 2, or compare)
- Results are automatically plotted with decoding accuracy over time
- Use --no-plot to disable plotting for headless environments

Author: Hugo Hirling
Date: 2026-02-18

The logging and comments in this script were created with the assistance of AI (VS Code Copilot).
"""

# ============================================================================
# IMPORTS
# ============================================================================

import os
import gc
from pathlib import Path
from typing import List, Tuple
import argparse
import pickle

import numpy as np
import pandas as pd
import mne
import time
from mne_bids import BIDSPath
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold
# plotting moved to plotting_utils


# Local imports
import config
from logging_utils import (
    terminal_log,
    redirect_streams,
    init_logging,
    get_progress_bar,
    get_live_display,
    LogRenderable,
    RICH_AVAILABLE,
)
from plotting_utils import (
    plot_decoding_accuracy,
    save_plot_data,
    load_plot_data,
    plot_only,
    plot_comparison,
)


# ============================================================================
# CONFIGURATION & CONSTANTS
# ============================================================================

ROOT = Path(config.BIDS_ROOT)

# List of all subject IDs in the dataset
SUBJECTS = [
    "sub-01", "sub-02", "sub-03", "sub-04", "sub-05", "sub-06", "sub-07", 
    "sub-08", "sub-09", "sub-11", "sub-12", "sub-13", "sub-14", "sub-15", 
    "sub-16", "sub-17", "sub-18", "sub-19", "sub-20", "sub-21", "sub-22", 
    "sub-25", "sub-26", "sub-27", "sub-28", "sub-29", "sub-30", "sub-31", 
    "sub-32", "sub-33", "sub-34"
]

# Canonical channel set derived from config (ensures consistency across subjects)
COMMON_CHANNELS = list(config.channel_labels.values())

# Biosemi 64-electrode montage file location
BIOSEMI_FILE = ROOT / "biosemi64.mat"

# RAM threshold (in GB) for deciding between chunked vs in-RAM processing
RAM_THRESHOLD_GB = 4


# ============================================================================
# DATA LOADING & PREPROCESSING
# ============================================================================

def load_refactor_split_data(subject: str, only_p1: bool = True):
    """
    Load raw EEG data from BDF file, preprocess, and split by player.
    
    Processing steps:
    1. Load raw BDF file
    2. Resample to target frequency
    3. Apply bandpass filter
    4. Set average EEG reference
    5. Split by player (channel prefix 1- or 2-)
    6. Rename channels to canonical labels
    7. Apply biosemi64 montage
    8. Select common channels for consistency
    
    Args:
        subject (str): Subject ID (e.g., 'sub-01')
        only_p1 (bool): If True, return only player 1 data; otherwise return (p1, p2) tuple
    
    Returns:
        mne.io.Raw or tuple of (mne.io.Raw, mne.io.Raw): Preprocessed raw data
    """
    
    terminal_log(f"Processing {subject}...")
    
    # Load raw BDF file
    bdf_path = ROOT / subject / "eeg" / f"{subject}_task-RPS_eeg.bdf"
    with redirect_streams():
        raw = mne.io.read_raw_bdf(bdf_path, preload=True)
    terminal_log(f"  Loaded raw file: {bdf_path.name}")
    
    # Resample to target frequency
    with redirect_streams():
        raw.resample(config.DOWNSAMPLE_SFREQ)
    terminal_log(f"  Resampled to {config.DOWNSAMPLE_SFREQ} Hz")
    
    # Apply bandpass filter
    with redirect_streams():
        raw.filter(l_freq=config.FREQ_LOWER, h_freq=config.FREQ_UPPER)
    terminal_log(f"  Filtered {config.FREQ_LOWER}-{config.FREQ_UPPER} Hz")
    
    # Set average EEG reference
    with redirect_streams():
        raw.set_eeg_reference("average")
    terminal_log("  Set average EEG reference")
    
    # Split channels by player (Player1 → 2-*, Player2 → 1-*)
    raw_p1 = raw.copy().pick_channels([ch for ch in raw.ch_names if ch.startswith("2-")])
    raw_p2 = raw.copy().pick_channels([ch for ch in raw.ch_names if ch.startswith("1-")]) if not only_p1 else None
    terminal_log(f"  Split channels: player1 {len(raw_p1.ch_names)} chs, player2 {len(raw_p2.ch_names) if raw_p2 else 0} chs")
    
    # Remove prefix and rename to canonical labels
    def remove_prefix_and_rename(raw_obj, prefix):
        mapping = {ch: ch.replace(prefix, "") for ch in raw_obj.ch_names}
        raw_obj.rename_channels(mapping)
    
    remove_prefix_and_rename(raw_p1, "2-")
    if raw_p2 is not None:
        remove_prefix_and_rename(raw_p2, "1-")

    raw_p1.pick_channels([ch for ch in raw_p1.ch_names if ch in config.channel_labels.keys()])
    
    raw_p1.rename_channels(config.channel_labels)
    if raw_p2 is not None:
        raw_p2.pick_channels([ch for ch in raw_p2.ch_names if ch in config.channel_labels.keys()])
        raw_p2.rename_channels(config.channel_labels)
    terminal_log("  Renamed channels to canonical labels")
    
    # Apply montage
    montage = mne.channels.make_standard_montage("biosemi64", head_size=0.105)
    with redirect_streams():
        raw_p1.set_montage(montage, on_missing="ignore")
        if raw_p2 is not None:
            raw_p2.set_montage(montage, on_missing="ignore")
    terminal_log("  Applied biosemi64 montage")
    
    # Select common channels for consistency and RAM efficiency
    available_channels_p1 = [ch for ch in COMMON_CHANNELS if ch in raw_p1.ch_names]
    raw_p1.pick_channels(available_channels_p1)
    if raw_p2 is not None:
        available_channels_p2 = [ch for ch in COMMON_CHANNELS if ch in raw_p2.ch_names]
        raw_p2.pick_channels(available_channels_p2)
    terminal_log(f"  Selected {len(raw_p1.ch_names)}/{len(COMMON_CHANNELS)} canonical channels")
    
    return raw_p1 if only_p1 else (raw_p1, raw_p2)


def create_events(subject: str) -> Tuple[np.ndarray, dict, pd.DataFrame]:
    """
    Load event markers and behavioral data from TSV file.
    
    Args:
        subject (str): Subject ID
    
    Returns:
        tuple: (events array [N, 3], event_id dict, behavioral dataframe)
    """
    
    # Load events from BIDS events file
    events_path = ROOT / subject / "eeg" / f"{subject}_task-RPS_events.tsv"
    events_df = pd.read_csv(events_path, sep="\t")
    terminal_log(f"  Loaded events TSV: {len(events_df)} events")
    
    # Create behavioral dataframe with lag features
    behav = events_df[["player1_resp", "player2_resp", "outcome"]].copy()
    behav["self_prev"] = behav["player1_resp"].shift(1)
    behav["other_prev"] = behav["player2_resp"].shift(1)
    
    # Convert event onsets to sample indices and create events array
    sfreq = config.DOWNSAMPLE_SFREQ
    events = np.array([
        [int(row["onset"] * sfreq), 0, row["player1_resp"]]
        for _, row in events_df.iterrows()
    ], dtype=int)
    
    # Define event ID mapping
    event_id = {"rock": 1, "paper": 2, "scissors": 3}
    
    terminal_log(f"  Created events array ({len(events)} events) and behavior dataframe")
    return events, event_id, behav


def create_epochs(
    raw: mne.io.Raw,
    events: np.ndarray,
    event_id: dict,
    behav: pd.DataFrame
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], 
           Tuple[np.ndarray, np.ndarray], np.ndarray]:
    """
    Create epochs and split into decision, response, and feedback windows.
    
    Processing:
    1. Create epochs from raw data and events
    2. Drop every 40th epoch (to match MATLAB preprocessing)
    3. Crop into task-specific windows (decision, response, feedback)
    4. Extract data arrays and filter by valid labels
    
    Args:
        raw (mne.io.Raw): Preprocessed raw data
        events (np.ndarray): Event array [N, 3]
        event_id (dict): Event ID mapping
        behav (pd.DataFrame): Behavioral data
    
    Returns:
        tuple: ((X_decision, times), (X_response, times), (X_feedback, times), Y)
    """
    
    # Create epochs with baseline correction
    with redirect_streams():
        epochs = mne.Epochs(
            raw,
            events=events,
            event_id=event_id,
            tmin=config.EPOCH_TMIN,
            tmax=config.EPOCH_TMAX,
            baseline=(config.EPOCH_BASELINE_MIN, config.EPOCH_BASELINE_MAX),
            preload=True,
            metadata=behav
        )
    terminal_log(f"  Created {len(epochs)} epochs")
    
    # Drop every 40th epoch (to match MATLAB downsampling)
    with redirect_streams():
        remove_idx = np.arange(0, len(epochs), 40)
        epochs = epochs.drop(remove_idx)
    terminal_log(f"  Dropped {len(remove_idx)} epochs (every 40th) → {len(epochs)} remain")
    
    # Crop into specific task windows
    with redirect_streams():
        epochs_decision = epochs.copy().crop(tmin=config.DECISION_TMIN, tmax=config.DECISION_TMAX)
        epochs_response = epochs.copy().crop(tmin=config.RESPONSE_TMIN, tmax=config.RESPONSE_TMAX)
        epochs_feedback = epochs.copy().crop(tmin=config.FEEDBACK_TMIN, tmax=config.FEEDBACK_TMAX)
    terminal_log("  Cropped into decision/response/feedback windows")
    
    # Extract labels and filter invalid trials
    Y = epochs_decision.metadata["player1_resp"].to_numpy()
    mask = Y > 0
    epochs_decision = epochs_decision[mask]
    epochs_response = epochs_response[mask]
    epochs_feedback = epochs_feedback[mask]
    Y = Y[mask]
    terminal_log(f"  After filtering: {len(Y)} valid trials")
    
    # Extract data arrays
    X_decision = epochs_decision.get_data()
    X_response = epochs_response.get_data()
    X_feedback = epochs_feedback.get_data()
    times = epochs_decision.times
    
    terminal_log(f"  Data shapes: decision {X_decision.shape}, response {X_response.shape}, feedback {X_feedback.shape}")
    
    return (X_decision, times), (X_response, times), (X_feedback, times), Y


# ============================================================================
# DECODING MODELS
# ============================================================================

def sliding_estimator(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Train a time-resolved LDA decoder on single-subject data.
    
    Uses cross-validation to estimate decoding accuracy across time.
    
    Args:
        X (np.ndarray): Feature array [N_trials, N_channels, N_times]
        Y (np.ndarray): Labels [N_trials]
    
    Returns:
        np.ndarray: Mean cross-validated accuracy across time [N_times]
    """
    clf = LinearDiscriminantAnalysis()
    time_decod = mne.decoding.SlidingEstimator(clf, n_jobs=1, scoring="accuracy")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    with redirect_streams():
        scores = mne.decoding.cross_val_multiscore(time_decod, X, Y, cv=cv, n_jobs=1)
    mean_scores = scores.mean(axis=0)
    
    return mean_scores


def sliding_estimator_chunked(
    X_list: List[np.ndarray],
    Y_list: List[np.ndarray],
    chunk_size: int = 1000
) -> np.ndarray:
    """
    Train a time-resolved LDA decoder on combined multi-subject data with RAM optimization.
    
    Processes time windows in chunks to reduce memory pressure for large datasets.
    
    Args:
        X_list (List[np.ndarray]): List of feature arrays per subject [N_trials, N_channels, N_times]
        Y_list (List[np.ndarray]): List of label arrays per subject
        chunk_size (int): Number of timepoints to process in each chunk
    
    Returns:
        np.ndarray: Mean cross-validated accuracy across time
    """
    
    # Combine all subject data
    X_combined = np.concatenate(X_list, axis=0)
    Y_combined = np.concatenate(Y_list)
    
    n_channels = X_combined.shape[1]
    n_times = X_combined.shape[2]
    n_epochs = X_combined.shape[0]
    
    estimated_ram_mb = (n_epochs * n_channels * chunk_size * 8) / (1024 ** 2)
    terminal_log(f"  Chunked processing: {n_epochs} epochs, {n_channels} channels, {n_times} timepoints")
    terminal_log(f"  Estimated RAM per chunk: {estimated_ram_mb:.1f} MB")
    
    scores_all = []
    
    # Process in time chunks
    for start_time in range(0, n_times, chunk_size):
        end_time = min(start_time + chunk_size, n_times)
        time_slice = slice(start_time, end_time)
        
        X_chunk = X_combined[:, :, time_slice]
        
        clf = LinearDiscriminantAnalysis()
        time_decod = mne.decoding.SlidingEstimator(clf, n_jobs=1, scoring="accuracy")
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        terminal_log(f"  Processing timepoints {start_time:4d}-{end_time:4d}...")
        with redirect_streams():
            scores_chunk = mne.decoding.cross_val_multiscore(time_decod, X_chunk, Y_combined, cv=cv, n_jobs=1)
        scores_all.append(scores_chunk.mean(axis=0))
        
        # Memory cleanup
        del X_chunk, scores_chunk
        gc.collect()
    
    return np.concatenate(scores_all)


def sliding_estimator_combined(
    X_list: List[np.ndarray],
    Y_list: List[np.ndarray]
) -> np.ndarray:
    """
    Train a time-resolved LDA decoder on combined multi-subject data (standard, non-chunked).
    
    Requires sufficient RAM (~6 GB+ for large datasets).
    
    Args:
        X_list (List[np.ndarray]): List of feature arrays per subject
        Y_list (List[np.ndarray]): List of label arrays per subject
    
    Returns:
        np.ndarray: Mean cross-validated accuracy across time
    """
    
    # Combine all subject data
    X_combined = np.concatenate(X_list, axis=0)
    Y_combined = np.concatenate(Y_list)
    
    n_epochs = X_combined.shape[0]
    n_channels = X_combined.shape[1]
    n_times = X_combined.shape[2]
    estimated_ram_gb = (n_epochs * n_channels * n_times * 8) / (1024 ** 3)
    
    terminal_log(f"  Combined data: {n_epochs} epochs, {n_channels} channels, {n_times} timepoints")
    terminal_log(f"  Estimated RAM: {estimated_ram_gb:.2f} GB")
    
    clf = LinearDiscriminantAnalysis()
    time_decod = mne.decoding.SlidingEstimator(clf, n_jobs=1, scoring="accuracy")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    terminal_log("  Training combined model with cross-validation...")
    with redirect_streams():
        scores = mne.decoding.cross_val_multiscore(time_decod, X_combined, Y_combined, cv=cv, n_jobs=1)
    mean_scores = scores.mean(axis=0)
    
    return mean_scores


def save_processed_data(data: dict, filepath: str):
    """
    Save processed EEG data to a pickle file.
    
    Args:
        data (dict): Dictionary containing the processed data
        filepath (str): Path to save the data
    """
    terminal_log(f"Saving processed data to {filepath}...")
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)
    terminal_log("Data saved successfully.")


def load_processed_data(filepath: str) -> dict:
    """
    Load processed EEG data from a pickle file.
    
    Args:
        filepath (str): Path to the saved data file
    
    Returns:
        dict: Dictionary containing the processed data
    """
    terminal_log(f"Loading processed data from {filepath}...")
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    terminal_log("Data loaded successfully.")
    return data


# ============================================================================
# MAIN PIPELINE ORCHESTRATION
# ============================================================================

def main_option1_individual(data_file: str = None, show_plot: bool = True):
    """
    Option 1: Train individual LDA models per subject, then average results.
    
    This approach trains a separate decoder for each subject and then averages
    the decoding accuracies across subjects.
    
    Args:
        data_file (str): Path to saved data file. If provided, load data from file instead of processing subjects.
        show_plot (bool): Whether to display the decoding accuracy plot.
    """
    
    terminal_log("=" * 60)
    terminal_log("OPTION 1: Individual Models (Per-Subject, Then Average)")
    terminal_log(f"Subjects: {len(SUBJECTS)}, Channels: {len(COMMON_CHANNELS)}")
    terminal_log("=" * 60)
    
    all_scores = []
    times = None
    
    if data_file and os.path.exists(data_file):
        # Load from saved data
        saved_data = load_processed_data(data_file)
        subject_data = saved_data['option1_data']
        times = saved_data['times']
        
        if RICH_AVAILABLE:
            with get_live_display() as live:
                with get_progress_bar() as progress:
                    task = progress.add_task("Process saved data", total=len(subject_data), eta="")
                    
                    for i, (X_decision, Y) in enumerate(subject_data):
                        mean_scores = sliding_estimator(X_decision, Y)
                        all_scores.append(mean_scores)
                        progress.update(task, advance=1)
                        live.refresh()
        else:
            for X_decision, Y in subject_data:
                mean_scores = sliding_estimator(X_decision, Y)
                all_scores.append(mean_scores)
    else:
        # Process subjects normally
        subject_data = []
        
        if RICH_AVAILABLE:
            with get_live_display() as live:
                with get_progress_bar() as progress:
                    task = progress.add_task("Train individual models", total=len(SUBJECTS), eta="")
                    subject_times: List[float] = []
                    
                    for i, subject in enumerate(SUBJECTS):
                        t0 = time.time()
                        raw_p1 = load_refactor_split_data(subject, only_p1=True)
                        events, event_id, behav = create_events(subject)
                        (X_decision, times), _, _, Y = create_epochs(raw_p1, events, event_id, behav)
                        
                        # Store data for potential saving
                        subject_data.append((X_decision, Y))
                        
                        mean_scores = sliding_estimator(X_decision, Y)
                        all_scores.append(mean_scores)

                        # record timing and update ETA field based on mean per-subject time
                        elapsed = time.time() - t0
                        subject_times.append(elapsed)
                        mean_time = float(np.mean(subject_times))
                        remaining = len(SUBJECTS) - (i + 1)
                        eta_seconds = int(mean_time * remaining)
                        hh = eta_seconds // 3600
                        mm = (eta_seconds % 3600) // 60
                        ss = eta_seconds % 60
                        eta_str = f"{hh:02d}:{mm:02d}:{ss:02d}"
                        # Advance the progress and set an artificial start time
                        # based on observed mean per-subject time so Rich can
                        # estimate TimeRemainingColumn.
                        progress.update(task, advance=1)
                        try:
                            task_obj = progress.get_task(task)
                            # prefer attribute names that exist across rich versions
                            started_attr = None
                            if hasattr(task_obj, "start_time"):
                                started_attr = "start_time"
                            elif hasattr(task_obj, "started_at"):
                                started_attr = "started_at"
                            if started_attr is not None:
                                # pretend the task started earlier by mean_time*(completed)
                                setattr(task_obj, started_attr, time.time() - mean_time * (i + 1))
                        except Exception:
                            pass
                        live.refresh()
        else:
            for subject in SUBJECTS:
                raw_p1 = load_refactor_split_data(subject, only_p1=True)
                events, event_id, behav = create_events(subject)
                (X_decision, times), _, _, Y = create_epochs(raw_p1, events, event_id, behav)
                
                # Store data for potential saving
                subject_data.append((X_decision, Y))
                
                mean_scores = sliding_estimator(X_decision, Y)
                all_scores.append(mean_scores)
        
        # Save processed data if no data_file was provided (normal run)
        if data_file is None:
            save_data = {
                'option1_data': subject_data,
                'times': times,
                'subjects': SUBJECTS
            }
            save_processed_data(save_data, 'processed_data_option1.pkl')
    
    # Average across subjects
    mean_scores_avg = np.mean(all_scores, axis=0)
    
    terminal_log(f"\nCompleted Option 1: Mean accuracy = {mean_scores_avg.mean():.3f}")
    
    # Save plot data
    save_plot_data(times, mean_scores_avg, 'plot_data_option1.pkl')
    
    # Plot results
    plot_decoding_accuracy(times, mean_scores_avg, "Option 1: Averaged Individual Models", show_plot)
    
    return times, mean_scores_avg


def main_option2_combined(data_file: str = None, show_plot: bool = True):
    """
    Option 2: Train a single LDA model on data combined across all subjects.
    
    This approach concatenates data from all subjects and trains a single decoder.
    It uses chunked processing if the total data size exceeds the RAM threshold.
    
    Args:
        data_file (str): Path to saved data file. If provided, load data from file instead of processing subjects.
        show_plot (bool): Whether to display the decoding accuracy plot.
    """
    
    terminal_log("=" * 60)
    terminal_log("OPTION 2: Combined Model (All Subjects in One Model)")
    terminal_log(f"Subjects: {len(SUBJECTS)}, Channels: {len(COMMON_CHANNELS)}")
    terminal_log("=" * 60)
    
    X_list = []
    Y_list = []
    times = None
    
    if data_file and os.path.exists(data_file):
        # Load from saved data
        saved_data = load_processed_data(data_file)
        X_list = saved_data['X_list']
        Y_list = saved_data['Y_list']
        times = saved_data['times']
    else:
        # Process subjects normally
        if RICH_AVAILABLE:
            with get_live_display() as live:
                with get_progress_bar() as progress:
                    
                    # Load subjects
                    task_load = progress.add_task("Load subjects", total=len(SUBJECTS), eta="")
                    load_times: List[float] = []
                    for i, subject in enumerate(SUBJECTS):
                        t0 = time.time()
                        raw_p1 = load_refactor_split_data(subject, only_p1=True)
                        events, event_id, behav = create_events(subject)
                        (X_decision, times), _, _, Y = create_epochs(raw_p1, events, event_id, behav)
                        X_list.append(X_decision)
                        Y_list.append(Y)

                        elapsed = time.time() - t0
                        load_times.append(elapsed)
                        mean_time = float(np.mean(load_times))
                        remaining = len(SUBJECTS) - (i + 1)
                        eta_seconds = int(mean_time * remaining)
                        hh = eta_seconds // 3600
                        mm = (eta_seconds % 3600) // 60
                        ss = eta_seconds % 60
                        eta_str = f"{hh:02d}:{mm:02d}:{ss:02d}"
                        progress.update(task_load, advance=1)
                        try:
                            task_obj = progress.get_task(task_load)
                            started_attr = None
                            if hasattr(task_obj, "start_time"):
                                started_attr = "start_time"
                            elif hasattr(task_obj, "started_at"):
                                started_attr = "started_at"
                            if started_attr is not None:
                                setattr(task_obj, started_attr, time.time() - mean_time * (i + 1))
                        except Exception:
                            pass
                        live.refresh()
        else:
            # Fallback without Rich
            for subject in SUBJECTS:
                raw_p1 = load_refactor_split_data(subject, only_p1=True)
                events, event_id, behav = create_events(subject)
                (X_decision, times), _, _, Y = create_epochs(raw_p1, events, event_id, behav)
                X_list.append(X_decision)
                Y_list.append(Y)
        
        # Save processed data if no data_file was provided (normal run)
        if data_file is None:
            save_data = {
                'X_list': X_list,
                'Y_list': Y_list,
                'times': times,
                'subjects': SUBJECTS
            }
            save_processed_data(save_data, 'processed_data_option2.pkl')
    
    # Estimate memory and decide on chunking strategy
    total_size_gb = sum(X.nbytes for X in X_list) / (1024 ** 3)
    terminal_log(f"\nEstimated total data size: {total_size_gb:.2f} GB")
    
    # Train combined model
    if RICH_AVAILABLE:
        with get_live_display() as live:
            with get_progress_bar() as progress:
                task_train = progress.add_task("Train combined model", total=1)
                if total_size_gb > RAM_THRESHOLD_GB:
                    terminal_log(f"Data exceeds {RAM_THRESHOLD_GB} GB threshold → using chunked processing")
                    mean_scores_combined = sliding_estimator_chunked(X_list, Y_list, chunk_size=500)
                else:
                    terminal_log(f"Data within {RAM_THRESHOLD_GB} GB threshold → using standard processing")
                    mean_scores_combined = sliding_estimator_combined(X_list, Y_list)
                progress.advance(task_train)
                live.refresh()
    else:
        if total_size_gb > RAM_THRESHOLD_GB:
            terminal_log(f"Data exceeds {RAM_THRESHOLD_GB} GB → using chunked processing")
            mean_scores_combined = sliding_estimator_chunked(X_list, Y_list, chunk_size=500)
        else:
            terminal_log(f"Data within {RAM_THRESHOLD_GB} GB → using standard processing")
            mean_scores_combined = sliding_estimator_combined(X_list, Y_list)
    
    terminal_log(f"\nCompleted Option 2: Mean accuracy = {mean_scores_combined.mean():.3f}")
    
    # Save plot data
    save_plot_data(times, mean_scores_combined, 'plot_data_option2.pkl')
    
    # Plot results
    plot_decoding_accuracy(times, mean_scores_combined, "Option 2: Combined Model", show_plot)
    
    return times, mean_scores_combined


def compare_both_options(data_file_option1: str = None, data_file_option2: str = None, show_plot: bool = True):
    """
    Compare Option 1 (individual models, then averaged) with Option 2 (combined model).
    
    Trains both approaches and visualizes the comparison.
    
    Args:
        data_file_option1 (str): Path to saved data file for option 1
        data_file_option2 (str): Path to saved data file for option 2
        show_plot (bool): Whether to display the plots.
    """
    
    terminal_log("=" * 60)
    terminal_log("COMPARISON: Option 1 vs Option 2")
    terminal_log(f"Subjects: {len(SUBJECTS)}, Channels: {len(COMMON_CHANNELS)}")
    terminal_log("=" * 60)
    
    # Option 1
    terminal_log("\n[Option 1] Training individual models...")
    times_opt1, scores_opt1 = main_option1_individual(data_file_option1, show_plot)
    
    # Option 2
    terminal_log("\n[Option 2] Training combined model...")
    times_opt2, scores_opt2 = main_option2_combined(data_file_option2, show_plot)
    
    # Use shared plotting helper for comparison (handles interpolation & stats)
    terminal_log("\nGenerating comparison plot...")
    plot_comparison(times_opt1, scores_opt1, times_opt2, scores_opt2, 
                    title="Comparison: Individual Models (averaged) vs Combined Model", 
                    show_plot=show_plot)



# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="EEG Decoding Pipeline")
    parser.add_argument("--data-file", type=str, help="Path to saved processed data file")
    parser.add_argument("--option", type=str, choices=["1", "2", "compare"], default="2",
                       help="Which option to run: 1 (individual), 2 (combined), compare (both)")
    parser.add_argument("--no-plot", action="store_true", help="Disable plotting of results")
    parser.add_argument("--plot-only", action="store_true", help="Skip calculations and only plot saved results")
    parser.add_argument("--plot-file", type=str, default="plot_data", 
                       help="File prefix for plot data when using --plot-only (default: plot_data)")
    
    args = parser.parse_args()
    
    # Initialize logging
    init_logging()
    
    terminal_log("EEG Decoding Pipeline: Starting...")
    terminal_log(f"Configuration: {len(SUBJECTS)} subjects, {len(COMMON_CHANNELS)} channels")
    
    # Handle plot-only mode
    if args.plot_only:
        plot_only(args.option, args.plot_file)
    else:
        show_plot = not args.no_plot
        
        # Run selected option
        if args.option == "1":
            main_option1_individual(args.data_file, show_plot)
        elif args.option == "2":
            main_option2_combined(args.data_file, show_plot)
        elif args.option == "compare":
            # For comparison, try to load saved data if available
            data_file_option1 = "processed_data_option1.pkl" if os.path.exists("processed_data_option1.pkl") else None
            data_file_option2 = "processed_data_option2.pkl" if os.path.exists("processed_data_option2.pkl") else None
            compare_both_options(data_file_option1, data_file_option2, show_plot)
    
    terminal_log("\nPipeline completed.")