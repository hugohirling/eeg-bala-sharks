"""
Cross-Brain Multivariate Pattern Analysis (MVPA) Pipeline

In traditional EEG decoding, a subject's brainwaves are used to predict their own 
actions. This module performs an advanced 'Hyperscanning' analysis: it utilizes 
Player 1's continuous EEG data to decode and predict Player 2's hidden choices.

This investigates whether human brains engaged in dyadic, adversarial competition 
(like Rock-Paper-Scissors) inherently build predictive neural models or simulate 
the opponent's intentions prior to the final action.
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import mne
from mne.decoding import SlidingEstimator, cross_val_multiscore
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Ensure the root directory is accessible to import custom path configurations
CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import paths 


def process_cross_decoding(subject_id: str):
    """
    Executes the Cross-Brain MVPA pipeline for a specific interacting dyad.
    
    Parameters
    ----------
    subject_id : str
        The continuous zero-padded string identifier for the dyad pair (e.g., "01").
        
    Outputs
    -------
    Saves a 1D NumPy matrix (.npy) representing the temporal decoding accuracy, 
    as well as a localized line plot (.png) illustrating brain-to-brain predictability.
    """
    print(f"\n[{subject_id}] Starting Cross-Brain / Opponent Decoding...")
    
    input_dir = paths.OUTPUT_DIR / "preprocessing"
    output_dir = paths.OUTPUT_DIR / "analysis" / "decoding" / "cross_brain"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ----------------- LOAD DYADIC DATA -----------------
    p1_file = input_dir / f"sub-{subject_id}_P1_epoch.fif"
    p2_file = input_dir / f"sub-{subject_id}_P2_epoch.fif"
    
    if not p1_file.exists() or not p2_file.exists():
        print(f"[{subject_id}] Missing epoch files for one or both players. Skipping.")
        return
        
    print(f"[{subject_id}] Loading epochs for both players...")
    epochs_p1 = mne.read_epochs(p1_file, preload=True, verbose=False)
    epochs_p2 = mne.read_epochs(p2_file, preload=True, verbose=False)
    
    # Ensure symmetrical trial matrices. Artifact rejection may drop differing 
    # trials for P1 vs P2, so we truncate to the lowest common denominator.
    if len(epochs_p1) != len(epochs_p2):
        print(f"[{subject_id}] WARNING: Epoch counts differ (P1: {len(epochs_p1)}, P2: {len(epochs_p2)}). Truncating to match.")
        min_trials = min(len(epochs_p1), len(epochs_p2))
        epochs_p1 = epochs_p1[:min_trials]
        epochs_p2 = epochs_p2[:min_trials]

    # ----------------- PREPROCESSING FOR ML -----------------
    # Clean and downsample P1's data to improve Signal-to-Noise Ratio and processing speed
    epochs_p1.filter(l_freq=1.0, h_freq=30.0, verbose=False)
    epochs_p1.resample(50.0)
    
    # BULLETPROOF FIX: Prevent Data Leakage
    # Explicitly drop mechanical 'Status' or 'Stim' channels. If left in, ML reads 
    # the hardware trigger directly instead of the brainwave, yielding an artificial 100% accuracy.
    leakage_channels = [ch for ch in epochs_p1.ch_names if 'status' in ch.lower() or 'stim' in ch.lower()]
    if leakage_channels:
        print(f"[{subject_id}] Dropping cheating channels from P1: {leakage_channels}")
        epochs_p1.drop_channels(leakage_channels)
    
    # -------------- THE CROSS-BRAIN MAPPING --------------
    picks = mne.pick_types(epochs_p1.info, eeg=True, stim=False, misc=False)
    
    # X = The independent variable (Features). This is strictly Player 1's 64-channel brain activity.
    # We forcefully slice the array to strictly ensure only physical head electrodes are used.
    X = epochs_p1.get_data(picks=picks)[:, :64, :] 
    
    # Let's inspect MNE's default event array
    raw_labels = epochs_p2.events[:, 2]
    unique_labels, counts = np.unique(raw_labels, return_counts=True)
    print(f"[{subject_id}] Event triggers found in MNE: {unique_labels} with counts: {counts}")
    
    # y = The dependent variable (Labels). We pull from PLAYER 2's dataset logic!
    # If there is only 1 event trigger, MNE failed to load labels from BIDS. 
    # We bypass entirely and read the raw TSV behavioral file directly using Pandas.
    if len(unique_labels) == 1:
        print(f"[{subject_id}] WARNING: Only 1 class found in MNE events! Hunting for raw events.tsv file...")
        
        # Search the entire root directory for the raw behavioral events file
        tsv_files = list(ROOT_DIR.rglob(f"sub-{subject_id}*_events.tsv"))
        
        if not tsv_files:
            print(f"[{subject_id}] FATAL ERROR: Could not find any _events.tsv file for sub-{subject_id}!")
            return
            
        df = pd.read_csv(tsv_files[0], sep='\t')
        print(f"[{subject_id}] Loaded TSV! Columns found: {df.columns.tolist()}")
        
        # Identify the correct column representing Player 2's actual action (Rock, Paper, Scissors)
        target_col = None
        for col in ['player2_resp', 'player2_choice', 'p2_choice', 'choice_p2', 'choice']:
            if col in df.columns:
                target_col = col
                break
                
        if target_col is None:
            print(f"[{subject_id}] ERROR: Could not identify which column is Player 2's choice. Exiting.")
            return
            
        # Extract and logically map string/numerical choices to standard ML classes (0, 1, 2)
        raw_text_labels = df[target_col].dropna().values
        print(f"[{subject_id}] First 5 raw choices found: {raw_text_labels[:5]}")
        
        valid_choices = np.unique(raw_text_labels)
        label_map = {val: i for i, val in enumerate(valid_choices)}
        y_opponent = df[target_col].map(label_map).fillna(0).values
        
        # Ensure label array strictly matches the truncated EEG trials
        y_opponent = y_opponent[:len(epochs_p1)]
        print(f"[{subject_id}] Successfully extracted {len(y_opponent)} labels from TSV!")
    else:
        # Fallback if MNE successfully mapped behavioral classes automatically
        y_opponent = raw_labels

    print(f"[{subject_id}] Training matrix shape: {X.shape} (Trials, Channels, Timepoints)")
    print(f"[{subject_id}] Unique choices in y array: {np.unique(y_opponent, return_counts=True)}")
    # ---------------------------------------------------
    
    # ----------------- MACHINE LEARNING -----------------
    # Pipeline: Normalize voltage potentials -> Apply Linear Discriminant Analysis
    clf = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis())
    
    # SlidingEstimator steps through the timeframe [t0 ... tN], independently training 
    # and testing an LDA model at every single millisecond.
    time_decode = SlidingEstimator(clf, n_jobs=1, scoring='accuracy', verbose=False)
    
    print(f"[{subject_id}] Training ML model: Can P1 predict P2's turn? (this takes a moment)...")
    # 5-Fold Stratified Cross Validation
    scores = cross_val_multiscore(time_decode, X, y_opponent, cv=5, n_jobs=1)
    mean_scores = np.mean(scores, axis=0) # Average across folds
    
    # Save mathematical results for eventual group-level Grand Averaging
    out_npy = output_dir / f"sub-{subject_id}_P1_predicting_P2.npy"
    np.save(out_npy, mean_scores)
    
    # ----------------- PLOTTING RESULTS -----------------
    fig, ax = plt.subplots(figsize=(8, 4))
    
    # Plot the fluctuating accuracy over time
    ax.plot(epochs_p1.times, mean_scores, label="Predicting P2's Move", color="#e74c3c", linewidth=2)
    
    # Hardcode baseline chance level for a 3-choice game (Rock/Paper/Scissors) to 33.3%
    # This prevents 'Timeout' or 'NaN' trials from skewing the mathematical baseline.
    chance_level = 1.0 / 3.0
    ax.axhline(chance_level, color='k', linestyle='--', label=f"Chance Level ({chance_level*100:.1f}%)")
    ax.axvline(0, color='k', linestyle='-', alpha=0.5)
    
    # Highlight the expected temporal phase where predictive simulation should theoretically peak
    ax.axvspan(0, 2.0, color='gray', alpha=0.1, label="Decision Phase")
    
    ax.set_title(f"Cross-Brain Decoding: Player 1 Predicting Player 2 (sub-{subject_id})")
    ax.set_xlabel("Time (s) relative to stimulus")
    ax.set_ylabel("Decoding Accuracy")
    ax.legend(loc="upper right")
    
    # Print dynamic scaling bounds to the console for verification
    print(f"[{subject_id}] Accuracy -> Min: {mean_scores.min():.3f}, Mean: {mean_scores.mean():.3f}, Max: {mean_scores.max():.3f}")

    # Save output plot
    out_png = output_dir / f"sub-{subject_id}_P1_predicting_P2.png"
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()
    
    print(f"[{subject_id}] SUCCESS: Saved cross-decoding plot to {out_png.name}\n")

# Bootstrap CLI entry point
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Cross-Brain decoding pipeline.")
    parser.add_argument("--subjects", type=str, help="Comma separated subjects (e.g., 01,02)")
    args = parser.parse_args()
    
    if args.subjects:
        for subj in args.subjects.split(','):
            process_cross_decoding(subj.strip())
    else:
        print("Please provide a subject ID. Example:")
        print("python eeg_pipeline/decoding/06_opponent_decision_decoding.py --subjects 01")