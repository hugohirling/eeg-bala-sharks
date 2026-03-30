from __future__ import annotations

import argparse
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

import mne
from mne.decoding import GeneralizingEstimator, cross_val_multiscore
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

CURRENT_DIR = Path(__file__).resolve().parent
# Go two levels up to the root folder (eeg-bala-sharks) where paths.py is located
ROOT_DIR = CURRENT_DIR.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


# Update with your custom paths module if needed
import paths 

def process_tgm(subject_id: str):
    print(f"\n[{subject_id}] Starting Temporal Generalization Matrix (TGM)...")
    
    input_dir = paths.OUTPUT_DIR / "preprocessing"
    output_dir = paths.OUTPUT_DIR / "analysis" / "decoding" / "tgm"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for player in ["P1", "P2"]:
        epoch_file = input_dir / f"sub-{subject_id}_{player}_epoch.fif"
        if not epoch_file.exists():
            print(f"[{subject_id}] File not found: {epoch_file.name}. Skipping.")
            continue
            
        print(f"[{subject_id} - {player}] Loading epochs...")
        epochs = mne.read_epochs(epoch_file, preload=True, verbose=False)
        
        # Isolate the stimulus/decision phase data
        epochs.filter(l_freq=1.0, h_freq=30.0, verbose=False)  # Clean bandpass for ML
        
        X = epochs.get_data()
        y = epochs.events[:, 2] # Assuming 1=Rock, 2=Paper, 3=Scissors
        
        # Pipeline: Normalize EEG data -> Classify with LDA
        clf = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis())
        
        # GeneralizingEstimator trains at time T, tests at time T'
        time_gen = GeneralizingEstimator(clf, n_jobs=1, scoring='accuracy', verbose=True)
        
        print(f"[{subject_id} - {player}] Fitting Generalization Matrix (this takes a moment)...")
        # 3-Fold Cross validation
        scores = cross_val_multiscore(time_gen, X, y, cv=3, n_jobs=1)
        mean_scores = np.mean(scores, axis=0) # Average over folds
        
        # Save exact numpy arrays for statistical testing later
        out_npy = output_dir / f"sub-{subject_id}_{player}_tgm_scores.npy"
        np.save(out_npy, mean_scores)
        
        # Create the beautiful 2D Heatmap
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        im = ax.imshow(
            mean_scores, 
            interpolation="lanczos", 
            origin="lower", 
            cmap="RdBu_r",
            extent=epochs.times[[0, -1, 0, -1]],
            vmin=0.2, vmax=0.5 # Chance level is 0.33 for 3 classes
        )
        ax.set_xlabel("Testing Time (s)")
        ax.set_ylabel("Training Time (s)")
        ax.set_title(f"Temporal Generalization: {player} (sub-{subject_id})")
        ax.axvline(0, color="k", linestyle="--", alpha=0.5)
        ax.axhline(0, color="k", linestyle="--", alpha=0.5)
        
        # Add a diagonal line (where Training Time == Testing Time)
        ax.plot(epochs.times[[0, -1]], epochs.times[[0, -1]], color='k', linestyle=":", alpha=0.5)
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Accuracy (Chance = 33%)")
        
        # Save Plot
        out_png = output_dir / f"sub-{subject_id}_{player}_tgm.png"
        plt.tight_layout()
        plt.savefig(out_png, dpi=300)
        plt.close()
        
        print(f"[{subject_id} - {player}] SUCCESS: Saved TGM to {out_png.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Temporal Generalization Decoding.")
    parser.add_argument("--subjects", type=str, help="Comma-separated subjects (e.g., 01,02)")
    args = parser.parse_args()
    
    if args.subjects:
        subject_list = [s.strip() for s in args.subjects.split(',')]
        for subj in subject_list:
            process_tgm(subj)
    else:
        print("Please provide subjects: --subjects 01,02,03")