import argparse
import sys
from pathlib import Path
import mne
import numpy as np

# Add the root directory to sys.path to import paths.py
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import paths
# =============================================================================
# FREQUENCY BANDS
# =============================================================================

FREQ_BANDS = {
    'Theta': (4, 8),
    'Alpha': (8, 13),
    'Beta': (13, 30),
    'Gamma': (30, 40)
}


# =============================================================================
# TIME-FREQUENCY ANALYSIS
# =============================================================================

def compute_tfr(epochs, freqs=None, n_cycles=None, average=True):
    """
    Compute time-frequency representation using Morlet wavelets.
    
    Parameters
    ----------
    epochs : mne.Epochs
        Epoched EEG data
    freqs : array-like, optional
        Frequencies to analyze (default: 4-40 Hz)
    average : bool
        Whether to average across epochs
    
    Returns
    -------
    power : mne.time_frequency.AverageTFR or EpochsTFR
    """
    if freqs is None:
        freqs = np.arange(4, 41, 2)
    
    if n_cycles is None:
        n_cycles = freqs / 2
    
    power = mne.time_frequency.tfr_morlet(
        epochs,
        freqs=freqs,
        n_cycles=n_cycles,
        return_itc=False,
        average=average,
        verbose=False
    )
    
    return power


def extract_band_power(power, band_name):
    """
    Extract power in a specific frequency band.
    
    Parameters
    ----------
    power : mne.time_frequency.AverageTFR
        Time-frequency power
    band_name : str
        Name of band ('Theta', 'Alpha', 'Beta', 'Gamma')
    
    Returns
    -------
    band_power : np.ndarray
        Power averaged across the frequency band
    times : np.ndarray
        Time points
    """
    freq_range = FREQ_BANDS[band_name]
    freq_mask = (power.freqs >= freq_range[0]) & (power.freqs <= freq_range[1])
    band_power = power.data[:, freq_mask, :].mean(axis=1)
    
    return band_power, power.times


def get_single_trial_band_power(epochs, freqs, band_range):
    """
    Get band power for each trial (for statistics).
    
    Returns
    -------
    band_power : np.ndarray
        Power per trial in dB (n_epochs,)
    """
    tfr = mne.time_frequency.tfr_morlet(
        epochs,
        freqs=freqs,
        n_cycles=freqs/2,
        return_itc=False,
        average=False,
        verbose=False
    )
    
    freq_mask = (tfr.freqs >= band_range[0]) & (tfr.freqs <= band_range[1])
    time_mask = tfr.times > 0
    
    power_data = tfr.data[:, :, freq_mask, :][:, :, :, time_mask]
    band_power = power_data.mean(axis=(1, 2, 3))
    band_power_db = 10 * np.log10(band_power + 1e-10)
    
    return band_power_db


def get_power_timecourse(epochs, freqs, band_range):
    """
    Get band power over time for each trial.
    
    Returns
    -------
    power_time : np.ndarray
        Power over time (n_epochs, n_times)
    times : np.ndarray
        Time points
    """
    tfr = mne.time_frequency.tfr_morlet(
        epochs,
        freqs=freqs,
        n_cycles=freqs/2,
        return_itc=False,
        average=False,
        verbose=False
    )
    
    freq_mask = (tfr.freqs >= band_range[0]) & (tfr.freqs <= band_range[1])
    power_time = tfr.data[:, :, freq_mask, :].mean(axis=(1, 2))
    power_time_db = 10 * np.log10(power_time + 1e-10)
    
    return power_time_db, tfr.times

def process_subject(subject_id: str):
    """
    Run time-frequency analysis for a specific subject's preprocessed EEG data.
    """
    print(f"\n[{subject_id}] Starting Time-Frequency Analysis...")
    
    # Inputs come from preprocessing, outputs go to a new analysis subfolder
    input_dir = paths.OUTPUT_DIR / "preprocessing"
    output_dir = paths.OUTPUT_DIR / "analysis" / "time_frequency"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process both Player 1 and Player 2
    for player in ["P1", "P2"]:
        epoch_file = input_dir / f"sub-{subject_id}_{player}_epoch.fif"
        
        if not epoch_file.exists():
            print(f"[{subject_id}] WARNING: Epochs file not found at {epoch_file.name}. Skipping {player}.")
            continue
            
        print(f"[{subject_id}] Loading epochs for {player}...")
        try:
            epochs = mne.read_epochs(epoch_file, preload=True, verbose=False)
            
            # NOTE: Assuming you have a `compute_tfr(epochs)` function in this file.
            # If your function requires specific frequency ranges, adjust the arguments below.
            print(f"[{subject_id}] Computing Time-Frequency Representation for {player}...")
            power = compute_tfr(epochs) 
            
            # MNE saves TFR objects as HDF5 files (.h5)
            out_file = output_dir / f"sub-{subject_id}_{player}_tfr.h5"
            power.save(out_file, overwrite=True)
            print(f"[{subject_id}] SUCCESS: Saved TFR to {out_file.name}")
            
        except Exception as e:
            print(f"[{subject_id}] ERROR processing {player}: {e}")

    print(f"[{subject_id}] Finished Time-Frequency Analysis.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run time-frequency analysis step.")
    parser.add_argument("--subject", type=str, required=True, help="Subject ID (e.g., 01)")
    args = parser.parse_args()
    
    process_subject(args.subject)