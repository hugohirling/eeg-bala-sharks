"""
Time-Frequency Analysis Module

This module performs spectral analysis on EEG epoched data. It utilizes complex 
Morlet wavelets to extract Event-Related Spectral Perturbations (ERSP). 
It maps raw brainwave voltage into specific frequency bands (Theta, Alpha, Beta, Gamma)
to observe how brain network power fluctuates over time relative to decision-making.
"""

import argparse
import sys
from pathlib import Path
import mne
import numpy as np

# Ensure the root directory is accessible to import custom path configurations
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import paths

# =============================================================================
# FREQUENCY BANDS
# =============================================================================

# Standard cognitive neuroscience frequency bands (in Hertz)
# Theta: Memory and cognitive control
# Alpha: Inhibitory control and visual attention
# Beta: Motor control and active concentration
# Gamma: High-level information processing and cross-brain synchrony
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
    Compute time-frequency representation (TFR) using complex Morlet wavelets.
    
    This applies wavelet convolution to extract power at specific frequencies over time.
    
    Parameters
    ----------
    epochs : mne.Epochs
        Epoched, preprocessed EEG data.
    freqs : array-like, optional
        Frequencies of interest to analyze. Defaults to 4 Hz - 40 Hz (step 2).
    n_cycles : array-like, optional
        Number of cycles per wavelet. Controls the trade-off between temporal 
        and spectral resolution. Defaults to dynamic scaling (freqs/2).
    average : bool
        If True, averages power across all trials. If False, retains single-trial data.
    
    Returns
    -------
    power : mne.time_frequency.AverageTFR or mne.time_frequency.EpochsTFR
        The computed time-frequency power matrices.
    """
    if freqs is None:
        # Array of frequencies from 4 to 40 Hz, skipping by 2 (4, 6, 8... 40)
        freqs = np.arange(4, 41, 2)
    
    if n_cycles is None:
        # Dynamic cycle sizing: higher frequencies get more cycles.
        # This provides good time resolution at low freqs and good freq resolution at high freqs.
        n_cycles = freqs / 2
    
    power = mne.time_frequency.tfr_morlet(
        epochs,
        freqs=freqs,
        n_cycles=n_cycles,
        return_itc=False, # We only want Power, not Inter-Trial Coherence (ITC)
        average=average,
        verbose=False
    )
    
    return power


def extract_band_power(power, band_name):
    """
    Isolate and extract average power within a specific frequency band.
    
    Parameters
    ----------
    power : mne.time_frequency.AverageTFR
        Pre-computed time-frequency representation.
    band_name : str
        Dictionary key mapping to a specific frequency range (e.g., 'Alpha').
    
    Returns
    -------
    band_power : np.ndarray
        Power arrays strictly constrained to and averaged across the requested band.
    times : np.ndarray
        Corresponding time points for plotting.
    """
    freq_range = FREQ_BANDS[band_name]
    
    # Create a boolean mask identifying which frequencies belong to the requested band
    freq_mask = (power.freqs >= freq_range[0]) & (power.freqs <= freq_range[1])
    
    # Slice the data array and average across the frequency dimension (axis 1)
    band_power = power.data[:, freq_mask, :].mean(axis=1)
    
    return band_power, power.times


def get_single_trial_band_power(epochs, freqs, band_range):
    """
    Extract scalar power estimations for every single trial. 
    Crucial for trial-by-trial statistical testing and machine learning.
    
    Parameters
    ----------
    epochs : mne.Epochs
        Epoched EEG data.
    freqs : array-like
        Frequencies to base the Morlet wavelets on.
    band_range : tuple
        (low_freq, high_freq) boundaries.
        
    Returns
    -------
    band_power_db : np.ndarray
        Collapsed scalar power score per trial, transformed to decibels (dB).
        Shape: (n_epochs,)
    """
    # Force average=False to retain the individual trial dimension
    tfr = mne.time_frequency.tfr_morlet(
        epochs,
        freqs=freqs,
        n_cycles=freqs/2,
        return_itc=False,
        average=False,
        verbose=False
    )
    
    # Mask specific frequency band and only positive timepoints (post-stimulus decision phase)
    freq_mask = (tfr.freqs >= band_range[0]) & (tfr.freqs <= band_range[1])
    time_mask = tfr.times > 0
    
    power_data = tfr.data[:, :, freq_mask, :][:, :, :, time_mask]
    
    # Collapse dimensions: Average across Channels (1), Frequencies (2), and Time (3)
    band_power = power_data.mean(axis=(1, 2, 3))
    
    # Convert raw voltage power to scientific decibels (dB) using 10*log10
    # +1e-10 prevents log(0) mathematical errors
    band_power_db = 10 * np.log10(band_power + 1e-10)
    
    return band_power_db


def get_power_timecourse(epochs, freqs, band_range):
    """
    Extract single-trial power traces fluctuating over time.
    
    Unlike 'get_single_trial_band_power', this function preserves the time axis,
    allowing visualization of how brain power changes millisecond by millisecond.
    
    Returns
    -------
    power_time_db : np.ndarray
        Power fluctuating over time per trial. Shape: (n_epochs, n_times)
    times : np.ndarray
        Time points corresponding to the data array.
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
    
    # Average across Channels (1) and Frequencies (2), but PRESERVE Time (3)
    power_time = tfr.data[:, :, freq_mask, :].mean(axis=(1, 2))
    
    # Convert to decibels (dB)
    power_time_db = 10 * np.log10(power_time + 1e-10)
    
    return power_time_db, tfr.times


# =============================================================================
# PIPELINE EXECUTION
# =============================================================================

def process_subject(subject_id: str):
    """
    Main orchestration function to run time-frequency analysis for a subject.
    Extracts multi-dimensional TFR matrices and saves them in HDF5 format.
    """
    print(f"\n[{subject_id}] Starting Time-Frequency Analysis...")
    
    # Define file routing
    input_dir = paths.OUTPUT_DIR / "preprocessing"
    output_dir = paths.OUTPUT_DIR / "analysis" / "time_frequency"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process both dyads independently
    for player in ["P1", "P2"]:
        epoch_file = input_dir / f"sub-{subject_id}_{player}_epoch.fif"
        
        if not epoch_file.exists():
            print(f"[{subject_id}] WARNING: Epochs file not found at {epoch_file.name}. Skipping {player}.")
            continue
            
        print(f"[{subject_id}] Loading epochs for {player}...")
        try:
            epochs = mne.read_epochs(epoch_file, preload=True, verbose=False)
            
            print(f"[{subject_id}] Computing Time-Frequency Representation for {player}...")
            power = compute_tfr(epochs) 
            
            # MNE saves heavy multidimensional TFR objects as HDF5 binaries (.h5)
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