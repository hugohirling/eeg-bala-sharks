"""
Time-Frequency Analysis Module
- Compute power in frequency bands
- Time-frequency representations

Author: Ayush
"""

import numpy as np
import mne


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
