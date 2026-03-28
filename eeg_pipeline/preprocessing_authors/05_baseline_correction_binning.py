"""
Baseline Correction and Time Binning
Following Moerel et al. (2025) preprocessing pipeline
Step 6: Apply baseline correction and bin data into 250 ms time bins
"""

import os
import logging
import mne
import numpy as np
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_epochs(epochs_file):
    """Load epoch data"""
    logger.info(f"Loading epochs from: {epochs_file}")
    epochs = mne.read_epochs(epochs_file, preload=True)
    logger.info(f"Loaded {len(epochs)} epochs")
    logger.info(f"Data shape: {epochs.get_data().shape}")
    return epochs


def apply_baseline_correction(epochs):
    """
    Apply baseline correction
    Following Moerel et al. (2025):
    "We applied baseline corrections for each separate epoch, using the window from -200 ms to 0 ms"
    
    Note: Baseline correction was already applied during epoching
    This function documents the process
    """
    logger.info("Baseline correction window: -200 ms to 0 ms")
    logger.info("(Already applied during epoching)")
    return epochs


def create_time_bins(epochs, bin_duration=0.25):
    """
    Bin data into 250 ms time bins
    Following Moerel et al. (2025):
    "we averaged the resulting data into 250 ms time bins, resulting in a total of 20 time bins 
    for the 0 to 5000 ms time-course"
    
    Parameters
    ----------
    epochs : mne.Epochs
        Epoch data
    bin_duration : float
        Duration of each time bin in seconds (default: 0.25 = 250 ms)
    """
    
    logger.info(f"Creating {bin_duration * 1000:.0f} ms time bins...")
    
    data = epochs.get_data()  # Shape: (n_epochs, n_channels, n_timepoints)
    sfreq = epochs.info['sfreq']
    times = epochs.times
    
    # Determine bin edges
    bin_size_samples = int(bin_duration * sfreq)
    
    # Create time bins
    binned_data = []
    bin_times = []
    
    start_idx = 0
    while start_idx < data.shape[2]:
        end_idx = min(start_idx + bin_size_samples, data.shape[2])
        
        # Average data in this bin
        bin_data = np.mean(data[:, :, start_idx:end_idx], axis=2)
        binned_data.append(bin_data)
        
        # Get time point for this bin (middle of the bin)
        bin_time = np.mean(times[start_idx:end_idx])
        bin_times.append(bin_time)
        
        start_idx = end_idx
    
    # Stack binned data
    binned_data = np.stack(binned_data, axis=2)  # Shape: (n_epochs, n_channels, n_bins)
    bin_times = np.array(bin_times)
    
    logger.info(f"Original time points: {data.shape[2]} (duration: {times[-1]:.3f} s)")
    logger.info(f"Binned time points: {binned_data.shape[2]} (bin duration: {bin_duration} s)")
    logger.info(f"Time range: {times[0]:.3f} to {times[-1]:.3f} seconds")
    
    return binned_data, bin_times, bin_size_samples


def create_binned_epochs(epochs, binned_data, bin_times):
    """
    Create new epochs object with binned data
    """
    logger.info("Creating binned epochs object...")
    
    # Create new epochs with binned times
    binned_epochs = epochs.copy()
    
    # Replace the data with binned data
    binned_epochs._data = binned_data
    
    # Update the times array
    binned_epochs.times = bin_times
    
    return binned_epochs


def process_all_phases(epochs_dir, subject_id, bin_duration=0.25):
    """Process all three phases (decision, response, feedback)"""
    
    logger.info("=" * 60)
    logger.info("Step 6: Baseline Correction and Time Binning")
    logger.info("=" * 60)
    
    phases = ['decision', 'response', 'feedback']
    binned_epochs_dict = {}
    metadata = {}
    
    for phase in phases:
        logger.info(f"\nProcessing {phase} phase...")
        
        # Load epochs for this phase
        epochs_file = os.path.join(epochs_dir, f'{subject_id}_{phase}-epo.fif')
        
        if not os.path.exists(epochs_file):
            logger.warning(f"Epochs file not found: {epochs_file}")
            continue
        
        epochs = load_epochs(epochs_file)
        
        # Apply baseline correction (already applied, this is for documentation)
        epochs = apply_baseline_correction(epochs)
        
        # Create time bins
        binned_data, bin_times, bin_size = create_time_bins(epochs, bin_duration=bin_duration)
        
        # Create binned epochs
        binned_epochs = create_binned_epochs(epochs, binned_data, bin_times)
        
        binned_epochs_dict[phase] = binned_epochs
        
        # Save binned epochs
        output_file = os.path.join(epochs_dir, f'{subject_id}_{phase}_binned-epo.fif')
        binned_epochs.save(output_file, overwrite=True)
        logger.info(f"Saved binned {phase} epochs: {output_file}")
        
        metadata[phase] = {
            'n_epochs': len(binned_epochs),
            'n_channels': len(binned_epochs.ch_names),
            'n_bins': binned_epochs.get_data().shape[2],
            'bin_duration_ms': int(bin_duration * 1000),
            'time_range': (float(bin_times[0]), float(bin_times[-1]))
        }
    
    logger.info("\n" + "=" * 60)
    logger.info("Step 6 completed successfully!")
    logger.info("=" * 60)
    logger.info("Summary of binned epochs:")
    for phase, meta in metadata.items():
        logger.info(f"\n{phase.upper()} phase:")
        logger.info(f"  - N epochs: {meta['n_epochs']}")
        logger.info(f"  - N channels: {meta['n_channels']}")
        logger.info(f"  - N time bins: {meta['n_bins']}")
        logger.info(f"  - Bin duration: {meta['bin_duration_ms']} ms")
        logger.info(f"  - Time range: {meta['time_range'][0]:.3f} to {meta['time_range'][1]:.3f} s")
    
    return binned_epochs_dict, metadata


def main(epochs_dir, subject_id, bin_duration=0.25):
    """Main processing function"""
    binned_epochs_dict, metadata = process_all_phases(
        epochs_dir, subject_id, bin_duration=bin_duration
    )


if __name__ == "__main__":
    # Example usage
    from config import settings
    
    subject = "sub-01"
    epochs_dir = str(Path(settings.OUTPUT_ROOT) / "preprocessing_authors")
    
    main(epochs_dir, subject, bin_duration=0.25)
