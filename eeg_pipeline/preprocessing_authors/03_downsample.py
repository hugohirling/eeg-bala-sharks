"""
Downsample EEG Data
Following Moerel et al. (2025) preprocessing pipeline
Step 4: Downsample from 2048 Hz to 256 Hz
"""

import os
import logging
import mne
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_data(input_file):
    """Load EEG data"""
    logger.info(f"Loading data from: {input_file}")
    raw = mne.io.read_raw_fif(input_file, preload=True)
    logger.info(f"Original sampling rate: {raw.info['sfreq']} Hz")
    logger.info(f"Data shape: {raw.get_data().shape}")
    return raw


def downsample_data(raw, target_sfreq=256):
    """
    Downsample data to target sampling rate
    Following Moerel et al. (2025): "We then down-sampled the data to 256 Hz."
    
    Parameters
    ----------
    raw : mne.io.Raw
        Raw EEG data
    target_sfreq : int
        Target sampling frequency (default: 256 Hz)
    """
    current_sfreq = raw.info['sfreq']
    
    if current_sfreq == target_sfreq:
        logger.info(f"Data already at {target_sfreq} Hz, skipping downsampling")
        return raw
    
    logger.info(f"Downsampling from {current_sfreq} Hz to {target_sfreq} Hz...")
    raw.resample(target_sfreq)
    logger.info(f"Downsampling completed. New sampling rate: {raw.info['sfreq']} Hz")
    
    return raw


def save_processed_data(raw, output_file):
    """Save processed data"""
    logger.info(f"Saving downsampled data to: {output_file}")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    raw.save(output_file, overwrite=True)
    logger.info(f"File saved: {output_file}")


def main(input_file, output_file, target_sfreq=256):
    """Main processing function"""
    logger.info("=" * 60)
    logger.info(f"Step 4: Downsample to {target_sfreq} Hz")
    logger.info("=" * 60)
    
    # Load data
    raw = load_data(input_file)
    
    # Downsample
    raw = downsample_data(raw, target_sfreq=target_sfreq)
    
    # Save
    save_processed_data(raw, output_file)
    
    logger.info("Step 4 completed successfully!")
    logger.info("=" * 60)


if __name__ == "__main__":
    # Example usage
    from config import settings
    
    subject = "sub-01"
    output_dir = Path(settings.OUTPUT_ROOT) / "preprocessing_authors"
    
    input_file = output_dir / f"{subject}_interpolated.fif"
    output_file = output_dir / f"{subject}_downsampled.fif"
    
    if input_file.exists():
        main(str(input_file), str(output_file), target_sfreq=256)
    else:
        logger.warning(f"Input file not found: {input_file}")
