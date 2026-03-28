"""
Interpolate Noisy Channels
Following Moerel et al. (2025) preprocessing pipeline
Step 3: Channel interpolation using neighboring channels
"""

import os
import logging
import mne
import json
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_data(input_file):
    """Load EEG data with CAR already applied"""
    logger.info(f"Loading data from: {input_file}")
    raw = mne.io.read_raw_fif(input_file, preload=True)
    logger.info(f"Data shape: {raw.get_data().shape}")
    return raw


def load_noisy_channels(log_file):
    """Load noisy channels from log file"""
    logger.info(f"Loading noisy channels from: {log_file}")
    
    if not os.path.exists(log_file):
        logger.warning(f"Log file not found: {log_file}")
        logger.info("No noisy channels will be interpolated")
        return []
    
    with open(log_file, 'r') as f:
        info = json.load(f)
    
    noisy_channels = info.get('noisy_channels', [])
    logger.info(f"Found {len(noisy_channels)} noisy channels to interpolate: {noisy_channels}")
    return noisy_channels


def interpolate_bad_channels(raw, bad_channels):
    """
    Interpolate noisy channels using neighboring channels
    Following Moerel et al. (2025): 
    "We interpolated noisy channels based on neighbouring channels, 
    using the ft_channelrepair function with a distance measure of 0.5 cm"
    
    MNE's interpolate_bads() function implements similar functionality
    """
    
    if not bad_channels:
        logger.info("No bad channels to interpolate")
        return raw
    
    logger.info(f"Setting bad channels: {bad_channels}")
    raw.info['bads'] = bad_channels
    
    logger.info(f"Interpolating {len(bad_channels)} bad channels using neighboring channels...")
    logger.info("(MNE's spherical spline interpolation, similar to FieldTrip's ft_channelrepair)")
    
    # Interpolate bad channels
    raw.interpolate_bads(reset_bads=True)
    
    logger.info("Channel interpolation completed")
    return raw


def save_processed_data(raw, output_file):
    """Save processed data"""
    logger.info(f"Saving processed data to: {output_file}")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    raw.save(output_file, overwrite=True)
    logger.info(f"File saved: {output_file}")


def main(input_file, noisy_channels_log, output_file):
    """Main processing function"""
    logger.info("=" * 60)
    logger.info("Step 3: Interpolate Noisy Channels")
    logger.info("=" * 60)
    
    # Load data
    raw = load_data(input_file)
    
    # Load noisy channels
    bad_channels = load_noisy_channels(noisy_channels_log)
    
    # Interpolate
    raw = interpolate_bad_channels(raw, bad_channels)
    
    # Save
    save_processed_data(raw, output_file)
    
    logger.info("Step 3 completed successfully!")
    logger.info("=" * 60)


if __name__ == "__main__":
    # Example usage
    from config import settings
    
    subject = "sub-01"
    input_dir = Path(settings.DATA_ROOT) / subject
    output_dir = Path(settings.OUTPUT_ROOT) / "preprocessing_authors"
    qc_dir = output_dir / "qc"
    
    input_file = input_dir / f"{subject}_car.fif"
    noisy_channels_log = qc_dir / f"{subject}_noisy_channels.json"
    output_file = output_dir / f"{subject}_interpolated.fif"
    
    if input_file.exists():
        main(str(input_file), str(noisy_channels_log), str(output_file))
    else:
        logger.warning(f"Input file not found: {input_file}")
