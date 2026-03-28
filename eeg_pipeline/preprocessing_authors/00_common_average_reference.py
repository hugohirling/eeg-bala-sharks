"""
Common Average Reference (CAR) re-referencing
Following Moerel et al. (2025) preprocessing pipeline
Step 1: Re-reference to common average
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


def load_raw_data(input_file):
    """Load raw EEG data from .fif file"""
    logger.info(f"Loading raw data from: {input_file}")
    raw = mne.io.read_raw_fif(input_file, preload=True)
    logger.info(f"Data shape: {raw.get_data().shape}")
    logger.info(f"Sampling rate: {raw.info['sfreq']} Hz")
    return raw


def apply_common_average_reference(raw):
    """
    Apply common average reference (CAR)
    Following Moerel et al. (2025): "we re-referenced the data to the common average"
    """
    logger.info("Applying common average reference...")
    raw.set_eeg_reference(ref_channels='average')
    logger.info("Common average reference applied successfully")
    return raw


def save_processed_data(raw, output_file):
    """Save processed data"""
    logger.info(f"Saving processed data to: {output_file}")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    raw.save(output_file, overwrite=True)
    logger.info(f"File saved: {output_file}")


def main(input_file, output_file):
    """Main processing function"""
    logger.info("=" * 60)
    logger.info("Step 1: Common Average Reference (CAR)")
    logger.info("=" * 60)
    
    # Load data
    raw = load_raw_data(input_file)
    
    # Apply CAR
    raw = apply_common_average_reference(raw)
    
    # Save
    save_processed_data(raw, output_file)
    
    logger.info("Step 1 completed successfully!")
    logger.info("=" * 60)


if __name__ == "__main__":
    # Example usage
    from config import settings
    
    subject = "sub-01"
    input_dir = Path(settings.DATA_ROOT) / subject
    output_dir = Path(settings.OUTPUT_ROOT) / "preprocessing_authors"
    
    input_file = input_dir / f"{subject}_raw.fif"
    output_file = output_dir / f"{subject}_car.fif"
    
    if input_file.exists():
        main(str(input_file), str(output_file))
    else:
        logger.warning(f"Input file not found: {input_file}")
