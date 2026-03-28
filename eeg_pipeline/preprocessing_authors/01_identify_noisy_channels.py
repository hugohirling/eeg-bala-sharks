"""
Identify Noisy Channels through Visual Inspection
Following Moerel et al. (2025) preprocessing pipeline
Step 2: Identify noisy channels
"""

import os
import logging
import mne
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_data(input_file):
    """Load processed EEG data"""
    logger.info(f"Loading data from: {input_file}")
    raw = mne.io.read_raw_fif(input_file, preload=True)
    logger.info(f"Data shape: {raw.get_data().shape}")
    return raw


def detect_noisy_channels_automated(raw, z_threshold=3.0):
    """
    Automated detection of noisy channels based on variance
    This complements visual inspection per Moerel et al. (2025) approach
    """
    logger.info("Performing automated noise detection (variance-based)...")
    
    data = raw.get_data()
    # Calculate variance for each channel
    variances = np.var(data, axis=1)
    mean_var = np.mean(variances)
    std_var = np.std(variances)
    
    # Identify outliers (channels with unusually high variance)
    z_scores = np.abs((variances - mean_var) / std_var)
    noisy_candidates = np.where(z_scores > z_threshold)[0]
    
    # Get channel names
    channel_names = raw.ch_names[:len(data)]
    noisy_channel_names = [channel_names[i] for i in noisy_candidates]
    
    logger.info(f"Variance-based detection found {len(noisy_channel_names)} candidate noisy channels:")
    for ch in noisy_channel_names:
        logger.info(f"  - {ch}")
    
    return noisy_channel_names, variances


def plot_channel_variances(raw, variances, output_dir):
    """Plot channel variances for visual inspection"""
    logger.info("Plotting channel variances...")
    
    plt.figure(figsize=(14, 6))
    plt.bar(range(len(variances)), variances)
    plt.xlabel('Channel Index')
    plt.ylabel('Variance (μV²)')
    plt.title('EEG Channel Variances - Noisy Channel Detection')
    plt.grid(axis='y', alpha=0.3)
    
    os.makedirs(output_dir, exist_ok=True)
    plot_file = os.path.join(output_dir, 'channel_variances_plot.png')
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    logger.info(f"Variance plot saved: {plot_file}")
    plt.close()


def visualize_raw_data(raw, output_dir):
    """
    Create scrollable plot of raw data for visual inspection
    per Moerel et al. (2025): "We identified noisy channels through visual inspection"
    """
    logger.info("Creating raw data visualization for visual inspection...")
    
    # Plot first 100 seconds at 100 Hz speed
    os.makedirs(output_dir, exist_ok=True)
    
    # Create multiple plots for different time windows
    n_channels = len(raw.ch_names)
    
    fig = raw.plot(
        duration=30,
        n_channels=min(32, n_channels),
        scalings='auto',
        title='Raw EEG Data - Visual Inspection for Noisy Channels'
    )
    
    logger.info("Raw data plot created (interactive window)")
    logger.info("Review channels for high-amplitude noise, flat signals, or abnormal patterns")
    
    return fig


def save_noisy_channels_log(noisy_channels, output_dir, subject_id):
    """Save identified noisy channels to a log file for reference"""
    logger.info("Saving noisy channels log...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    log_file = os.path.join(output_dir, f'{subject_id}_noisy_channels.json')
    
    noisy_info = {
        'subject': subject_id,
        'noisy_channels': noisy_channels,
        'n_noisy_channels': len(noisy_channels),
        'method': 'variance-based + visual inspection'
    }
    
    with open(log_file, 'w') as f:
        json.dump(noisy_info, f, indent=2)
    
    logger.info(f"Noisy channels log saved: {log_file}")
    return log_file


def main(input_file, output_dir, subject_id, interactive=False):
    """Main processing function"""
    logger.info("=" * 60)
    logger.info("Step 2: Identify Noisy Channels")
    logger.info("=" * 60)
    logger.info("Note: Moerel et al. (2025) used visual inspection for channel noise detection")
    logger.info("This script provides automated + visual inspection tools")
    
    # Load data
    raw = load_data(input_file)
    
    # Automated detection
    noisy_channels, variances = detect_noisy_channels_automated(raw)
    
    # Visualizations
    plot_channel_variances(raw, variances, output_dir)
    
    if interactive:
        visualize_raw_data(raw, output_dir)
    
    # Save log
    save_noisy_channels_log(noisy_channels, output_dir, subject_id)
    
    logger.info("Step 2 completed!")
    logger.info("=" * 60)
    logger.info("NEXT STEPS:")
    logger.info("1. Review the generated plots")
    logger.info("2. Identify channels for interpolation based on visual inspection")
    logger.info("3. Pass noisy channel list to next step: interpolation")
    logger.info("=" * 60)


if __name__ == "__main__":
    # Example usage
    from config import settings
    
    subject = "sub-01"
    input_dir = Path(settings.DATA_ROOT) / subject
    output_dir = Path(settings.OUTPUT_ROOT) / "preprocessing_authors" / "qc"
    
    input_file = input_dir / f"{subject}_car.fif"
    
    if input_file.exists():
        main(str(input_file), str(output_dir), subject)
    else:
        logger.warning(f"Input file not found: {input_file}")
