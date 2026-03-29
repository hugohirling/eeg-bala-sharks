"""
Configuration for Authors' Preprocessing Pipeline
Moerel et al. (2025) preprocessing methodology
"""

from pathlib import Path

from preprocessing import config  # also registers project root on sys.path
import paths as _project_paths  # noqa: E402

# ============================================================================
# DATA PATHS
# ============================================================================

#: Output directory for the authors pipeline (sub-folder of the central OUTPUT_DIR).
DEFAULT_OUTPUT_DIR: Path = _project_paths.OUTPUT_DIR / "preprocessing_authors"

# ============================================================================
# PIPELINE ORCHESTRATION
# ============================================================================

STEP_FILE_MAP = {
    "common_average_reference": "preprocessing_authors/00_common_average_reference.py",
    "identify_noisy_channels": "preprocessing_authors/01_identify_noisy_channels.py",
    "interpolate_bad_channels": "preprocessing_authors/02_interpolate_bad_channels.py",
    "downsample": "preprocessing_authors/03_downsample.py",
    "epoch": "preprocessing_authors/04_epoch.py",
    "baseline_correction_binning": "preprocessing_authors/05_baseline_correction_binning.py",
}

PIPELINE_STEPS = list(STEP_FILE_MAP.keys())

STEP_OUTPUT_SUFFIXES = {
    "common_average_reference": "car_authors",
    "identify_noisy_channels": "noisy_checked_authors",
    "interpolate_bad_channels": "interpolated_authors",
    "downsample": "downsampled_authors",
    "epoch": "epoch_authors",
    "baseline_correction_binning": "binned_authors",
}

RAW_FLOW_STEPS = {
    "common_average_reference",
    "identify_noisy_channels",
    "interpolate_bad_channels",
    "downsample",
}

# ============================================================================
# PREPROCESSING PARAMETERS
# ============================================================================

# Step 1: Common Average Reference
# Standard method - re-reference to the average of all channels
CAR_ENABLED = True

# Step 2: Noisy Channel Detection
# Parameters for automated detection (variance-based)
NOISY_CHANNEL_DETECTION = {
    'method': 'variance',           # Method: 'variance' or 'custom'
    'z_threshold': 3.0,             # Z-score threshold for outliers
    'manual_review': True            # Flag for manual visual inspection
}

# Step 3: Channel Interpolation
# Based on Moerel et al.: "ft_channelrepair function with a distance measure of 0.5 cm"
CHANNEL_INTERPOLATION = {
    'method': 'spherical_spline',   # MNE equivalent to FieldTrip ft_channelrepair
    'enabled': True
}

# Step 4: Downsampling
# Target sampling frequency
DOWNSAMPLE_TARGET_SFREQ = 256      # 256 Hz (from original 2048 Hz)

# Step 5: Epoching
# Task phases as per Moerel et al. (2025)
EPOCHING_PARAMS = {
    'decision': {
        'tmin': -0.2,               # -200 ms
        'tmax': 2.0,                # 2000 ms
        'baseline': (-0.2, 0),      # Baseline: -200 to 0 ms
        'description': 'Decision phase'
    },
    'response': {
        'tmin': -0.2,               # -200 ms
        'tmax': 2.0,                # 2000 ms
        'baseline': (-0.2, 0),      # Baseline: -200 to 0 ms
        'description': 'Response phase',
        'offset_from_trial_start': 2.0  # Starts 2 seconds after trial start
    },
    'feedback': {
        'tmin': -0.2,               # -200 ms
        'tmax': 1.0,                # 1000 ms (shorter phase)
        'baseline': (-0.2, 0),      # Baseline: -200 to 0 ms
        'description': 'Feedback phase',
        'offset_from_trial_start': 4.0  # Starts 4 seconds after trial start
    }
}

# Step 6: Time Binning
# Moerel et al.: "250 ms time bins, resulting in 20 time bins for 0 to 5000 ms"
TIME_BINNING = {
    'bin_duration': 0.250,          # 250 ms bins
    'method': 'mean',               # Average within each bin
    'description': '250 ms bins for statistical analysis'
}

# ============================================================================
# FILTERING
# ============================================================================

# Important Note from Moerel et al. (2025):
# "We did not apply filtering, as this has been shown to cause artefacts 
#  or temporally smear the signal"
# References: Delorme (2023), Grootswagers et al. (2017), van Driel et al. (2021)

APPLY_FILTERING = False

# ============================================================================
# HARDWARE SPECIFICATIONS
# ============================================================================

# EEG System: BioSemi Active-Two
EEG_SYSTEM = {
    'name': 'BioSemi Active-Two',
    'n_channels': 64,
    'electrode_system': '10-20',
    'original_sfreq': 2048,         # Original sampling rate
    'reference': 'CMS'              # Common Mode Sense
}

# ============================================================================
# QUALITY CONTROL
# ============================================================================

# Generate QC plots and reports
QC_ENABLED = True

QC_PLOTS = {
    'channel_variances': True,
    'raw_signal_visualization': True,
    'noisy_channel_detection_plots': True,
    'epoch_statistics': True,
    'time_bin_visualization': True
}

# ============================================================================
# LOGGING
# ============================================================================

LOG_LEVEL = 'INFO'                  # DEBUG, INFO, WARNING, ERROR
LOG_DIR = 'eeg_pipeline/preprocessing_authors/logs'
SAVE_LOG_FILE = True

# ============================================================================
# PROCESSING
# ============================================================================

# Parallel processing
PARALLEL_PROCESSING = False
N_JOBS = 4                          # Number of parallel jobs

# Memory settings
MEMORY_EFFICIENT = True             # Use streaming processing for large files
PRELOAD_DATA = True                 # Preload data into memory

# ============================================================================
# VALIDATION
# ============================================================================

# Validate data at each step
VALIDATE_OUTPUT = True

# Expected data shapes/properties
EXPECTED_N_CHANNELS = 64
EXPECTED_DOWNSAMPLE_FREQ = 256
