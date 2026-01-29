"""
Configuration file for EEG Hyperscanning Pipeline
"""

from pathlib import Path
import os

# =============================================================================
# DIRECTORY PATHS
# =============================================================================

# Base directory (where the project is located)
BASE_DIR = Path(__file__).parent.parent

# Raw data directory - pointing to actual data location
RAW_DATA_DIR = BASE_DIR / "MNE-sample-data" / "ds006761"

# BIDS root (alias for compatibility with preprocessing scripts)
BIDS_ROOT = RAW_DATA_DIR

# Processed data directory (pipeline outputs)
PROCESSED_DATA_DIR = BASE_DIR / "processed_data"

# Output directory (alias for compatibility)
OUTPUT_DIR = PROCESSED_DATA_DIR

# Create directories if they don't exist
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# DYAD LIST - Auto-detect from raw_data or processed_data
# =============================================================================

def get_dyads():
    """Auto-detect dyad IDs from available data."""
    dyads = []
    
    # Try to find dyads from processed epochs first
    epochs_dir = PROCESSED_DATA_DIR / "06_epochs"
    if epochs_dir.exists():
        for f in epochs_dir.glob("*-epo.fif"):
            dyad_id = f.stem.replace("-epo", "")
            # Remove 'sub-' prefix if present
            dyad_id = dyad_id.replace("sub-", "")
            if dyad_id not in dyads:
                dyads.append(dyad_id)
    
    # If no epochs, try raw data directory (BIDS format)
    if not dyads and RAW_DATA_DIR.exists():
        for item in RAW_DATA_DIR.iterdir():
            if item.is_dir() and item.name.startswith("sub-"):
                # Remove 'sub-' prefix for BIDS compatibility
                dyad_id = item.name.replace("sub-", "")
                dyads.append(dyad_id)
    
    # Fallback to manual list if nothing found (WITHOUT 'sub-' prefix)
    if not dyads:
        dyads = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10"]
    
    return sorted(dyads)

# ...existing code...

DYADS = get_dyads()

# Limit to first 9 subjects only
DYADS = [d for d in DYADS if d in ["01", "02", "03", "04", "05", "06", "07", "08", "09"]]
DYADS = ["01"]

# Aliases for compatibility with preprocessing scripts
SUBJECTS = DYADS
SUBJECT_IDS = DYADS

# ...existing code...

# =============================================================================
# TASK NAME
# =============================================================================

TASK = "rps"  # Rock-Paper-Scissors task name in BIDS (lowercase to match files)

# =============================================================================
# PREPROCESSING PARAMETERS
# =============================================================================

# Filter settings
FREQ_LOWER = 1.0    # High-pass filter cutoff (Hz)
FREQ_UPPER = 40.0   # Low-pass filter cutoff (Hz)

# Alternative names for compatibility
L_FREQ = FREQ_LOWER
H_FREQ = FREQ_UPPER

# Downsampling
DOWNSAMPLE_SFREQ = 512
SFREQ = DOWNSAMPLE_SFREQ  # Alias for compatibility

# Epoching
MAX_EPOCHS = 500
EPOCH_TMIN = -0.2
EPOCH_TMAX = 5.0
TMIN = EPOCH_TMIN  # Alias
TMAX = EPOCH_TMAX  # Alias

# Baseline correction
BASELINE = (None, 0)  # From start of epoch to time 0

# AutoReject config
AR_N_JOBS = 1       # Number of parallel jobs
AR_VERBOSE = True   # Verbose output

# ICA
ICA_N_COMPONENTS = 20
ICA_MAX_ITER = 500
ICA_METHOD = 'fastica'
ICA_RANDOM_STATE = 42

# Reference
REFERENCE = 'average'  # Common average reference

# =============================================================================
# CHANNEL CONFIGURATION
# =============================================================================

# Channels to drop (if any)
CHANNELS_TO_DROP = []

# EOG channels for ICA
EOG_CHANNELS = ['Fp1', 'Fp2']  # Adjust based on your montage

# =============================================================================
# ANALYSIS PARAMETERS (Steps 07-10)
# =============================================================================

# Frequency bands for time-frequency analysis
FREQ_BANDS = {
    'delta': (1, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 50)
}

# ERP component time windows (in seconds)
ERP_WINDOWS = {
    'N100': (0.08, 0.12),
    'P200': (0.15, 0.25),
    'N200': (0.20, 0.28),
    'P300': (0.28, 0.40),
    'LPP': (0.40, 0.70)
}

# Time-frequency analysis
TFR_FREQS_MIN = 4      # Min frequency for TFR (Hz)
TFR_FREQS_MAX = 50     # Max frequency for TFR (Hz)
TFR_N_FREQS = 25       # Number of frequency bins

# Statistical testing
N_PERMUTATIONS = 1000  # For cluster permutation tests
CLUSTER_THRESHOLD = 2.0
ALPHA = 0.05           # Significance level

# Bayesian analysis
NULL_INTERVAL = (0, 0.5)  # Null interval for effect sizes
PRIOR_SCALE = 0.707       # Cauchy prior scale for Bayes Factor

# =============================================================================
# OUTPUT SUBDIRECTORIES
# =============================================================================

# Define output subdirectories for each step
OUTPUT_DIRS = {
    'load': PROCESSED_DATA_DIR / "00_loaded",
    'inspect': PROCESSED_DATA_DIR / "01_inspected",
    'rereference': PROCESSED_DATA_DIR / "02_rereferenced",
    'filter': PROCESSED_DATA_DIR / "03_filtered",
    'ica': PROCESSED_DATA_DIR / "04_ica",
    'downsample': PROCESSED_DATA_DIR / "05_downsampled",
    'epochs': PROCESSED_DATA_DIR / "06_epochs",
    'predictability': PROCESSED_DATA_DIR / "07_predictability",
    'plv': PROCESSED_DATA_DIR / "08_plv",
    'time_frequency': PROCESSED_DATA_DIR / "09_time_frequency",
    'statistics': PROCESSED_DATA_DIR / "10_statistics",
}

# Create all output directories
for dir_path in OUTPUT_DIRS.values():
    dir_path.mkdir(parents=True, exist_ok=True)

# =============================================================================
# SAVING FORMAT
# =============================================================================

SAVE_FORMAT = 'fif'  # MNE native format
OVERWRITE = True     # Overwrite existing files

# =============================================================================
# PRINT CONFIGURATION (when run directly)
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("EEG HYPERSCANNING PIPELINE CONFIGURATION")
    print("=" * 60)
    
    print("\n--- Directories ---")
    print(f"BASE_DIR:           {BASE_DIR}")
    print(f"RAW_DATA_DIR:       {RAW_DATA_DIR}")
    print(f"  Exists:           {RAW_DATA_DIR.exists()}")
    print(f"BIDS_ROOT:          {BIDS_ROOT}")
    print(f"PROCESSED_DATA_DIR: {PROCESSED_DATA_DIR}")
    print(f"  Exists:           {PROCESSED_DATA_DIR.exists()}")
    
    print("\n--- Subjects/Dyads ---")
    print(f"Number of subjects: {len(SUBJECTS)}")
    print(f"Subjects:           {SUBJECTS}")
    
    print("\n--- Task ---")
    print(f"Task name:          {TASK}")
    
    print("\n--- Preprocessing ---")
    print(f"Filter:             {FREQ_LOWER} - {FREQ_UPPER} Hz")
    print(f"Downsample to:      {DOWNSAMPLE_SFREQ} Hz")
    print(f"Epoch window:       {EPOCH_TMIN} to {EPOCH_TMAX} s")
    print(f"ICA components:     {ICA_N_COMPONENTS}")
    print(f"Reference:          {REFERENCE}")
    
    print("\n--- Analysis ---")
    print(f"Frequency bands:    {list(FREQ_BANDS.keys())}")
    print(f"ERP windows:        {list(ERP_WINDOWS.keys())}")
    print(f"Permutations:       {N_PERMUTATIONS}")
    
    print("\n--- Output Directories ---")
    for name, path in OUTPUT_DIRS.items():
        exists = "✓" if path.exists() else "✗"
        n_files = len(list(path.glob("*"))) if path.exists() else 0
        print(f"  {name:15} {exists} ({n_files} files) {path}")
    
    # Check raw data structure
    print("\n--- Raw Data Check ---")
    if RAW_DATA_DIR.exists():
        sub_dirs = [d.name for d in RAW_DATA_DIR.iterdir() if d.is_dir() and d.name.startswith("sub-")]
        print(f"  Found {len(sub_dirs)} subject folders")
        if sub_dirs:
            # Check first subject structure
            first_sub = RAW_DATA_DIR / sub_dirs[0]
            eeg_dir = first_sub / "eeg"
            if eeg_dir.exists():
                eeg_files = list(eeg_dir.glob("*"))
                print(f"  First subject ({sub_dirs[0]}) has {len(eeg_files)} files in eeg/")
                for f in eeg_files[:5]:
                    print(f"    - {f.name}")
            else:
                print(f"  Warning: No 'eeg' folder in {first_sub}")
    else:
        print(f"  Warning: RAW_DATA_DIR does not exist!")
    
    print("\n" + "=" * 60)