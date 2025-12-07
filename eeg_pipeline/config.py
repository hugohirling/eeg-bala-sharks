# config.py
import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent.parent  # EEG_Bala Sharks
BIDS_ROOT = BASE_DIR / "MNE-sample-data" / "ds006761"
OUTPUT_DIR = BASE_DIR / "data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Subjects
SUBJECTS = ["01"]

# Filter
FREQ_LOWER = 1.0
FREQ_UPPER = 40.0

# Epoching
EPOCH_DURATION = 1.0
MAX_EPOCHS = 500

# AutoReject config
AR_N_JOBS = 1       # Number of parallel jobs
AR_VERBOSE = True   # Verbose output
MAX_EPOCHS = 500    # Max number of epochs AutoReject uses
EPOCH_DURATION = 2  # seconds
FREQ_LOWER = 1.0    # Hz
FREQ_UPPER = 40.0   # Hz

# ICA
ICA_N_COMPONENTS = 20
ICA_MAX_ITER = 500

# Saving format
SAVE_FORMAT = "fif"  # MNE Standard
