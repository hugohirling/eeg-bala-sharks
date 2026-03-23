# config.py
import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent.parent  # EEG_Bala Sharks
BIDS_ROOT = BASE_DIR / "MNE-sample-data" / "ds006761"
OUTPUT_DIR = BASE_DIR / "data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
BIOSEMI64_MAT_PATH = BASE_DIR / "biosemi64.mat"

# Subjects
SUBJECTS = ["01"]

# Filter
FREQ_LOWER = 1.0
FREQ_UPPER = 40.0

#Downsampling
DOWNSAMPLE_SFREQ = 200

# Player labels in the raw file are swapped relative to behavioral metadata.
# P1 uses channels prefixed with "2-", P2 uses channels prefixed with "1-".
PLAYER_PREFIX_MAP = {
    "P1": "2-",
    "P2": "1-",
}

# Optional TSV with bad-channel annotations.
# Leave as None to skip manual interpolation lists.
BAD_CHANNELS_FILE = None

# Automatic bad-channel detection (robust z-score on channel STD).
BAD_CHANNEL_ZSCORE_THRESHOLD = 4.0
BAD_CHANNEL_FLAT_STD_THRESHOLD = 1e-12
QC_DIR = OUTPUT_DIR / "qc"
QC_DIR.mkdir(parents=True, exist_ok=True)

# Epoching
MAX_EPOCHS = 500
EPOCH_TMIN = -0.2
EPOCH_TMAX = 5.0
EPOCH_BASELINE_MIN = -0.2
EPOCH_BASELINE_MAX = 0.0

DECISION_TMIN = -0.2
DECISION_TMAX = 2.0
RESPONSE_TMIN = 1.8
RESPONSE_TMAX = 4.0
FEEDBACK_TMIN = 3.8
FEEDBACK_TMAX = 5.0

# AutoReject config
AR_N_JOBS = 1       # Number of parallel jobs
AR_VERBOSE = True   # Verbose output

# ICA
ICA_N_COMPONENTS = 20
ICA_MAX_ITER = 500

# Saving format
SAVE_FORMAT = "fif"  # MNE Standard

# Interbrain Synchrony parameters
IBS_FMIN = 8     # Alpha band (RPS: attention / anticipation)
IBS_FMAX = 12

channel_labels = {
    "A1": "Fp1",
    "A2": "AF7",
    "A3": "AF3",
    "A4": "F1",
    "A5": "F3",
    "A6": "F5",
    "A7": "F7",
    "A8": "FT7",
    "A9": "FC5",
    "A10": "FC3",
    "A11": "FC1",
    "A12": "C1",
    "A13": "C3",
    "A14": "C5",
    "A15": "T7",
    "A16": "TP7",
    "A17": "CP5",
    "A18": "CP3",
    "A19": "CP1",
    "A20": "P1",
    "A21": "P3",
    "A22": "P5",
    "A23": "P7",
    "A24": "P9",
    "A25": "PO7",
    "A26": "PO3",
    "A27": "O1",
    "A28": "Iz",
    "A29": "Oz",
    "A30": "POz",
    "A31": "Pz",
    "A32": "CPz",
    "B1": "Fpz",
    "B2": "Fp2",
    "B3": "AF8",
    "B4": "AF4",
    "B5": "AFz",
    "B6": "Fz",
    "B7": "F2",
    "B8": "F4",
    "B9": "F6",
    "B10": "F8",
    "B11": "FT8",
    "B12": "FC6",
    "B13": "FC4",
    "B14": "FC2",
    "B15": "FCz",
    "B16": "Cz",
    "B17": "C2",
    "B18": "C4",
    "B19": "C6",
    "B20": "T8",
    "B21": "TP8",
    "B22": "CP6",
    "B23": "CP4",
    "B24": "CP2",
    "B25": "P2",
    "B26": "P4",
    "B27": "P6",
    "B28": "P8",
    "B29": "P10",
    "B30": "PO8",
    "B31": "PO4",
    "B32": "O2"
}

channel_types = {
    "Erg1": "eog",
    "Erg2": "eog",
    "Resp": "resp",
    "Plet": "misc",
    "Temp": "temperature",
    "GSR1": "gsr",
    "GSR2": "gsr"
}