"""
Central path configuration for the EEG Bala Sharks project.

Edit INPUT_DIR and OUTPUT_DIR directly, or override via environment variables:
  EEG_INPUT_DIR  – BIDS root folder containing sub-* subject folders
  EEG_OUTPUT_DIR – base output folder (each pipeline creates its own subfolder here)
"""

import os
from pathlib import Path

_BASE = Path(__file__).resolve().parent


# Edit the defaults below to match your system.

# BIDS root – the folder that contains the input data
# e.g. for data on a USB stick: Path(r"E:\ds006761")  or  Path(r"D:\my_data\ds006761")
INPUT_DIR: Path = Path(os.environ.get("EEG_INPUT_DIR", str(_BASE / "MNE-sample-data" / "ds006761")))

# Base output folder – every pipeline component writes into a subfolder here.
# e.g.  output/preprocessing/, output/author_code/, output/preprocessing_authors/
OUTPUT_DIR: Path = Path(os.environ.get("EEG_OUTPUT_DIR", str(_BASE / "output")))
# OUTPUT_DIR: Path = Path(r"D:/uni/rock_paper_scissors/output")

# BioSemi 64-channel layout file used by the author_code scripts.
BIOSEMI64_MAT: Path = Path(os.environ.get("EEG_BIOSEMI64_MAT", str(_BASE / "biosemi64.mat")))


# The subfolders below are created automatically – you normally do not need
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
