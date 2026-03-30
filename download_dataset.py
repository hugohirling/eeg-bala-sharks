import openneuro
from mne.datasets import sample
import os
this_dir = os.path.dirname(os.path.abspath(__file__))

dataset = "ds006761"

bids_root = sample.data_path(this_dir) / dataset
bids_root.mkdir(parents=True, exist_ok=True)


TRIAL_MODE = True  # Set to False when you are ready to download everything

if TRIAL_MODE:
    print("Trial mode active: Downloading only 3 subjects...")
    # List the specific subjects and essential BIDS metadata files
    files_to_download = [
        "sub-01",
        "sub-02",
        "sub-03",
        "dataset_description.json",
        "participants.tsv"
    ]
    openneuro.download(dataset=dataset, target_dir=bids_root, include=files_to_download)
else:
    print("Full mode active: Downloading the entire dataset...")
    openneuro.download(dataset=dataset, target_dir=bids_root)