import openneuro
from mne.datasets import sample
import os
this_dir = os.path.dirname(os.path.abspath(__file__))

dataset = "ds006761"

bids_root = sample.data_path(this_dir) / dataset
bids_root.mkdir(parents=True, exist_ok=True)
openneuro.download(dataset=dataset, target_dir=bids_root)
