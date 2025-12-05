# utils.py
import mne
import pickle
from pathlib import Path

def save_raw(raw, filename):
    raw.save(filename, overwrite=True)

def load_raw(filename, preload=True):
    return mne.io.read_raw_fif(filename, preload=preload)

def save_object(obj, filename):
    with open(filename, "wb") as f:
        pickle.dump(obj, f)

def load_object(filename):
    import pickle
    with open(filename, "rb") as f:
        return pickle.load(f)
