import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent)) 

# 04_epoching.py
import mne
from utils import load_raw, save_raw
import config

def epoch_data(subject_id):
    raw_file = config.OUTPUT_DIR / f"raw_ref_{subject_id}.fif"
    raw = load_raw(raw_file)

    # Short Epochs
    events = mne.make_fixed_length_events(raw, duration=config.EPOCH_DURATION)
    tmax = config.EPOCH_DURATION - 1.0 / raw.info['sfreq']
    epochs = mne.Epochs(raw, events, event_id=1, tmin=0.0, tmax=tmax, baseline=None, preload=True)
    epochs.resample(250)
    epochs.filter(config.FREQ_LOWER, config.FREQ_UPPER, n_jobs=1)

    out_file = config.OUTPUT_DIR / f"epochs_{subject_id}.fif"
    save_raw(epochs, out_file)
    print(f"[04_epoching] Saved epochs for {subject_id} -> {out_file}")
    return out_file

if __name__ == "__main__":
    for subj in config.SUBJECTS:
        epoch_data(subj)
