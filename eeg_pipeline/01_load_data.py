from utils import save_raw, load_raw 

# 01_load_data.py
from mne_bids import BIDSPath, read_raw_bids
import mne
import config
from utils import save_raw

def load_data(subject_id):
    bids_path = BIDSPath(subject=subject_id, task="RPS", datatype='eeg', suffix='eeg', root=config.BIDS_ROOT)
    raw = read_raw_bids(bids_path)

    # Define channel types
    eeg_channels = [ch for ch in raw.ch_names if ch.startswith(("1-A","1-B","2-A","2-B"))]
    eog_channels = ['1-Erg1', '1-Erg2', '2-Erg1', '2-Erg2']
    resp_channels = ['1-Resp', '2-Resp']
    bio_channels = ['1-Plet', '2-Plet']
    temp_channels = ['1-Temp', '2-Temp']
    stim_channels = ['Status']

    raw.set_channel_types({ch: 'eeg' for ch in eeg_channels})
    raw.set_channel_types({ch: 'eog' for ch in eog_channels})
    raw.set_channel_types({ch: 'resp' for ch in resp_channels})
    raw.set_channel_types({ch: 'bio' for ch in bio_channels})
    raw.set_channel_types({ch: 'temperature' for ch in temp_channels})
    raw.set_channel_types({ch: 'stim' for ch in stim_channels})

    # Saving
    out_file = config.OUTPUT_DIR / f"sub-{subject_id}_task-RPS_raw.fif"
    save_raw(raw, out_file)
    print(f"[01_load_data] Saved raw data for subject {subject_id} to {out_file}")
    return out_file

if __name__ == "__main__":
    for subj in config.SUBJECTS:
        load_data(subj)
