from pathlib import Path
from utils import load_raw, save_raw

def filter_data(raw, l_freq, h_freq):
    """
    Apply bandpass filter to raw data.
    """
    print(f"Applying bandpass filter: {l_freq} - {h_freq} Hz")
    raw.filter(l_freq=l_freq, h_freq=h_freq)
    return raw

def process_subject(path_in, path_out):
    """
    Load raw EEG, filter channels, and save annotated raw file.
    """
    raw = load_raw(path_in, preload=True)

    # Run bandpass filter
    raw = filter_data(raw, l_freq=config.FREQ_LOWER, h_freq=config.FREQ_UPPER)

    # Save annotated raw file
    save_raw(raw, path_out)

    print(f"Saved filtered-channels file to: {path_out}")

if __name__ == "__main__":
    import config
    for subj in config.SUBJECTS:
        p1 = Path(config.OUTPUT_DIR) / f"sub-{subj}_P1_raw.fif"
        p2 = Path(config.OUTPUT_DIR) / f"sub-{subj}_P2_raw.fif"

        out_p1 = Path(config.OUTPUT_DIR) / f"sub-{subj}_P1_filtered.fif"
        out_p2 = Path(config.OUTPUT_DIR) / f"sub-{subj}_P2_filtered.fif"

        process_subject(p1, out_p1)
        process_subject(p2, out_p2)