from pathlib import Path
from utils import load_raw, save_raw

def down_sample_data(raw, target_sfreq=512):
    """
    Apply down-sampling to raw data.
    """
    print(f"Down-sampling data to {target_sfreq} Hz")
    raw.resample(sfreq=target_sfreq)
    return raw

def process_subject(path_in, path_out):
    """
    Load raw EEG, filter channels, and save annotated raw file.
    """
    raw = load_raw(path_in, preload=True)

    # Run down-sampling
    raw = down_sample_data(raw, 512)

    # Save annotated raw file
    save_raw(raw, path_out)

    print(f"Saved down-sampled-channels file to: {path_out}")

if __name__ == "__main__":
    import config
    for subj in config.SUBJECTS:
        p1 = Path(config.OUTPUT_DIR) / f"sub-{subj}_P1_raw.fif"
        p2 = Path(config.OUTPUT_DIR) / f"sub-{subj}_P2_raw.fif"

        out_p1 = Path(config.OUTPUT_DIR) / f"sub-{subj}_P1_downsampled.fif"
        out_p2 = Path(config.OUTPUT_DIR) / f"sub-{subj}_P2_downsampled.fif"

        process_subject(p1, out_p1)
        process_subject(p2, out_p2)