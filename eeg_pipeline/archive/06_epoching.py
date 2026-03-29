from pathlib import Path
from utils import load_raw, save_raw
import mne

def create_epochs(raw, tmin, tmax):

    # --- Extract events from stimulus channel ---
    events, event_id = mne.events_from_annotations(raw)

    # --- Create epochs ---
    epochs = mne.Epochs(
        raw,
        events=events,
        event_id=event_id,
        tmin=tmin,
        tmax=tmax,
        preload=True
    )
    return epochs

def process_subject(path_in, path_out):
    """
    Load raw EEG, filter channels, and save annotated raw file.
    """
    raw = load_raw(path_in, preload=True)

    # Run bandpass filter
    raw = create_epochs(raw, config.EPOCH_TMIN, config.EPOCH_TMAX)

    # Save annotated raw file
    save_raw(raw, path_out)

    print(f"Saved epoches file to: {path_out}")

if __name__ == "__main__":
    from preprocessing import config
    for subj in config.SUBJECTS:
        p1 = Path(config.OUTPUT_DIR) / f"sub-{subj}_P1_downsampled.fif"
        p2 = Path(config.OUTPUT_DIR) / f"sub-{subj}_P2_downsampled.fif"

        out_p1 = Path(config.OUTPUT_DIR) / f"sub-{subj}_P1_epoch.fif"
        out_p2 = Path(config.OUTPUT_DIR) / f"sub-{subj}_P2_epoch.fif"

        process_subject(p1, out_p1)
        process_subject(p2, out_p2)
