# 04_re_reference_common_average.py
import mne
from utils import load_raw, save_raw
import config
from pathlib import Path

def rereference_subject(file_path: Path):
    """
    Re-reference a Raw EEG file to the Common Average (CAR) of all EEG channels.

    Justification:
    - No mastoid reference: BioSemi Active 2 does not have dedicated mastoid channels.
    - Cz reference: Cz (B16) might contain artifacts and using a single channel as reference
      could propagate noise to all other channels.
    - Subset reference: Using only central channels may overemphasize or bias the reference
      if those channels are contaminated. 
      
    -> CAR uses all EEG channels to distribute the reference evenly and robustly.
    """
    file_path = Path(file_path)

    if not file_path.exists():
        print(f"File not found: {file_path}")
        return

    print(f"\nProcessing {file_path.name}...")

    # --- Load raw data ---
    raw = load_raw(file_path)

    # --- Re-reference to Common Average ---
    eeg_channels = mne.pick_types(raw.info, eeg=True, exclude='bads')
    if len(eeg_channels) == 0:
        print("No EEG channels found, skipping re-referencing.")
    else:
        raw.set_eeg_reference(ref_channels='average')
        print("Re-referenced using Common Average Reference (CAR).")

    # --- Save output ---
    out_file = file_path.with_name(file_path.stem + '_CAR_raw.fif')
    save_raw(raw, out_file)
    print(f"Saved re-referenced data to {out_file}")
    return out_file


def rereference_all_subjects():
    """
    Iterate over all subjects and both persons (P1, P2), re-referencing each file.
    """
    for subj in config.SUBJECTS:
        for person in ['P1', 'P2']:
            file_path = config.OUTPUT_DIR / f"sub-{subj}_{person}_renamed_raw.fif"
            rereference_subject(file_path)


if __name__ == "__main__":
    rereference_all_subjects()
