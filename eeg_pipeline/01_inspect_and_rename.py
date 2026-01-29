import mne
from pathlib import Path
import config
import warnings
warnings.filterwarnings('ignore')

# BioSemi64 → Standard 10-20 Mapping
BIOS64_TO_1020 = {
    # A row
    "A1": "Fp1", "A2": "AF7", "A3": "AF3", "A4": "F1", "A5": "F3",
    "A6": "F5", "A7": "F7", "A8": "FT7", "A9": "FC5", "A10": "FC3",
    "A11": "FC1", "A12": "C1", "A13": "C3", "A14": "C5", "A15": "T7",
    "A16": "TP7", "A17": "CP5", "A18": "CP3", "A19": "CP1", "A20": "P1",
    "A21": "P3", "A22": "P5", "A23": "P7", "A24": "P9", "A25": "PO7",
    "A26": "PO3", "A27": "O1", "A28": "Iz", "A29": "Oz", "A30": "POz",
    "A31": "Pz", "A32": "CPz",
    # B row
    "B1": "Fpz", "B2": "Fp2", "B3": "AF8", "B4": "AF4", "B5": "AFz",
    "B6": "Fz", "B7": "F2", "B8": "F4", "B9": "F6", "B10": "F8",
    "B11": "FT8", "B12": "FC6", "B13": "FC4", "B14": "FC2", "B15": "FCz",
    "B16": "Cz", "B17": "C2", "B18": "C4", "B19": "C6", "B20": "T8",
    "B21": "TP8", "B22": "CP6", "B23": "CP4", "B24": "CP2", "B25": "P2",
    "B26": "P4", "B27": "P6", "B28": "P8", "B29": "P10", "B30": "PO8",
    "B31": "PO4", "B32": "O2",
}

STANDARD_1020_NAMES = list(BIOS64_TO_1020.values())


def inspect_and_rename(subj, person):
    """Inspect raw data and rename channels to standard 10-20."""
    
    in_path = config.PROCESSED_DATA_DIR / f"sub-{subj}_{person}_raw.fif"
    out_path = config.PROCESSED_DATA_DIR / f"sub-{subj}_{person}_renamed_raw.fif"
    
    if not in_path.exists():
        print(f"  File not found: {in_path}")
        return False
    
    print(f"Loading: {in_path}")
    raw = mne.io.read_raw_fif(in_path, preload=False, verbose=False)
    
    # Get prefix for this player (e.g., "1-" or "2-")
    prefix = "1-" if person == "P1" else "2-"
    
    # Build rename mapping for this player's channels
    rename_map = {}
    for old_name, new_name in BIOS64_TO_1020.items():
        full_old = f"{prefix}{old_name}"
        if full_old in raw.ch_names:
            rename_map[full_old] = new_name
    
    if rename_map:
        print(f"  Renaming {len(rename_map)} EEG channels...")
        raw.rename_channels(rename_map)
    
    # Now identify EEG vs non-EEG channels after renaming
    eeg_channels = [ch for ch in raw.ch_names if ch in STANDARD_1020_NAMES]
    non_eeg_channels = [ch for ch in raw.ch_names if ch not in STANDARD_1020_NAMES]
    
    # Set channel types explicitly
    print(f"  Setting {len(eeg_channels)} channels as EEG type")
    raw.set_channel_types({ch: 'eeg' for ch in eeg_channels})
    
    if non_eeg_channels:
        print(f"  Setting {len(non_eeg_channels)} channels as misc type")
        # Handle stim channel separately
        for ch in non_eeg_channels:
            if 'Status' in ch or 'STI' in ch:
                raw.set_channel_types({ch: 'stim'})
            else:
                raw.set_channel_types({ch: 'misc'})
    
    # Set montage for EEG channels
    if eeg_channels:
        montage = mne.channels.make_standard_montage('standard_1020')
        raw.set_montage(montage, match_case=False, on_missing='ignore')
        print(f"  Montage set for {len(eeg_channels)} EEG channels")
    
    # Load data for saving
    print(f"  Loading data into memory...")
    raw.load_data()
    
    # Save
    raw.save(out_path, overwrite=True)
    print(f"  Saved: {out_path}")
    
    return True


if __name__ == "__main__":
    print("="*60)
    print("STEP 01: INSPECT AND RENAME CHANNELS")
    print("="*60)
    
    success_count = 0
    fail_count = 0
    
    for subj in config.SUBJECTS:
        for person in ["P1", "P2"]:
            try:
                if inspect_and_rename(subj, person):
                    success_count += 1
                else:
                    fail_count += 1
            except Exception as e:
                print(f"  Error for {subj} {person}: {e}")
                fail_count += 1
    
    print("\n" + "="*60)
    print(f"STEP 01 COMPLETE: {success_count} succeeded, {fail_count} failed")
    print("="*60)