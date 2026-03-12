import mne
from pathlib import Path
import config

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

def inspect_and_rename(subject_id, person):
    """
    Load raw data, rename EEG channels from BioSemi64 to 10-20,
    keep EOG/physio channels intact, set montage, and save.
    """
    raw_path = config.OUTPUT_DIR / f"sub-{subject_id}_{person}_raw.fif"
    print(f"Loading: {raw_path}")

    raw = mne.io.read_raw_fif(raw_path, preload=True)

    # ---- Identify EEG channels only ----
    eeg_picks = mne.pick_types(raw.info, eeg=True)
    eeg_ch_names = [raw.ch_names[i] for i in eeg_picks]

    # ---- Remove subject prefix ONLY from EEG ----
    prefix = "1-" if person == "P1" else "2-"
    mapping_prefix = {
        ch: ch[len(prefix):]
        for ch in eeg_ch_names
        if ch.startswith(prefix)
    }
    if mapping_prefix:
        raw.rename_channels(mapping_prefix)

    # ---- Apply BioSemi → 10–20 mapping (EEG ONLY) ----
    mapping_1020 = {
        ch: BIOS64_TO_1020[ch]
        for ch in raw.ch_names
        if ch in BIOS64_TO_1020
    }
    if mapping_1020:
        raw.rename_channels(mapping_1020)

    # ---- Set montage (EEG only, others ignored safely) ----
    montage = mne.channels.make_standard_montage("standard_1020")
    raw.set_montage(montage, match_case=False)

    # ---- Save FULL raw (EEG + EOG + physio + stim) ----
    out_path = config.OUTPUT_DIR / f"sub-{subject_id}_{person}_renamed_raw.fif"
    print(f"Saving renamed/montaged file to: {out_path}")
    raw.save(out_path, overwrite=True)

    return out_path


if __name__ == "__main__":
    for subj in config.SUBJECTS:
        for person in ["P1", "P2"]:
            inspect_and_rename(subj, person)
