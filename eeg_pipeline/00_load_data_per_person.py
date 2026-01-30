# 01_load_data.py
from mne_bids import BIDSPath, read_raw_bids
import config
from utils import save_raw
import mne

def existing(raw, chs):
    """Return only channels that actually exist in the raw object."""
    return [ch for ch in chs if ch in raw.ch_names]

def load_and_split(subject_id):
    print(f"\n=== Loading subject {subject_id} ===")

    # --- Load raw data from BIDS ---
    bids_path = BIDSPath(
        subject=subject_id,
        task="RPS",
        datatype="eeg",
        suffix="eeg",
        root=config.BIDS_ROOT,
    )

    raw = read_raw_bids(bids_path, verbose=False)

    # --- Define channel groups (EEG dynamic, others guarded) ---

    # EEG (dynamic, safe)
    eeg_chs_p1 = [ch for ch in raw.ch_names if ch.startswith(("1-A", "1-B"))]
    eeg_chs_p2 = [ch for ch in raw.ch_names if ch.startswith(("2-A", "2-B"))]

    # EOG
    eog_chs_p1 = existing(raw, ["1-Erg1", "1-Erg2"])
    eog_chs_p2 = existing(raw, ["2-Erg1", "2-Erg2"])

    # Respiration
    resp_chs_p1 = existing(raw, ["1-Resp"])
    resp_chs_p2 = existing(raw, ["2-Resp"])

    # Bio / Pleth
    bio_chs_p1 = existing(raw, ["1-Plet"])
    bio_chs_p2 = existing(raw, ["2-Plet"])

    # Temperature
    temp_chs_p1 = existing(raw, ["1-Temp"])
    temp_chs_p2 = existing(raw, ["2-Temp"])

    # Stim
    stim_chs = existing(raw, ["Status"])

    # --- Optional: log what was found ---
    print("P1 channels:",
          eeg_chs_p1, eog_chs_p1, resp_chs_p1, bio_chs_p1, temp_chs_p1, stim_chs)
    print("P2 channels:",
          eeg_chs_p2, eog_chs_p2, resp_chs_p2, bio_chs_p2, temp_chs_p2, stim_chs)

    # --- Split raw per person ---
    raw_p1 = raw.copy().pick_channels(
        eeg_chs_p1
        + eog_chs_p1
        + resp_chs_p1
        + bio_chs_p1
        + temp_chs_p1
        + stim_chs
    )

    raw_p2 = raw.copy().pick_channels(
        eeg_chs_p2
        + eog_chs_p2
        + resp_chs_p2
        + bio_chs_p2
        + temp_chs_p2
        + stim_chs
    )

    # --- Set channel types (only existing channels) ---
    raw_p1.set_channel_types({ch: "eeg" for ch in eeg_chs_p1})
    raw_p1.set_channel_types({ch: "eog" for ch in eog_chs_p1})
    raw_p1.set_channel_types({ch: "resp" for ch in resp_chs_p1})
    raw_p1.set_channel_types({ch: "bio" for ch in bio_chs_p1})
    raw_p1.set_channel_types({ch: "temperature" for ch in temp_chs_p1})
    raw_p1.set_channel_types({ch: "stim" for ch in stim_chs})

    raw_p2.set_channel_types({ch: "eeg" for ch in eeg_chs_p2})
    raw_p2.set_channel_types({ch: "eog" for ch in eog_chs_p2})
    raw_p2.set_channel_types({ch: "resp" for ch in resp_chs_p2})
    raw_p2.set_channel_types({ch: "bio" for ch in bio_chs_p2})
    raw_p2.set_channel_types({ch: "temperature" for ch in temp_chs_p2})
    raw_p2.set_channel_types({ch: "stim" for ch in stim_chs})

    # --- Save outputs ---
    out_p1 = config.OUTPUT_DIR / f"sub-{subject_id}_P1_raw.fif"
    out_p2 = config.OUTPUT_DIR / f"sub-{subject_id}_P2_raw.fif"

    save_raw(raw_p1, out_p1)
    save_raw(raw_p2, out_p2)

    print(f"Saved Person 1: {out_p1}")
    print(f"Saved Person 2: {out_p2}")

    return out_p1, out_p2


if __name__ == "__main__":
    for subj in config.SUBJECTS:
        load_and_split(subj)
