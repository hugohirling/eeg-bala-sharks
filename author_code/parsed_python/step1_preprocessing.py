"""
Pre-processing script (Python version of step1_preprocessing.m):
   - Plot the data so we can identify noisy channels
   - Interpolate noisy channels
   - Down-sample to 256 Hz
   - Save

This script mirrors the MATLAB ordering as closely as practical in MNE:
   epoch -> optional plot-only filtering -> interpolate -> downsample -> save
"""

import os
import warnings
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from scipy.io import loadmat, savemat

warnings.filterwarnings("ignore")

# Central path configuration
import sys
BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
import paths as _project_paths  # noqa: E402

path_to_data = str(_project_paths.INPUT_DIR)
OUTPUT_ROOT = _project_paths.OUTPUT_DIR / "author_code"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

# Set parameters
identify_bad_channels = False
interpolate_bad_channels = True
num_trials = 480
DEFAULT_PAIR_IDS = np.concatenate([np.arange(1, 10), np.arange(11, 23), np.arange(25, 35)])
FS = 2048


def _resolve_pair_ids(default_ids):
    raw = os.environ.get("RPS_SUBJECTS") or os.environ.get("EEG_SUBJECTS")
    if not raw:
        return default_ids
    parsed = []
    for token in raw.split(","):
        token = token.strip().lower().replace("sub-", "")
        if not token:
            continue
        try:
            parsed.append(int(token))
        except ValueError:
            print(f"Warning: invalid subject token ignored: {token}")
    if not parsed:
        return default_ids
    return np.array([sid for sid in parsed if sid in set(default_ids)], dtype=int)


pair_ids = _resolve_pair_ids(DEFAULT_PAIR_IDS)
num_pairs = len(pair_ids)
FORCE_REPROCESS = (os.environ.get("RPS_FORCE_REPROCESS", "0").strip().lower() in {"1", "true", "yes", "on"})

participants_path = os.path.join(path_to_data, "participants.tsv")
if os.path.exists(participants_path):
    participants = pd.read_csv(participants_path, sep="\t")
else:
    print(f"Warning: participants.tsv missing: {participants_path}. Bad-channel lookup disabled.")
    participants = pd.DataFrame(columns=["participant_id"])

# Standard BioSemi-64 labels used in the MATLAB pipeline.
ch_names_biosemi = [
    "Fp1", "AF7", "AF3", "F1", "F3", "F5", "F7", "FT7", "FC5", "FC3", "FC1", "C1", "C3", "C5", "T7", "TP7",
    "CP5", "CP3", "CP1", "P1", "P3", "P5", "P7", "P9", "PO7", "PO3", "O1", "Iz", "Oz", "POz", "Pz", "CPz",
    "Fpz", "Fp2", "AF8", "AF4", "AFz", "Fz", "F2", "F4", "F6", "F8", "FT8", "FC6", "FC4", "FC2", "FCz", "C2",
    "C4", "C6", "T8", "TP8", "CP6", "CP4", "CP2", "Cz", "P2", "P4", "P6", "P8", "P10", "PO8", "PO4", "O2",
]


def _load_biosemi_positions():
    candidates = [
        str(_project_paths.BIOSEMI64_MAT),
        os.path.join(path_to_data, "biosemi64.mat"),
        os.path.join("author_code", "helper_files", "biosemi64.mat"),
        "biosemi64.mat",
    ]
    for cand in candidates:
        if os.path.exists(cand):
            mat_data = loadmat(cand)
            if "biosemi64" in mat_data:
                return mat_data["biosemi64"]
    raise FileNotFoundError("Could not find biosemi64.mat in expected locations.")


biosemi64 = _load_biosemi_positions()
montage = mne.channels.make_dig_montage(ch_pos=dict(zip(ch_names_biosemi, biosemi64)), coord_frame="head")


def _participant_bad_channels(pair, ppt):
    row = participants[participants["participant_id"] == f"sub-{pair:02d}"]
    if row.empty:
        return []

    # MATLAB selects columns [7,12] then picks column by player.
    # Use those positions when present, fallback to named columns.
    bad_str = ""
    try:
        idx = 6 if ppt == 1 else 11
        if idx < len(row.columns):
            bad_str = row.iloc[0, idx]
    except Exception:
        bad_str = ""

    if (not isinstance(bad_str, str)) or (bad_str.strip() == ""):
        fallback_cols = ["player1_pre_processing_channels_fixed", "player2_pre_processing_channels_fixed"]
        if ppt == 1 and fallback_cols[0] in row.columns:
            bad_str = row.iloc[0][fallback_cols[0]]
        if ppt == 2 and fallback_cols[1] in row.columns:
            bad_str = row.iloc[0][fallback_cols[1]]

    if not isinstance(bad_str, str) or bad_str.strip() == "":
        return []

    return [c.strip() for c in bad_str.split(",") if c.strip()]


def _player_outputs_exist(pair, ppt):
    out_fif = OUTPUT_ROOT / f"pair-{pair:02d}_player-{ppt}_task-RPS_eeg-epo.fif"
    out_mat = OUTPUT_ROOT / f"pair-{pair:02d}_player-{ppt}_task-RPS_eeg.mat"
    return out_fif.exists() and out_mat.exists()


for p, pair in enumerate(pair_ids):
    print(f"Loading pair {p + 1} of {num_pairs}: {pair}")

    players_to_run = []
    for ppt in (1, 2):
        if (not FORCE_REPROCESS) and _player_outputs_exist(pair, ppt):
            print(f"Info: skipping sub-{pair:02d} player-{ppt}, outputs already exist.")
            continue
        players_to_run.append(ppt)

    if not players_to_run:
        print(f"Info: skipping sub-{pair:02d}, all player outputs already exist.")
        continue

    events_filename = os.path.join(path_to_data, f"sub-{pair:02d}", "eeg", f"sub-{pair:02d}_task-RPS_events.tsv")
    if not os.path.exists(events_filename):
        print(f"Warning: skipping sub-{pair:02d}, events missing: {events_filename}")
        continue
    try:
        events = pd.read_csv(events_filename, sep="\t")
    except Exception as exc:
        print(f"Warning: skipping sub-{pair:02d}, events unreadable: {exc}")
        continue
    stimonsample = events["onset_sample"].to_numpy(dtype=int)

    prestim = 0.2
    poststim = 5.0

    raw_filename = os.path.join(path_to_data, f"sub-{pair:02d}", "eeg", f"sub-{pair:02d}_task-RPS_eeg.bdf")
    if not os.path.exists(raw_filename):
        print(f"Warning: skipping sub-{pair:02d}, BDF missing: {raw_filename}")
        continue
    # Keep raw lazy to avoid allocating multi-GB arrays for full recordings.
    try:
        raw = mne.io.read_raw_bdf(raw_filename, preload=False)
    except Exception as exc:
        print(f"Warning: skipping sub-{pair:02d}, BDF read failed: {exc}")
        continue

    # Epoch definitions from event sample indices.
    events_array = np.column_stack(
        [
            stimonsample,
            np.zeros(stimonsample.shape[0], dtype=int),
            np.ones(stimonsample.shape[0], dtype=int),
        ]
    )

    for ppt in players_to_run:
        # MATLAB player-channel mapping (swapped relative to behavioral labels).
        if ppt == 1:
            wanted = [ch for ch in raw.ch_names if ("2-A" in ch or "2-B" in ch)]
        else:
            wanted = [ch for ch in raw.ch_names if ("1-A" in ch or "1-B" in ch)]

        if len(wanted) < 64:
            print(f"Pair {pair}, Player {ppt}: expected >=64 channels, found {len(wanted)}. Skipping.")
            continue

        raw_ppt = raw.copy().pick(wanted)

        # Rename first 64 channels to BioSemi labels and keep those 64.
        rename_dict = {raw_ppt.ch_names[i]: ch_names_biosemi[i] for i in range(64)}
        raw_ppt.rename_channels(rename_dict)
        raw_ppt.pick(ch_names_biosemi)
        raw_ppt.set_montage(montage)

        epochs = mne.Epochs(
            raw_ppt,
            events_array,
            event_id={"trial_start": 1},
            tmin=-prestim,
            tmax=poststim,
            baseline=None,
            preload=True,
            reject_by_annotation=False,
        )

        if identify_bad_channels:
            # Plot-only filter (no filtering in saved data).
            epochs_f = epochs.copy().filter(
                l_freq=0.1,
                h_freq=100.0,
                method="iir",
                iir_params={"order": 4, "ftype": "butter"},
            )
            epochs_f.plot(n_epochs=min(20, len(epochs_f)), n_channels=64, scalings="auto", title=f"Pair {pair}, Player {ppt}")

        if interpolate_bad_channels:
            bad_chans = _participant_bad_channels(pair, ppt)
            if bad_chans:
                valid_bads = [ch for ch in bad_chans if ch in epochs.ch_names]
                if valid_bads:
                    epochs.info["bads"] = valid_bads
                    epochs.interpolate_bads(reset_bads=True)
                    print(f"pair {pair}, player {ppt}: fixed {', '.join(valid_bads)}")

            # Downsample once at end, as in MATLAB workflow.
            epochs_ds = epochs.copy().resample(256)

            # Save FIF (used by parsed_python step2a).
            out_fif = str(OUTPUT_ROOT / f"pair-{pair:02d}_player-{ppt}_task-RPS_eeg-epo.fif")
            os.makedirs(os.path.dirname(out_fif), exist_ok=True)
            epochs_ds.save(out_fif, overwrite=True)

            # Save lightweight MAT for easier MATLAB-side checks.
            out_mat = str(OUTPUT_ROOT / f"pair-{pair:02d}_player-{ppt}_task-RPS_eeg.mat")
            savemat(
                out_mat,
                {
                    "eeg_data": epochs_ds.get_data(),
                    "times": epochs_ds.times,
                    "ch_names": np.array(epochs_ds.ch_names, dtype=object),
                    "sfreq": float(epochs_ds.info["sfreq"]),
                },
            )

            # Free memory before moving to the next participant stream.
            del epochs_ds
        del epochs
        del raw_ppt

    del raw

print("Preprocessing completed.")
