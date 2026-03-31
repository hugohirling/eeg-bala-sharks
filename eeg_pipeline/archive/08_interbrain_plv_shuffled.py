from pathlib import Path
import numpy as np
import mne
from mne_connectivity import spectral_connectivity_epochs
from utils import load_raw, save_raw
from preprocessing import config

# Interbrain Synchrony (IBS) parameters
# -------------------------------
# IBS quantifies phase-based coupling between the EEG signals of two participants.
# In our Rock-Paper-Scissors (RPS) dyads, IBS can capture moments of shared attention,
# anticipation, and strategic interaction between players.
# We focus on the alpha band (8–12 Hz) because alpha activity is linked to attention
# and anticipation processes, which are crucial during competitive decision-making.
# Computing phase-locking values (PLV) between homologous channels of P1 and P2
# allows us to measure how synchronized their brain activity is during the game.

def equalize_epochs_np(data_list):
    """
    Equalize number of epochs across multiple 3D arrays (n_epochs, n_channels, n_times)
    Truncate to the minimum number of epochs.
    """
    n_epochs_list = [d.shape[0] for d in data_list]
    min_epochs = min(n_epochs_list)
    return [d[:min_epochs] for d in data_list]

def compute_interbrain_plv(epochs_p1, epochs_p2, fmin, fmax):
    """
    Compute interbrain PLV between homologous EEG channels of two participants.
    """
    # --- Pick EEG only ---
    epochs_p1.pick_types(eeg=True)
    epochs_p2.pick_types(eeg=True)

    # --- Basic checks ---
    if epochs_p1.info["sfreq"] != epochs_p2.info["sfreq"]:
        raise ValueError("Sampling frequencies differ between participants.")

    if epochs_p1.ch_names != epochs_p2.ch_names:
        raise ValueError("Channel names/order differ between participants."
                         " Align channels before running PLV.")

    # --- Convert to numpy arrays ---
    data_p1 = epochs_p1.get_data()  # (n_epochs, n_channels, n_times)
    data_p2 = epochs_p2.get_data()

    # --- Equalize epoch counts ---
    data_p1, data_p2 = equalize_epochs_np([data_p1, data_p2])

    ch_names = epochs_p1.ch_names
    n_ch = len(ch_names)

    # --- Concatenate channels so connectivity can be computed across subjects ---
    # combined shape: (n_epochs, n_ch * 2, n_times)
    combined = np.concatenate([data_p1, data_p2], axis=1)

    # Homologous channel indices: (i in first subject, i+n_ch in second)
    indices = (np.arange(n_ch), np.arange(n_ch) + n_ch)

    # --- Compute PLV ---
    conn_res = spectral_connectivity_epochs(
        data=combined,
        method="plv",
        mode="fourier",
        sfreq=epochs_p1.info["sfreq"],
        fmin=fmin,
        fmax=fmax,
        faverage=True,
        indices=indices,
        n_jobs=1,
    )

    # `spectral_connectivity_epochs` may return a SpectralConnectivity
    # object (newer mne_connectivity) or a tuple (older API). Handle both.
    if isinstance(conn_res, tuple):
        con = conn_res[0]
        freqs = conn_res[1] if len(conn_res) > 1 else None
        times = conn_res[2] if len(conn_res) > 2 else None
        n_epochs = conn_res[3] if len(conn_res) > 3 else None
    else:
        # SpectralConnectivity-like object
        if hasattr(conn_res, "get_data"):
            con = conn_res.get_data()
        elif hasattr(conn_res, "con"):
            con = np.asarray(conn_res.con)
        else:
            con = np.asarray(conn_res)

        freqs = getattr(conn_res, "freqs", None)
        times = getattr(conn_res, "times", None)
        n_epochs = getattr(conn_res, "n_epochs", None)

    # con shape typically: (n_connections, n_freqs) or (n_connections,)
    con = np.asarray(con)
    if con.ndim == 2:
        plv_per_pair = np.mean(con, axis=1)
    else:
        plv_per_pair = con.squeeze()

    return plv_per_pair, ch_names

def process_subject(path_p1, path_p2, out_path):
    """
    Load epoch files for P1 and P2, compute interbrain PLV,
    and save results as numpy file.
    """
    print(f"\nProcessing dyad:\n  {path_p1.name}\n  {path_p2.name}")

    # --- Load epochs ---
    try:
        epochs_p1 = mne.read_epochs(path_p1, preload=True)
        epochs_p2 = mne.read_epochs(path_p2, preload=True)
    except FileNotFoundError:
        print(f"Epoch file not found for dyad: {path_p1}, {path_p2}")
        return

    # --- Compute PLV ---
    plv, ch_names = compute_interbrain_plv(
        epochs_p1,
        epochs_p2,
        fmin=config.IBS_FMIN,
        fmax=config.IBS_FMAX,
    )

    # --- Save results ---
    result = {
        "plv": plv,                  # shape: (n_channels,)
        "channels": ch_names,
        "fmin": config.IBS_FMIN,
        "fmax": config.IBS_FMAX,
        "subject": path_p1.stem.split("_")[0],
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, result, allow_pickle=True)
    print(f"Saved interbrain PLV to: {out_path}")

if __name__ == "__main__":
    for subj in config.SUBJECTS:
        # Nutze MNE-konforme Dateinamen (_epo.fif)
        p1 = Path(config.OUTPUT_DIR) / f"sub-{subj}_P1_epoch.fif"
        p2 = Path(config.OUTPUT_DIR) / f"sub-{subj}_P2_epoch.fif"

        out_file = Path(config.OUTPUT_DIR) / f"sub-{subj}_interbrain_plv.npy"

        process_subject(p1, p2, out_file)
