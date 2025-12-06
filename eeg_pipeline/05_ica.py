import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent)) 

# 05_ica.py
from mne.preprocessing import ICA
from utils import load_raw, save_object
import config

def run_ica(subject_id):
    epochs_file = config.OUTPUT_DIR / f"epochs_{subject_id}.fif"
    epochs = load_raw(epochs_file)

    n_components = min(config.ICA_N_COMPONENTS, len(epochs.ch_names) - 1)
    ica = ICA(n_components=n_components, max_iter=config.ICA_MAX_ITER, random_state=42, method='fastica')
    ica.fit(epochs, picks='eeg')

    save_object(ica, config.OUTPUT_DIR / f"ica_{subject_id}.pkl")
    print(f"[05_ica] ICA fit complete for {subject_id}, {n_components} components saved.")
    return ica

if __name__ == "__main__":
    for subj in config.SUBJECTS:
        run_ica(subj)
