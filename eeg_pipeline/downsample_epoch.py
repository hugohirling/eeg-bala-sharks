# downsample_epoch.py
import mne
from pathlib import Path
from utils import load_raw
import config

EPOCH_PARAMS = dict(tmin=0, tmax=2)  # seconds per epoch
DOWNSAMPLE_RATE = 200  # Hz

def epoch_downsample(subject_id):
    # Step 1: Find all split raw files for this subject
    raw_files = sorted(Path(config.OUTPUT_DIR).glob(f"raw_{subject_id}*.fif"))
    if not raw_files:
        print(f"No raw files found for subject {subject_id}")
        return

    all_epochs_files = []

    # Step 2: Process each raw file separately
    for raw_file in raw_files:
        print(f"Processing raw file {raw_file}...")

        # Load raw with memory-mapping (safer for large files)
        raw = load_raw(raw_file, preload='memmap')

        # Downsample all channels (memory-efficient with memmap)
        print(f"Downsampling {raw_file.name} to {DOWNSAMPLE_RATE} Hz...")
        raw.resample(DOWNSAMPLE_RATE, npad='auto', n_jobs=1)

        # Create fixed-length epochs
        events = mne.make_fixed_length_events(raw, duration=EPOCH_PARAMS['tmax'] - EPOCH_PARAMS['tmin'])
        epochs = mne.Epochs(raw, events, tmin=EPOCH_PARAMS['tmin'], tmax=EPOCH_PARAMS['tmax'],
                            baseline=None, preload=True)

        # Save this chunk
        out_file = Path(config.OUTPUT_DIR) / f"epochs_{subject_id}_{raw_file.stem}-epo.fif"
        print(f"Saving epochs to {out_file}...")
        epochs.save(out_file, overwrite=True)
        all_epochs_files.append(out_file)

    # Optional: concatenate all epoch chunks
    if len(all_epochs_files) > 1:
        print("Concatenating all epoch chunks into one file...")
        epochs_list = [mne.read_epochs(f, preload=True) for f in all_epochs_files]
        combined_epochs = mne.concatenate_epochs(epochs_list)
        final_file = Path(config.OUTPUT_DIR) / f"epochs_{subject_id}-epo.fif"
        combined_epochs.save(final_file, overwrite=True)
        print(f"Saved combined epochs to {final_file}")

if __name__ == "__main__":
    for subj in config.SUBJECTS:
        epoch_downsample(subj)
