import mne
from pathlib import Path
import config
import warnings
warnings.filterwarnings('ignore')


def filter_data(subj, person):
    """Apply notch filter and bandpass filter to EEG data."""
    
    in_path = config.PROCESSED_DATA_DIR / f"sub-{subj}_{person}_reref_raw.fif"
    out_path = config.PROCESSED_DATA_DIR / f"sub-{subj}_{person}_filtered_raw.fif"
    
    if not in_path.exists():
        print(f"  File not found: {in_path}")
        return False
    
    print(f"Loading: {in_path}")
    raw = mne.io.read_raw_fif(in_path, preload=True, verbose=False)
    
    # Check for EEG channels
    eeg_picks = mne.pick_types(raw.info, eeg=True)
    
    if len(eeg_picks) == 0:
        print(f"  Warning: No EEG channels found!")
        raw.save(out_path, overwrite=True)
        return True
    
    print(f"  Found {len(eeg_picks)} EEG channels")
    
    # Apply NOTCH filter for line noise (50 Hz for Europe, 60 Hz for US)
    print(f"  Applying notch filter at 50 Hz (and harmonics)...")
    raw.notch_filter(freqs=[50, 100, 150], picks='eeg', verbose=False)
    
    # Apply BANDPASS filter
    print(f"  Applying bandpass filter: {config.FREQ_LOWER} - {config.FREQ_UPPER} Hz")
    raw.filter(l_freq=config.FREQ_LOWER, h_freq=config.FREQ_UPPER, picks='eeg', verbose=False)
    
    # Save
    raw.save(out_path, overwrite=True)
    print(f"  Saved: {out_path}")
    
    return True


if __name__ == "__main__":
    print("="*60)
    print("STEP 03: NOTCH + BANDPASS FILTER")
    print("="*60)
    
    success_count = 0
    fail_count = 0
    
    for subj in config.SUBJECTS:
        for person in ["P1", "P2"]:
            try:
                if filter_data(subj, person):
                    success_count += 1
                else:
                    fail_count += 1
            except Exception as e:
                print(f"  Error for {subj} {person}: {e}")
                fail_count += 1
    
    print("\n" + "="*60)
    print(f"STEP 03 COMPLETE: {success_count} succeeded, {fail_count} failed")
    print("="*60)