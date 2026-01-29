import mne
from pathlib import Path
import config
import warnings
warnings.filterwarnings('ignore')


def rereference(subj, person):
    """Apply common average reference to EEG channels."""
    
    in_path = config.PROCESSED_DATA_DIR / f"sub-{subj}_{person}_renamed_raw.fif"
    out_path = config.PROCESSED_DATA_DIR / f"sub-{subj}_{person}_reref_raw.fif"
    
    if not in_path.exists():
        print(f"  File not found: {in_path}")
        return False
    
    print(f"Loading: {in_path}")
    raw = mne.io.read_raw_fif(in_path, preload=True, verbose=False)
    
    # Check for EEG channels
    eeg_picks = mne.pick_types(raw.info, eeg=True)
    
    if len(eeg_picks) == 0:
        print(f"  Warning: No EEG channels found!")
        # Still save the file to continue pipeline
        raw.save(out_path, overwrite=True)
        return True
    
    print(f"  Found {len(eeg_picks)} EEG channels")
    print(f"  Applying common average reference...")
    raw.set_eeg_reference('average', projection=False)
    
    # Save
    raw.save(out_path, overwrite=True)
    print(f"  Saved: {out_path}")
    
    return True


if __name__ == "__main__":
    print("="*60)
    print("STEP 02: RE-REFERENCE (COMMON AVERAGE)")
    print("="*60)
    
    success_count = 0
    fail_count = 0
    
    for subj in config.SUBJECTS:
        for person in ["P1", "P2"]:
            try:
                if rereference(subj, person):
                    success_count += 1
                else:
                    fail_count += 1
            except Exception as e:
                print(f"  Error for {subj} {person}: {e}")
                fail_count += 1
    
    print("\n" + "="*60)
    print(f"STEP 02 COMPLETE: {success_count} succeeded, {fail_count} failed")
    print("="*60)