import mne
from pathlib import Path
import config
import warnings
warnings.filterwarnings('ignore')


def downsample(subj, person):
    """Check/apply downsampling (may already be done in step 00)."""
    
    in_path = config.PROCESSED_DATA_DIR / f"sub-{subj}_{person}_ica_raw.fif"
    out_path = config.PROCESSED_DATA_DIR / f"sub-{subj}_{person}_downsampled_raw.fif"
    
    if not in_path.exists():
        print(f"  File not found: {in_path}")
        return False
    
    print(f"Loading: {in_path}")
    raw = mne.io.read_raw_fif(in_path, preload=True, verbose=False)
    
    current_sfreq = raw.info['sfreq']
    target_sfreq = config.DOWNSAMPLE_SFREQ
    
    print(f"  Current sampling rate: {current_sfreq} Hz")
    
    if current_sfreq > target_sfreq:
        print(f"  Downsampling to {target_sfreq} Hz...")
        raw.resample(target_sfreq, verbose=False)
    else:
        print(f"  Already at target rate ({current_sfreq} Hz), skipping downsample")
    
    # Save
    raw.save(out_path, overwrite=True)
    print(f"  Saved: {out_path}")
    
    return True


if __name__ == "__main__":
    print("="*60)
    print("STEP 05: DOWNSAMPLE")
    print("="*60)
    
    success_count = 0
    fail_count = 0
    
    for subj in config.SUBJECTS:
        for person in ["P1", "P2"]:
            try:
                if downsample(subj, person):
                    success_count += 1
                else:
                    fail_count += 1
            except Exception as e:
                print(f"  Error for {subj} {person}: {e}")
                fail_count += 1
    
    print("\n" + "="*60)
    print(f"STEP 05 COMPLETE: {success_count} succeeded, {fail_count} failed")
    print("="*60)