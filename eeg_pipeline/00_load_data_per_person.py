"""
Step 00: Load and split hyperscanning data per person.
Downsamples immediately to reduce memory usage.
"""

import mne
from mne_bids import BIDSPath, read_raw_bids
import config
import warnings
warnings.filterwarnings('ignore')


def load_and_split(subj):
    """Load BIDS data, split into two participants, and downsample immediately."""
    print(f"\n=== Loading subject {subj} ===")
    
    bids_path = BIDSPath(
        subject=subj,
        task=config.TASK,
        datatype='eeg',
        root=config.BIDS_ROOT
    )
    
    try:
        raw = read_raw_bids(bids_path, verbose=False)
    except Exception as e:
        print(f"  Error loading {subj}: {e}")
        return False
    
    # Get all channel names
    all_chs = raw.ch_names
    
    # Find channels for each player dynamically
    p1_chs = [ch for ch in all_chs if ch.startswith('1-')]
    p2_chs = [ch for ch in all_chs if ch.startswith('2-')]
    stim_chs = [ch for ch in all_chs if 'Status' in ch or 'STI' in ch]
    
    if not p1_chs or not p2_chs:
        print(f"  Warning: Missing player channels for {subj}")
        return False
    
    print(f"  Found {len(p1_chs)} P1 channels, {len(p2_chs)} P2 channels")
    print(f"  Original sampling rate: {raw.info['sfreq']} Hz")
    
    # Downsample BEFORE splitting to reduce memory
    print(f"  Downsampling to {config.DOWNSAMPLE_SFREQ} Hz (memory optimization)...")
    raw.load_data()
    raw.resample(config.DOWNSAMPLE_SFREQ, verbose=False)
    
    # Now split into P1 and P2
    for person, p_chs in [("P1", p1_chs), ("P2", p2_chs)]:
        print(f"  Processing {person}...")
        
        # Pick channels for this player
        raw_p = raw.copy().pick(p_chs + stim_chs)
        
        # Set channel types
        for ch in raw_p.ch_names:
            if 'Temp' in ch:
                raw_p.set_channel_types({ch: 'misc'})
            elif 'Erg' in ch or 'Resp' in ch or 'Plet' in ch:
                raw_p.set_channel_types({ch: 'misc'})
            elif 'Status' in ch:
                raw_p.set_channel_types({ch: 'stim'})
        
        # Save
        out_path = config.PROCESSED_DATA_DIR / f"sub-{subj}_{person}_raw.fif"
        raw_p.save(out_path, overwrite=True)
        print(f"  Saved: {out_path}")
        
        # Free memory
        del raw_p
    
    del raw
    return True


if __name__ == "__main__":
    print("="*60)
    print("STEP 00: LOAD DATA PER PERSON (WITH DOWNSAMPLING)")
    print("="*60)
    
    success_count = 0
    fail_count = 0
    
    for subj in config.SUBJECTS:
        try:
            if load_and_split(subj):
                success_count += 1
            else:
                fail_count += 1
        except Exception as e:
            print(f"  Unexpected error for {subj}: {e}")
            fail_count += 1
    
    print("\n" + "="*60)
    print(f"STEP 00 COMPLETE: {success_count} succeeded, {fail_count} failed")
    print("="*60)