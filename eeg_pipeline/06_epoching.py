"""
Step 06: Create THREE separate epoch types:
1. Trial-locked epochs (around game events)
2. Response-locked epochs (around player responses)
3. Fixed-length epochs (for continuous analysis)
"""

import mne
import numpy as np
from pathlib import Path
import config
import warnings
warnings.filterwarnings('ignore')


def create_epochs(subj, person):
    """Create three types of epochs from continuous EEG data."""
    
    in_path = config.PROCESSED_DATA_DIR / f"sub-{subj}_{person}_downsampled_raw.fif"
    
    if not in_path.exists():
        print(f"  File not found: {in_path}")
        return False
    
    print(f"Loading: {in_path}")
    raw = mne.io.read_raw_fif(in_path, preload=True, verbose=False)
    
    epochs_created = []
    
    # ===== EPOCH TYPE 1: Trial-locked epochs =====
    print(f"  Creating trial-locked epochs...")
    try:
        events = mne.find_events(raw, stim_channel='Status', verbose=False)
        
        if len(events) > 0:
            # Filter for trial start events (adjust event_id based on your data)
            trial_events = events[events[:, 2] > 0]  # Non-zero events
            
            if len(trial_events) > 0:
                epochs_trial = mne.Epochs(
                    raw,
                    trial_events,
                    tmin=-0.5,  # 500ms before event
                    tmax=1.5,   # 1500ms after event
                    baseline=(-0.5, 0),
                    preload=True,
                    verbose=False
                )
                
                out_path = config.OUTPUT_DIRS['epochs'] / f"sub-{subj}_{person}_trial-epo.fif"
                epochs_trial.save(out_path, overwrite=True)
                print(f"    Trial-locked: {len(epochs_trial)} epochs -> {out_path.name}")
                epochs_created.append('trial')
    except Exception as e:
        print(f"    Trial-locked epochs failed: {e}")
    
    # ===== EPOCH TYPE 2: Response-locked epochs =====
    print(f"  Creating response-locked epochs...")
    try:
        events = mne.find_events(raw, stim_channel='Status', verbose=False)
        
        if len(events) > 0:
            # Filter for response events (adjust based on your event coding)
            # Assuming response events have specific codes
            response_events = events[events[:, 2] >= 10]  # Example filter
            
            if len(response_events) > 0:
                epochs_response = mne.Epochs(
                    raw,
                    response_events,
                    tmin=-1.0,  # 1000ms before response
                    tmax=0.5,   # 500ms after response
                    baseline=(-1.0, -0.5),
                    preload=True,
                    verbose=False
                )
                
                out_path = config.OUTPUT_DIRS['epochs'] / f"sub-{subj}_{person}_response-epo.fif"
                epochs_response.save(out_path, overwrite=True)
                print(f"    Response-locked: {len(epochs_response)} epochs -> {out_path.name}")
                epochs_created.append('response')
    except Exception as e:
        print(f"    Response-locked epochs failed: {e}")
    
    # ===== EPOCH TYPE 3: Fixed-length epochs (always created) =====
    print(f"  Creating fixed-length epochs...")
    try:
        epoch_duration = config.EPOCH_TMAX - config.EPOCH_TMIN
        
        epochs_fixed = mne.make_fixed_length_epochs(
            raw,
            duration=epoch_duration,
            preload=True,
            verbose=False
        )
        
        out_path = config.OUTPUT_DIRS['epochs'] / f"sub-{subj}_{person}_fixed-epo.fif"
        epochs_fixed.save(out_path, overwrite=True)
        print(f"    Fixed-length: {len(epochs_fixed)} epochs ({epoch_duration}s each) -> {out_path.name}")
        epochs_created.append('fixed')
        
        # Also save as the default epoch file for backward compatibility
        out_path_default = config.OUTPUT_DIRS['epochs'] / f"sub-{subj}_{person}-epo.fif"
        epochs_fixed.save(out_path_default, overwrite=True)
        
    except Exception as e:
        print(f"    Fixed-length epochs failed: {e}")
    
    print(f"  Created epoch types: {epochs_created}")
    
    return len(epochs_created) > 0


if __name__ == "__main__":
    print("="*60)
    print("STEP 06: EPOCHING (3 TYPES)")
    print("="*60)
    
    # Ensure output directory exists
    config.OUTPUT_DIRS['epochs'].mkdir(parents=True, exist_ok=True)
    
    success_count = 0
    fail_count = 0
    
    for subj in config.SUBJECTS:
        for person in ["P1", "P2"]:
            try:
                if create_epochs(subj, person):
                    success_count += 1
                else:
                    fail_count += 1
            except Exception as e:
                print(f"  Error for {subj} {person}: {e}")
                fail_count += 1
    
    print("\n" + "="*60)
    print(f"STEP 06 COMPLETE: {success_count} succeeded, {fail_count} failed")
    print("="*60)