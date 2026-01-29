"""
Step 09: Time-Frequency Analysis (ERPs, TFRs, PSD)
"""

import mne
import numpy as np
import pandas as pd
from pathlib import Path
import config
import warnings
warnings.filterwarnings('ignore')


def compute_time_frequency(subj):
    """Compute time-frequency analysis for a subject."""
    
    results = []
    
    for person in ["P1", "P2"]:
        epochs_path = config.OUTPUT_DIRS['epochs'] / f"sub-{subj}_{person}-epo.fif"
        
        if not epochs_path.exists():
            print(f"  Warning: Epochs not found: {epochs_path}")
            continue
        
        print(f"  Loading {person}: {epochs_path.name}")
        epochs = mne.read_epochs(epochs_path, preload=True, verbose=False)
        
        # Pick only EEG channels
        epochs = epochs.pick('eeg')
        
        # Compute PSD for each frequency band
        for band_name, (fmin, fmax) in config.FREQ_BANDS.items():
            try:
                # Compute PSD using Welch method
                psd = epochs.compute_psd(method='welch', fmin=fmin, fmax=fmax, verbose=False)
                psd_data = psd.get_data().mean()  # Mean power in band
                
                results.append({
                    'subject': subj,
                    'person': person,
                    'band': band_name,
                    'fmin': fmin,
                    'fmax': fmax,
                    'power': psd_data
                })
                
            except Exception as e:
                print(f"    Error computing {band_name}: {e}")
        
        # Compute ERP (evoked response)
        try:
            evoked = epochs.average()
            erp_peak = np.max(np.abs(evoked.data))
            erp_latency = evoked.times[np.argmax(np.abs(evoked.data.mean(axis=0)))]
            
            results.append({
                'subject': subj,
                'person': person,
                'band': 'ERP',
                'fmin': np.nan,
                'fmax': np.nan,
                'power': erp_peak,
                'latency': erp_latency
            })
        except Exception as e:
            print(f"    Error computing ERP: {e}")
    
    return results


if __name__ == "__main__":
    print("="*60)
    print("STEP 09: TIME-FREQUENCY ANALYSIS")
    print("="*60)
    
    # Ensure output directory exists
    config.OUTPUT_DIRS['time_frequency'].mkdir(parents=True, exist_ok=True)
    
    all_results = []
    
    for subj in config.SUBJECTS:
        print(f"\nProcessing subject {subj}...")
        try:
            results = compute_time_frequency(subj)
            all_results.extend(results)
        except Exception as e:
            print(f"  Error: {e}")
    
    # Save results
    if all_results:
        df = pd.DataFrame(all_results)
        out_path = config.OUTPUT_DIRS['time_frequency'] / "time_frequency_summary.csv"
        df.to_csv(out_path, index=False)
        print(f"\n✓ Saved results to {out_path}")
        print(df.to_string())
    else:
        print("\n⚠ No results to save")
    
    print("\n" + "="*60)
    print("STEP 09 COMPLETE")
    print("="*60)