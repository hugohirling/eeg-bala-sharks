import mne
from mne.preprocessing import ICA
from pathlib import Path
import numpy as np
import config
import warnings
warnings.filterwarnings('ignore')


def run_ica(subj, person):
    """Run ICA for artifact removal using frontal channels for EOG detection."""
    
    in_path = config.PROCESSED_DATA_DIR / f"sub-{subj}_{person}_filtered_raw.fif"
    out_path = config.PROCESSED_DATA_DIR / f"sub-{subj}_{person}_ica_raw.fif"
    
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
    
    # Determine number of ICA components
    n_components = min(config.ICA_N_COMPONENTS, len(eeg_picks) - 1)
    print(f"  Running ICA with {n_components} components...")
    
    ica = ICA(
        n_components=n_components,
        method='fastica',
        random_state=config.ICA_RANDOM_STATE,
        max_iter=config.ICA_MAX_ITER
    )
    
    ica.fit(raw, picks='eeg', verbose=False)
    
    # Auto-detect artifacts using frontal channels (Fp1, Fp2, AF7, AF8)
    # These channels capture eye blinks/movements
    eog_indices = []
    
    # Try using frontal channels as EOG proxy
    frontal_channels = ['Fp1', 'Fp2', 'Fpz', 'AF7', 'AF8']
    available_frontal = [ch for ch in frontal_channels if ch in raw.ch_names]
    
    if available_frontal:
        print(f"  Using frontal channels for EOG detection: {available_frontal}")
        try:
            # Create EOG epochs from frontal channels
            eog_epochs = mne.preprocessing.create_eog_epochs(
                raw, ch_name=available_frontal[0], verbose=False
            )
            if len(eog_epochs) > 0:
                eog_indices, eog_scores = ica.find_bads_eog(
                    eog_epochs, ch_name=available_frontal[0], verbose=False
                )
        except Exception as e:
            print(f"  EOG epoch detection failed: {e}")
    
    # If no EOG detected, use correlation with frontal channels
    if not eog_indices and available_frontal:
        print(f"  Trying correlation-based detection...")
        try:
            # Get ICA sources
            sources = ica.get_sources(raw).get_data()
            
            # Get frontal channel data
            frontal_data = raw.copy().pick(available_frontal).get_data()
            frontal_mean = frontal_data.mean(axis=0)
            
            # Find components most correlated with frontal activity
            correlations = []
            for i in range(sources.shape[0]):
                corr = np.abs(np.corrcoef(sources[i], frontal_mean)[0, 1])
                correlations.append(corr)
            
            # Mark components with high frontal correlation as artifacts
            threshold = 0.3
            eog_indices = [i for i, c in enumerate(correlations) if c > threshold]
            
            if eog_indices:
                print(f"  Found {len(eog_indices)} components correlated with frontal activity")
        except Exception as e:
            print(f"  Correlation detection failed: {e}")
    
    # Also detect muscle artifacts (high frequency components)
    muscle_indices = []
    try:
        muscle_indices, muscle_scores = ica.find_bads_muscle(raw, verbose=False)
        if muscle_indices:
            print(f"  Found {len(muscle_indices)} muscle artifact components")
    except Exception as e:
        print(f"  Muscle detection not available: {e}")
    
    # Combine all artifact indices
    all_artifact_indices = list(set(eog_indices + muscle_indices))
    
    # Exclude bad components and apply
    if all_artifact_indices:
        # Limit to max 3 components to avoid over-cleaning
        all_artifact_indices = all_artifact_indices[:3]
        ica.exclude = all_artifact_indices
        ica.apply(raw)
        print(f"  Removed {len(all_artifact_indices)} artifact components: {all_artifact_indices}")
    else:
        print(f"  No artifact components detected, keeping all")
    
    # Save
    raw.save(out_path, overwrite=True)
    print(f"  Saved: {out_path}")
    
    return True


if __name__ == "__main__":
    print("="*60)
    print("STEP 04: ICA ARTIFACT REMOVAL")
    print("="*60)
    
    success_count = 0
    fail_count = 0
    
    for subj in config.SUBJECTS:
        for person in ["P1", "P2"]:
            try:
                if run_ica(subj, person):
                    success_count += 1
                else:
                    fail_count += 1
            except Exception as e:
                print(f"  Error for {subj} {person}: {e}")
                fail_count += 1
    
    print("\n" + "="*60)
    print(f"STEP 04 COMPLETE: {success_count} succeeded, {fail_count} failed")
    print("="*60)