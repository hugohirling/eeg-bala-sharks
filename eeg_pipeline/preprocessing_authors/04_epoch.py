"""
Epoch EEG Data
Following Moerel et al. (2025) preprocessing pipeline
Step 5: Create epochs for Decision, Response, and Feedback phases
"""

import os
import logging
import mne
import numpy as np
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_data(input_file):
    """Load EEG data"""
    logger.info(f"Loading data from: {input_file}")
    raw = mne.io.read_raw_fif(input_file, preload=True)
    logger.info(f"Data shape: {raw.get_data().shape}")
    logger.info(f"Sampling rate: {raw.info['sfreq']} Hz")
    return raw


def create_events_from_raw(raw):
    """
    Create event markers for epoching
    
    According to Moerel et al. (2025):
    - Each game consisted of three phases: Decision (2s), Response (2s), Feedback (1s)
    - Epochs are time-locked to the onset of each phase
    
    This function looks for stim channels or annotations to create the event structure
    """
    logger.info("Creating event structure from raw data...")
    
    # Check if there are stimulus channels
    stim_channels = mne.pick_types(raw.info, stim=True)
    
    if len(stim_channels) > 0:
        # Extract events from stimulus channel
        logger.info(f"Found {len(stim_channels)} stimulus channel(s)")
        events = mne.find_events(raw, stim_channel=raw.ch_names[stim_channels[0]])
        logger.info(f"Found {len(events)} events in stimulus channel")
        return events
    
    # Alternative: check for annotations
    if len(raw.annotations) > 0:
        logger.info(f"Found {len(raw.annotations)} annotations in raw data")
        events, event_dict = mne.events_from_annotations(raw)
        logger.info(f"Converted {len(events)} annotations to events")
        logger.info(f"Event types: {event_dict}")
        return events
    
    logger.warning("No stimulus channel or annotations found in raw data")
    logger.warning("Creating synthetic events for demonstration (every 5 seconds, corresponding to trial duration)")
    
    # Create synthetic events for demonstration (5 seconds per trial = Decision 2s + Response 2s + Feedback 1s)
    sfreq = raw.info['sfreq']
    trial_duration_samples = int(5 * sfreq)  # 5 seconds per trial
    n_samples = raw.n_times
    
    event_samples = np.arange(0, n_samples, trial_duration_samples)
    events = np.column_stack([event_samples, np.zeros_like(event_samples), np.ones_like(event_samples)])
    
    logger.info(f"Created {len(events)} synthetic trial events")
    return events


def epoch_data(raw, events):
    """
    Create epochs for all three phases
    Following Moerel et al. (2025):
    "we made three separate epochs for each trial, locked to the onset of the Decision screen 
    (-200 ms – 2000 ms), Response screen (-200 ms – 2000 ms), and Feedback screen (-200 ms – 1000 ms) respectively"
    """
    
    logger.info("Creating epochs for three phases...")
    
    # Create event dictionary
    event_id = {
        'trial_start': 1
    }
    
    sfreq = raw.info['sfreq']
    
    # Define tmin and tmax for each phase
    # Note: We need to adjust this based on actual event markers in the data
    # For now, we create epochs relative to trial start
    
    epochs_dict = {}
    
    # Epoch 1: Decision phase (-200 ms to 2000 ms from Decision screen onset)
    # The Decision screen appears at trial start
    logger.info("Creating Decision phase epochs (-200 ms to 2000 ms)...")
    epochs_decision = mne.Epochs(
        raw,
        events,
        event_id=event_id,
        tmin=-0.2,  # -200 ms
        tmax=2.0,   # 2000 ms
        baseline=(-0.2, 0),  # Baseline correction from -200 to 0 ms
        preload=True
    )
    epochs_dict['decision'] = epochs_decision
    logger.info(f"Decision phase: {len(epochs_decision)} epochs")
    
    # Epoch 2: Response phase (-200 ms to 2000 ms from Response screen onset)
    # Response occurs after Decision phase (2 seconds after trial start)
    logger.info("Creating Response phase epochs (-200 ms to 2000 ms)...")
    response_onset = 2.0  # 2 seconds after trial start
    
    # Create response events
    response_events = events.copy()
    response_events[:, 0] = response_events[:, 0] + int(response_onset * sfreq)
    
    epochs_response = mne.Epochs(
        raw,
        response_events,
        event_id=event_id,
        tmin=-0.2,  # -200 ms
        tmax=2.0,   # 2000 ms
        baseline=(-0.2, 0),
        preload=True
    )
    epochs_dict['response'] = epochs_response
    logger.info(f"Response phase: {len(epochs_response)} epochs")
    
    # Epoch 3: Feedback phase (-200 ms to 1000 ms from Feedback screen onset)
    # Feedback occurs after Response phase (4 seconds after trial start)
    logger.info("Creating Feedback phase epochs (-200 ms to 1000 ms)...")
    feedback_onset = 4.0  # 4 seconds after trial start
    
    # Create feedback events
    feedback_events = events.copy()
    feedback_events[:, 0] = feedback_events[:, 0] + int(feedback_onset * sfreq)
    
    epochs_feedback = mne.Epochs(
        raw,
        feedback_events,
        event_id=event_id,
        tmin=-0.2,   # -200 ms
        tmax=1.0,    # 1000 ms (Feedback phase is shorter: 1 second)
        baseline=(-0.2, 0),
        preload=True
    )
    epochs_dict['feedback'] = epochs_feedback
    logger.info(f"Feedback phase: {len(epochs_feedback)} epochs")
    
    return epochs_dict


def save_epochs(epochs_dict, output_dir, subject_id):
    """Save epochs for each phase"""
    logger.info("Saving epochs...")
    os.makedirs(output_dir, exist_ok=True)
    
    for phase, epochs in epochs_dict.items():
        output_file = os.path.join(output_dir, f'{subject_id}_{phase}-epo.fif')
        epochs.save(output_file, overwrite=True)
        logger.info(f"Saved {phase} phase epochs: {output_file}")
    
    # Also save metadata
    metadata = {
        'n_epochs_decision': len(epochs_dict['decision']),
        'n_epochs_response': len(epochs_dict['response']),
        'n_epochs_feedback': len(epochs_dict['feedback']),
        'sfreq': int(epochs_dict['decision'].info['sfreq']),
        'n_channels': len(epochs_dict['decision'].ch_names)
    }
    
    return metadata


def main(input_file, output_dir, subject_id):
    """Main processing function"""
    logger.info("=" * 60)
    logger.info("Step 5: Epoch Data into Three Phases")
    logger.info("=" * 60)
    logger.info("Phases: Decision, Response, Feedback")
    logger.info("Following Moerel et al. (2025) paradigm")
    
    # Load data
    raw = load_data(input_file)
    
    # Create events
    events = create_events_from_raw(raw)
    
    # Create epochs
    epochs_dict = epoch_data(raw, events)
    
    # Save epochs
    metadata = save_epochs(epochs_dict, output_dir, subject_id)
    
    logger.info("Step 5 completed successfully!")
    logger.info(f"Epochs created - Decision: {metadata['n_epochs_decision']}, "
                f"Response: {metadata['n_epochs_response']}, "
                f"Feedback: {metadata['n_epochs_feedback']}")
    logger.info("=" * 60)


if __name__ == "__main__":
    # Example usage
    from config import settings
    
    subject = "sub-01"
    output_dir = Path(settings.OUTPUT_ROOT) / "preprocessing_authors"
    
    input_file = output_dir / f"{subject}_downsampled.fif"
    
    if input_file.exists():
        main(str(input_file), str(output_dir), subject)
    else:
        logger.warning(f"Input file not found: {input_file}")
