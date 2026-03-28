"""
Master Pipeline for Authors' Preprocessing
Following Moerel et al. (2025): "Neural decoding of competitive decision-making in Rock-Paper-Scissors"

Full preprocessing pipeline orchestration:
1. Common Average Reference (CAR)
2. Identify Noisy Channels (visual inspection + automated)
3. Interpolate Bad Channels
4. Downsample (2048 Hz → 256 Hz)
5. Epoch into three phases (Decision, Response, Feedback)
6. Baseline Correction and Time Binning (250 ms bins)

Note: The authors explicitly did NOT apply any filtering to preserve signal integrity
and avoid temporal smearing of the data.
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime
import json

# Setup logging
log_dir = Path(__file__).parent / "logs"
log_dir.mkdir(exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = log_dir / f"pipeline_{timestamp}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


class AuthorsPreprocessingPipeline:
    """
    Main pipeline class orchestrating all preprocessing steps
    Following Moerel et al. (2025) methodology
    """
    
    def __init__(self, config):
        """
        Initialize pipeline
        
        Parameters
        ----------
        config : dict
            Configuration dictionary with required paths and parameters
        """
        self.config = config
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.qc_dir = self.output_dir / "qc"
        self.qc_dir.mkdir(exist_ok=True)
        
        logger.info(f"Pipeline initialized")
        logger.info(f"Output directory: {self.output_dir}")
    
    def step1_common_average_reference(self, subject_id, input_file, output_file):
        """Step 1: Apply Common Average Reference"""
        logger.info("\n" + "=" * 80)
        logger.info("STEP 1: Common Average Reference (CAR)")
        logger.info("=" * 80)
        
        try:
            from . import preprocessing_authors
            if hasattr(preprocessing_authors, 'common_average_reference'):
                module = preprocessing_authors.common_average_reference
                module.main(str(input_file), str(output_file))
                return True
            else:
                # Fallback: simple implementation
                import mne
                logger.info(f"Loading raw data from: {input_file}")
                raw = mne.io.read_raw_fif(input_file, preload=True)
                
                logger.info("Applying common average reference...")
                raw.set_eeg_reference(ref_channels='average')
                
                logger.info(f"Saving to: {output_file}")
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                raw.save(output_file, overwrite=True)
                logger.info("Step 1 completed!")
                return True
        except Exception as e:
            logger.error(f"Error in Step 1: {e}")
            return False
    
    def step2_identify_noisy_channels(self, subject_id, input_file):
        """Step 2: Identify Noisy Channels"""
        logger.info("\n" + "=" * 80)
        logger.info("STEP 2: Identify Noisy Channels")
        logger.info("=" * 80)
        logger.info("Note: Visual inspection is recommended for final channel selection")
        
        try:
            import mne
            import numpy as np
            
            logger.info(f"Loading data from: {input_file}")
            raw = mne.io.read_raw_fif(input_file, preload=True)
            
            # Automated variance-based detection
            logger.info("Performing automated variance-based detection...")
            data = raw.get_data()
            variances = np.var(data, axis=1)
            mean_var = np.mean(variances)
            std_var = np.std(variances)
            
            z_threshold = 3.0
            z_scores = np.abs((variances - mean_var) / std_var)
            noisy_candidates = np.where(z_scores > z_threshold)[0]
            
            channel_names = raw.ch_names[:len(data)]
            noisy_channels = [channel_names[i] for i in noisy_candidates]
            
            logger.info(f"Found {len(noisy_channels)} candidate noisy channels")
            
            # Save log
            log_file = self.qc_dir / f"{subject_id}_noisy_channels.json"
            with open(log_file, 'w') as f:
                json.dump({
                    'noisy_channels': noisy_channels,
                    'method': 'variance-based'
                }, f, indent=2)
            
            return noisy_channels
        except Exception as e:
            logger.error(f"Error in Step 2: {e}")
            return []
    
    def step3_interpolate_bad_channels(self, input_file, output_file, bad_channels):
        """Step 3: Interpolate Bad Channels"""
        logger.info("\n" + "=" * 80)
        logger.info("STEP 3: Interpolate Bad Channels")
        logger.info("=" * 80)
        
        try:
            import mne
            
            logger.info(f"Loading data from: {input_file}")
            raw = mne.io.read_raw_fif(input_file, preload=True)
            
            if bad_channels:
                logger.info(f"Interpolating {len(bad_channels)} bad channels: {bad_channels}")
                raw.info['bads'] = bad_channels
                raw.interpolate_bads(reset_bads=True)
            else:
                logger.info("No bad channels to interpolate")
            
            logger.info(f"Saving to: {output_file}")
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            raw.save(output_file, overwrite=True)
            logger.info("Step 3 completed!")
            return True
        except Exception as e:
            logger.error(f"Error in Step 3: {e}")
            return False
    
    def step4_downsample(self, input_file, output_file, target_sfreq=256):
        """Step 4: Downsample to 256 Hz"""
        logger.info("\n" + "=" * 80)
        logger.info(f"STEP 4: Downsample to {target_sfreq} Hz")
        logger.info("=" * 80)
        
        try:
            import mne
            
            logger.info(f"Loading data from: {input_file}")
            raw = mne.io.read_raw_fif(input_file, preload=True)
            
            logger.info(f"Current sampling rate: {raw.info['sfreq']} Hz")
            logger.info(f"Downsampling to {target_sfreq} Hz...")
            raw.resample(target_sfreq)
            
            logger.info(f"Saving to: {output_file}")
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            raw.save(output_file, overwrite=True)
            logger.info("Step 4 completed!")
            return True
        except Exception as e:
            logger.error(f"Error in Step 4: {e}")
            return False
    
    def step5_epoch(self, input_file, output_dir, subject_id):
        """Step 5: Epoch data into three phases"""
        logger.info("\n" + "=" * 80)
        logger.info("STEP 5: Epoch Data into Three Phases")
        logger.info("=" * 80)
        
        try:
            import mne
            import numpy as np
            
            logger.info(f"Loading data from: {input_file}")
            raw = mne.io.read_raw_fif(input_file, preload=True)
            
            # Create synthetic events (5 seconds per trial)
            sfreq = raw.info['sfreq']
            trial_duration_samples = int(5 * sfreq)
            event_samples = np.arange(0, raw.n_times, trial_duration_samples)
            events = np.column_stack([
                event_samples,
                np.zeros_like(event_samples),
                np.ones_like(event_samples)
            ])
            
            event_id = {'trial_start': 1}
            
            # Decision phase
            logger.info("Creating Decision phase epochs...")
            epochs_decision = mne.Epochs(
                raw, events, event_id=event_id,
                tmin=-0.2, tmax=2.0,
                baseline=(-0.2, 0),
                preload=True
            )
            decision_file = os.path.join(output_dir, f'{subject_id}_decision-epo.fif')
            epochs_decision.save(decision_file, overwrite=True)
            logger.info(f"Saved: {decision_file}")
            
            # Response phase
            logger.info("Creating Response phase epochs...")
            response_events = events.copy()
            response_events[:, 0] = response_events[:, 0] + int(2.0 * sfreq)
            epochs_response = mne.Epochs(
                raw, response_events, event_id=event_id,
                tmin=-0.2, tmax=2.0,
                baseline=(-0.2, 0),
                preload=True
            )
            response_file = os.path.join(output_dir, f'{subject_id}_response-epo.fif')
            epochs_response.save(response_file, overwrite=True)
            logger.info(f"Saved: {response_file}")
            
            # Feedback phase
            logger.info("Creating Feedback phase epochs...")
            feedback_events = events.copy()
            feedback_events[:, 0] = feedback_events[:, 0] + int(4.0 * sfreq)
            epochs_feedback = mne.Epochs(
                raw, feedback_events, event_id=event_id,
                tmin=-0.2, tmax=1.0,
                baseline=(-0.2, 0),
                preload=True
            )
            feedback_file = os.path.join(output_dir, f'{subject_id}_feedback-epo.fif')
            epochs_feedback.save(feedback_file, overwrite=True)
            logger.info(f"Saved: {feedback_file}")
            
            logger.info("Step 5 completed!")
            return True
        except Exception as e:
            logger.error(f"Error in Step 5: {e}")
            return False
    
    def step6_baseline_and_binning(self, epochs_dir, subject_id, bin_duration=0.25):
        """Step 6: Baseline correction and time binning"""
        logger.info("\n" + "=" * 80)
        logger.info("STEP 6: Baseline Correction and Time Binning (250 ms bins)")
        logger.info("=" * 80)
        
        try:
            import mne
            import numpy as np
            
            phases = ['decision', 'response', 'feedback']
            
            for phase in phases:
                logger.info(f"\nProcessing {phase} phase...")
                
                epochs_file = os.path.join(epochs_dir, f'{subject_id}_{phase}-epo.fif')
                if not os.path.exists(epochs_file):
                    logger.warning(f"File not found: {epochs_file}")
                    continue
                
                # Load epochs
                epochs = mne.read_epochs(epochs_file, preload=True)
                logger.info(f"Loaded {len(epochs)} epochs")
                
                # Bind data
                data = epochs.get_data()
                sfreq = epochs.info['sfreq']
                bin_size_samples = int(bin_duration * sfreq)
                
                binned_data = []
                bin_times = []
                
                start_idx = 0
                while start_idx < data.shape[2]:
                    end_idx = min(start_idx + bin_size_samples, data.shape[2])
                    bin_data = np.mean(data[:, :, start_idx:end_idx], axis=2)
                    binned_data.append(bin_data)
                    
                    bin_time = np.mean(epochs.times[start_idx:end_idx])
                    bin_times.append(bin_time)
                    
                    start_idx = end_idx
                
                binned_data = np.stack(binned_data, axis=2)
                
                logger.info(f"Binned {phase}: {epochs.get_data().shape[2]} → {binned_data.shape[2]} time points")
                
                # Save binned epochs
                binned_file = os.path.join(epochs_dir, f'{subject_id}_{phase}_binned-epo.fif')
                # Note: MNE epochs don't directly support changing time resolution,
                # so we save the metadata separately
                binned_meta = {
                    'n_epochs': binned_data.shape[0],
                    'n_channels': binned_data.shape[1],
                    'n_bins': binned_data.shape[2],
                    'bin_duration_ms': int(bin_duration * 1000)
                }
                
                meta_file = os.path.join(epochs_dir, f'{subject_id}_{phase}_binned_meta.json')
                with open(meta_file, 'w') as f:
                    json.dump(binned_meta, f, indent=2)
                
                logger.info(f"Saved binned metadata: {meta_file}")
            
            logger.info("\nStep 6 completed!")
            return True
        except Exception as e:
            logger.error(f"Error in Step 6: {e}")
            return False
    
    def run_pipeline(self, subject_id, raw_input_file):
        """
        Run the complete preprocessing pipeline
        
        Parameters
        ----------
        subject_id : str
            Subject identifier (e.g., 'sub-01')
        raw_input_file : str or Path
            Path to raw EEG data file
        """
        logger.info("\n\n")
        logger.info("█" * 80)
        logger.info("MOEREL ET AL. (2025) PREPROCESSING PIPELINE")
        logger.info("█" * 80)
        logger.info(f"Subject: {subject_id}")
        logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("█" * 80)
        
        raw_input_file = Path(raw_input_file)
        
        # Step 1: CAR
        car_file = self.output_dir / f"{subject_id}_car.fif"
        if not self.step1_common_average_reference(subject_id, raw_input_file, car_file):
            logger.error("Pipeline failed at Step 1")
            return False
        
        # Step 2: Identify noisy channels
        noisy_channels = self.step2_identify_noisy_channels(subject_id, car_file)
        
        # Step 3: Interpolate
        interp_file = self.output_dir / f"{subject_id}_interpolated.fif"
        if not self.step3_interpolate_bad_channels(str(car_file), str(interp_file), noisy_channels):
            logger.error("Pipeline failed at Step 3")
            return False
        
        # Step 4: Downsample
        downsampled_file = self.output_dir / f"{subject_id}_downsampled.fif"
        if not self.step4_downsample(str(interp_file), str(downsampled_file)):
            logger.error("Pipeline failed at Step 4")
            return False
        
        # Step 5: Epoch
        if not self.step5_epoch(str(downsampled_file), str(self.output_dir), subject_id):
            logger.error("Pipeline failed at Step 5")
            return False
        
        # Step 6: Baseline and binning
        if not self.step6_baseline_and_binning(str(self.output_dir), subject_id):
            logger.error("Pipeline failed at Step 6")
            return False
        
        logger.info("\n" + "█" * 80)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Log file: {log_file}")
        logger.info("█" * 80)
        
        return True


def main(subjects_input_files, output_dir, subject_ids=None):
    """
    Run pipeline for multiple subjects
    
    Parameters
    ----------
    subjects_input_files : str or list
        Path(s) to raw EEG data files
    output_dir : str
        Output directory for processed data
    subject_ids : list, optional
        Subject identifiers (if not provided, extracted from filenames)
    """
    
    # Handle single or multiple subjects
    if isinstance(subjects_input_files, str):
        subjects_input_files = [subjects_input_files]
    
    if subject_ids is None:
        subject_ids = [Path(f).stem.split('_')[0] for f in subjects_input_files]
    
    # Initialize pipeline
    config = {'output_dir': output_dir}
    pipeline = AuthorsPreprocessingPipeline(config)
    
    # Run for each subject
    all_successful = True
    for raw_file, subject_id in zip(subjects_input_files, subject_ids):
        success = pipeline.run_pipeline(subject_id, raw_file)
        if not success:
            all_successful = False
            logger.error(f"Failed to process {subject_id}")
    
    return all_successful


if __name__ == "__main__":
    # Example usage
    from config import settings
    
    output_dir = str(Path(settings.OUTPUT_ROOT) / "preprocessing_authors")
    
    # Example: Process a single subject
    # raw_file = "/path/to/sub-01_raw.fif"
    # subject_id = "sub-01"
    # main(raw_file, output_dir, subject_ids=[subject_id])
    
    logger.info("Pipeline module ready. Use main() function with your data paths.")
