"""
Quick Start Guide for Authors' Preprocessing Pipeline
Moerel et al. (2025) - Rock-Paper-Scissors EEG Study
"""

# ============================================================================
# WHAT IS THIS PIPELINE?
# ============================================================================

"""
This folder contains a complete implementation of the preprocessing pipeline
described in the supplementary methods of:

    Moerel et al. (2025): "Neural decoding of competitive decision-making 
    in Rock-Paper-Scissors"

The pipeline implements their exact methodology for EEG preprocessing and is
designed to preserve signal integrity (notably, NO FILTERING is applied).

Output: All processed data goes to output/preprocessing_authors/
"""

# ============================================================================
# QUICK START (3 STEPS)
# ============================================================================

"""
1. PREPARE YOUR DATA
   - Place raw .fif files in the location expected by your config
   - Ensure files contain proper event markers/annotations
   - Recommended: Have your raw data preprocessed by an engineer to confirm
     all events are properly marked

2. RUN THE PIPELINE
   From command line or Python:
   
   >>> from eeg_pipeline.preprocessing_authors import main
   >>> main(
   ...     subjects_input_files=['sub-01_raw.fif', 'sub-02_raw.fif'],
   ...     output_dir='output/preprocessing_authors'
   ... )

3. CHECK THE OUTPUT
   - Review logs in: eeg_pipeline/preprocessing_authors/logs/
   - Review QC plots in: output/preprocessing_authors/qc/
   - Find processed data in: output/preprocessing_authors/
"""

# ============================================================================
# PIPELINE STEPS AT A GLANCE
# ============================================================================

PIPELINE_OVERVIEW = {
    'Step 1': {
        'name': 'Common Average Reference',
        'file': '00_common_average_reference.py',
        'what_it_does': 'Re-references all channels to their average',
        'output': '{subject}_car.fif'
    },
    'Step 2': {
        'name': 'Identify Noisy Channels',
        'file': '01_identify_noisy_channels.py',
        'what_it_does': 'Finds bad channels via variance detection + visualization',
        'output': '{subject}_noisy_channels.json + plots'
    },
    'Step 3': {
        'name': 'Interpolate Bad Channels',
        'file': '02_interpolate_bad_channels.py',
        'what_it_does': 'Reconstructs bad channels from neighboring channels',
        'output': '{subject}_interpolated.fif'
    },
    'Step 4': {
        'name': 'Downsample',
        'file': '03_downsample.py',
        'what_it_does': 'Reduces sampling rate from 2048 Hz to 256 Hz',
        'output': '{subject}_downsampled.fif'
    },
    'Step 5': {
        'name': 'Epoch (3 Phases)',
        'file': '04_epoch.py',
        'what_it_does': 'Splits data into Decision, Response, Feedback epochs',
        'output': '{subject}_{decision/response/feedback}-epo.fif'
    },
    'Step 6': {
        'name': 'Baseline Correction & Binning',
        'file': '05_baseline_correction_binning.py',
        'what_it_does': 'Applies baseline (-200-0ms) and bins into 250ms windows',
        'output': '{subject}_{phase}_binned-epo.fif + metadata'
    }
}

# ============================================================================
# KEY FEATURES
# ============================================================================

FEATURES = [
    "✓ Exact implementation of Moerel et al. (2025) methodology",
    "✓ NO FILTERING (preserves signal integrity)",
    "✓ Common Average Reference (CAR)",
    "✓ Automated + visual noisy channel detection",
    "✓ Channel interpolation using spherical spline",
    "✓ Optimized downsampling (2048→256 Hz)",
    "✓ Three-phase epoching (Decision/Response/Feedback)",
    "✓ 250 ms time binning for statistical analysis",
    "✓ Comprehensive logging and QC plots",
    "✓ Fully documented with references"
]

# ============================================================================
# IMPORTANT: WHY NO FILTERING?
# ============================================================================

"""
The authors explicitly state:
    "We did not apply filtering, as this has been shown to cause artefacts 
     or temporally smear the signal"

This is based on peer-reviewed literature:
- Delorme (2023)
- Grootswagers et al. (2017)
- van Driel et al. (2021)

If you need filtering for your analysis, apply it AFTER the main preprocessing
but keep the authors' version unfiltered for reproducibility.
"""

# ============================================================================
# TASK DESIGN CONTEXT
# ============================================================================

EXPERIMENT_DESIGN = {
    'task': 'Rock-Paper-Scissors (competitive)',
    'participants': 62 + ' (31 pairs)',
    'games_per_pair': 480,
    'phases_per_game': 3,
    'phase_durations': {
        'Decision': '2 seconds',
        'Response': '2 seconds',
        'Feedback': '1 second'
    },
    'total_trial_duration': '5 seconds',
    'eeg_system': 'BioSemi Active-Two',
    'n_channels': 64,
    'original_sampling_rate': '2048 Hz',
    'electrode_system': '10-20 international'
}

# ============================================================================
# OUTPUT STRUCTURE
# ============================================================================

OUTPUT_STRUCTURE = """
output/preprocessing_authors/
├── sub-01_car.fif                    ← Step 1 output (CAR reference)
├── sub-01_interpolated.fif           ← Step 3 output (interpolation)
├── sub-01_downsampled.fif            ← Step 4 output (256 Hz)
├── sub-01_decision-epo.fif           ← Step 5 output (Decision phase)
├── sub-01_response-epo.fif           ← Step 5 output (Response phase)
├── sub-01_feedback-epo.fif           ← Step 5 output (Feedback phase)
├── sub-01_decision_binned-epo.fif    ← Step 6 output (binned)
├── sub-01_response_binned-epo.fif    ← Step 6 output (binned)
├── sub-01_feedback_binned-epo.fif    ← Step 6 output (binned)
├── qc/                               ← Quality control files
│   ├── channel_variances_plot.png
│   ├── sub-01_noisy_channels.json
│   └── ... (more QC outputs)
└── logs/                             ← Processing logs
    └── pipeline_YYYYMMDD_HHMMSS.log
"""

# ============================================================================
# COMMON QUESTIONS
# ============================================================================

FAQ = {
    'Q: Do I need filtering?': 
        'A: No. Authors explicitly avoided filtering. Filter afterwards if needed.',
    
    'Q: Can I modify the time bins?':
        'A: Yes, change TIME_BINNING in config_authors.py (default: 250 ms)',
    
    'Q: How many noisy channels should I expect?':
        'A: Typically 0-5 per subject. Review visualizations in qc/ directory.',
    
    'Q: What if my data has different event markers?':
        'A: Modify the event_id dictionary in 04_epoch.py',
    
    'Q: How long does processing take?':
        'A: ~5-10 minutes per subject depending on computer speed',
    
    'Q: Can I use this with other EEG systems?':
        'A: Yes, but verify channel count and sampling rate match expectations',
    
    'Q: Where should I look if something fails?':
        'A: Check eeg_pipeline/preprocessing_authors/logs/pipeline_*.log'
}

# ============================================================================
# FILES IN THIS FOLDER
# ============================================================================

FILES_REFERENCE = """
00_common_average_reference.py          - Step 1: CAR
01_identify_noisy_channels.py           - Step 2: Channel detection
02_interpolate_bad_channels.py          - Step 3: Interpolation
03_downsample.py                        - Step 4: Downsampling
04_epoch.py                             - Step 5: Epoching
05_baseline_correction_binning.py       - Step 6: Baseline & binning
master_pipeline_authors.py              - Main orchestration script
config_authors.py                       - Configuration file
__init__.py                             - Python package initialization
README.md                               - Detailed documentation
QUICKSTART.md                           - This file
"""

# ============================================================================
# REFERENCES
# ============================================================================

REFERENCE = """
Primary Paper:
Moerel D., Grootswagers T., Chin J.L.L., Ciardo F., Nijhuis P., 
Quek G.L., Smit S., Varlet M. (2025)
"Neural decoding of competitive decision-making in Rock-Paper-Scissors"
bioRxiv preprint. https://doi.org/10.1101/2025.01.09.632285

Tools Used:
- MNE-Python: https://mne.tools/
- NumPy, SciPy, Pandas
- Matplotlib for visualizations
"""

# ============================================================================
# NEXT STEPS
# ============================================================================

NEXT_STEPS = """
After running the preprocessing pipeline:

1. INSPECT QC PLOTS
   - Review eeg_pipeline/preprocessing_authors/logs/
   - Check output/preprocessing_authors/qc/ for channel statistics

2. VERIFY YOUR DATA
   - Check that correct number of epochs were created
   - Verify no unexpected bad channels were flagged
   - Review log file for any warnings

3. DECODING ANALYSIS
   - The next step is multivariate decoding (as per authors' analysis)
   - Use the binned epoch files as input
   - See authors' original code for CoSMoMVPA implementation

4. KEEP RECORDS
   - Store the preprocessing log file
   - Save QC plots for your record
   - Document any manual channel selections

5. REPRODUCIBILITY
   - All processing steps are logged
   - Parameters are configured in config_authors.py
   - Share config + logs for full reproducibility
"""

if __name__ == "__main__":
    print("""
╔════════════════════════════════════════════════════════════════════════════╗
║                                                                            ║
║          PREPROCESSING PIPELINE: MOEREL ET AL. (2025)                     ║
║          Rock-Paper-Scissors EEG Hyperscanning Study                      ║
║                                                                            ║
║  📁 Location: eeg_pipeline/preprocessing_authors/                         ║
║  📊 Output:   output/preprocessing_authors/                               ║
║                                                                            ║
║  ⚠️  No filtering applied (as per authors' methodology)                   ║
║  ✓  Full preprocessing pipeline documented and implemented                ║
║  ✓  Ready for analysis with decoding methods                             ║
║                                                                            ║
╚════════════════════════════════════════════════════════════════════════════╝
""")
