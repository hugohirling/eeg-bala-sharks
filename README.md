# EEG Hyperscanning: Competitive Decision-Making (Bala Sharks)

**Authors:** Bahar Kalyoncu, Hugo Hirling, Ayush Batra
**Dataset:** ds006761 (Rock-Paper-Scissors)

## Project Overview
This repository contains a EEG pipeline designed to process and analyze 64-channel hyperscanning data. Our group investigated the cognitive dynamics of adversarial, zero-sum competition by mapping brain activity temporally, spectrally, and behaviorally. 

We specifically engineered our architecture to mitigate the severe RAM bottlenecks inherent in processing dual-brain (128-channel) simultaneous recordings, ensuring the codebase can be easily reproduced on standard hardware.

## Installation & Reproducibility
To ensure seamless reproducibility across different hardware environments, we have locked our software versions. We highly recommend running this within a standard Python virtual environment (Python 3.10+).

1. Clone this repository.
2. Ensure your terminal is at the root of the project.
3. Install the exact package versions using pip:
   ```bash
   pip install -r requirements.txt

```
eeg-bala-sharks/
├─ eeg_pipeline/         # Core Python scripts (preprocessing, PLV, encoding/decoding)
├─ sanity_checks/        # Jupyter notebooks dedicated to visual diagnostic sanity checks
├─ milestones/           # Legacy developmental checkpoints
├─ requirements.txt      # Dependency version control
├─ input                 # Input files
├─ output                # Output files
├─ paths.py              # File to provide paths to input/output
└─ README.md             # Project documentation
```

## Abstracted Preprocessing Pipeline
To mitigate memory bottlenecks and ensure data quality, our preprocessing pipeline executes the following sequential logic:

1. Data Downsampling: Reducing the native 2048 Hz sampling rate to 200 Hz.
2. Separating by Player: Isolate Player 1 and Player 2 data streams.
3. Renaming & Montage: Renaming the channels to 10-20 labels and adding 3D locations.
4. Interpolation of Bad Channels: Automated robust Z-score flagging and spherical spline interpolation to repair broken electrodes.
5. Bandpass Filtering: 1.0 Hz - 40.0 Hz.
6. ICA- Based Artifact Removal: Automated biological artifact rejection via mne-icalabel to remove EOG/ECG interference.
7. Epoching: Slicing the continuous signal based on customized TSV event triggers into structured decision-making windows.

## Core Methodologies
Our analysis pipeline is divided into five distinct methodological approaches:

1. Interbrain Synchrony (Phase-Locking Value): Measuring dynamic temporal coupling and shared neural states between competing pairs.
2. Time-Resolved Decoding (Standard MVPA): Single-subject classification using Linear Discriminant Analysis (LDA) to track the neural representation of a participants own choices over the decision timeline.
3. Advanced Neural Decoding: Extending standard decoding by introducing Temporal Generalization Matrices (TGM) to evaluate cognitive state stability, alongside Cross-Brain MVPA to test for opponent predictability.
4. Behavioral Heuristics and Markov Modeling: Quantifying non-random human biases using transition probabilities.
5. Time-Frequency Representations (ERSP): Utilizing complex Morlet wavelets to map Alpha-band Event-Related Desynchronization (ERD) prior to motor execution.

## Executing the Analysis
1. Place the BIDS-formatted dataset ds006761 in the structured data directory.(Option: Use the download_dataset.py file)
2. Run the preprocessing scripts found in the codebase directory.
3. Execute the respective decoding, PLV, TFR, and behavioral scripts. 
4. View the sanitychecks notebooks to reproduce the intermediate visualizations and interpret the data quality.
