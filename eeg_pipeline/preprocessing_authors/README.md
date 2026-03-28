# Preprocessing Authors Pipeline
## Following Moerel et al. (2025)

Implementation of the preprocessing pipeline described in:
**"Neural decoding of competitive decision-making in Rock-Paper-Scissors"**
Moerel D., Grootswagers T., Chin J.L.L., Ciardo F., Nijhuis P., Quek G.L., Smit S., Varlet M.

---

## Overview

This preprocessing pipeline implements the exact methodology described in the authors' supplementary materials. The pipeline has been specifically designed to preserve signal integrity by avoiding filtering, which the authors noted "has been shown to cause artefacts or temporally smear the signal."

### Key Features
- ✅ **No Filtering**: Explicitly avoids filtering per Moerel et al. (2025) methodology
- ✅ **Common Average Reference (CAR)**: Standard re-referencing approach
- ✅ **Robust Channel Interpolation**: Identifies and interpolates noisy channels
- ✅ **Optimized Downsampling**: Reduces from 2048 Hz to 256 Hz
- ✅ **Multi-Phase Epoching**: Separate epochs for Decision, Response, and Feedback phases
- ✅ **Baseline Correction & Binning**: 250 ms time bins for statistical analysis

---

## Pipeline Steps

### Step 1: Common Average Reference (CAR)
**File:** `00_common_average_reference.py`

Re-references the data to the common average across all EEG channels.

```python
raw.set_eeg_reference(ref_channels='average')
```

**Input:** Raw EEG data (2048 Hz, 64 channels)  
**Output:** CAR-referenced data  

---

### Step 2: Identify Noisy Channels
**File:** `01_identify_noisy_channels.py`

Following Moerel et al.: "We identified noisy channels through visual inspection"

This step provides:
- Automated variance-based detection (z-score > 3.0)
- Channel variance visualization
- Raw data visualization for visual inspection
- Noisy channels log for next steps

**Input:** CAR-referenced data  
**Output:** 
- `{subject}_noisy_channels.json` - List of identified noisy channels
- QC plots for review

---

### Step 3: Interpolate Bad Channels
**File:** `02_interpolate_bad_channels.py`

Interpolates noisy channels using neighboring channels.

Following Moerel et al.: 
"We interpolated noisy channels based on neighbouring channels, using the ft_channelrepair function with a distance measure of 0.5 cm"

**Method:** MNE's spherical spline interpolation (equivalent to FieldTrip's ft_channelrepair)

**Input:** CAR-referenced data + noisy channels list  
**Output:** Interpolated data

---

### Step 4: Downsample
**File:** `03_downsample.py`

Downsamples from 2048 Hz to 256 Hz.

Following Moerel et al.: "We then down-sampled the data to 256 Hz"

**Input:** Interpolated data  
**Output:** Downsampled data (256 Hz)

---

### Step 5: Epoch into Three Phases
**File:** `04_epoch.py`

Creates separate epochs for three task phases:

Following Moerel et al.: 
"Each game consisted of three phases: Decision (2 s), Response (2 s) and Feedback (1 s)"

**Epoching windows:**
- **Decision phase:** -200 ms to 2000 ms (from Decision screen onset)
- **Response phase:** -200 ms to 2000 ms (from Response screen onset)
- **Feedback phase:** -200 ms to 1000 ms (from Feedback screen onset)

**Baseline correction:** -200 ms to 0 ms for each epoch

**Input:** Downsampled continuous data  
**Output:** Three sets of epochs (decision, response, feedback)

---

### Step 6: Baseline Correction & Time Binning
**File:** `05_baseline_correction_binning.py`

Applies baseline correction and bins data into 250 ms time bins.

Following Moerel et al.: 
"We applied baseline corrections for each separate epoch, using the window from -200 ms to 0 ms"
"we averaged the resulting data into 250 ms time bins, resulting in a total of 20 time bins for the 0 to 5000 ms time-course"

**Input:** Epoched data  
**Output:** 
- Binned epochs for each phase
- Metadata file with binning information

---

## Usage

### Simple Usage (Master Pipeline)

```python
from preprocessing_authors import AuthorsPreprocessingPipeline

# Initialize pipeline
config = {'output_dir': '/path/to/output/preprocessing_authors'}
pipeline = AuthorsPreprocessingPipeline(config)

# Run for single subject
pipeline.run_pipeline('sub-01', '/path/to/sub-01_raw.fif')
```

### Using Individual Steps

```python
# Step 1: CAR
from preprocessing_authors import common_average_reference
common_average_reference.main('input_raw.fif', 'output_car.fif')

# Step 2: Identify noisy channels
from preprocessing_authors import identify_noisy_channels
identify_noisy_channels.main('input_car.fif', './qc', 'sub-01')

# Continue with other steps...
```

### Command Line

```bash
# Run from workspace root
python -c "
from eeg_pipeline.preprocessing_authors import main
main(
    subjects_input_files=['path/to/sub-01_raw.fif', 'path/to/sub-02_raw.fif'],
    output_dir='output/preprocessing_authors',
    subject_ids=['sub-01', 'sub-02']
)
"
```

---

## Output Structure

```
output/preprocessing_authors/
├── sub-01_car.fif                          # Step 1 output
├── sub-01_interpolated.fif                 # Step 3 output
├── sub-01_downsampled.fif                  # Step 4 output
├── sub-01_decision-epo.fif                 # Step 5 outputs
├── sub-01_response-epo.fif
├── sub-01_feedback-epo.fif
├── sub-01_decision_binned-epo.fif         # Step 6 outputs
├── sub-01_response_binned-epo.fif
├── sub-01_feedback_binned-epo.fif
├── qc/
│   └── sub-01_noisy_channels.json          # Step 2 output
└── logs/
    └── pipeline_YYYYMMDD_HHMMSS.log       # Pipeline execution log
```

---

## Important Notes

### Why No Filtering?

The authors explicitly chose not to apply any frequency filtering:

> "We did not apply filtering, as this has been shown to cause artefacts or temporally smear the signal"

This decision is based on:
- Delorme, 2023
- Grootswagers et al., 2017
- van Driel et al., 2021

### Baseline Correction

Baseline correction is applied separately for each of the three phases, using the window from **-200 ms to 0 ms** relative to each phase's screen onset.

### Time Binning

Data is binned into **250 ms** time bins, which results in **20 time bins** for the full trial duration (0-5000 ms).

### Task Design

The experiment consisted of:
- **31 pairs** of participants (62 total)
- **480 games** per pair
- **64-channel EEG** (BioSemi Active-Two)
- **2048 Hz** original sampling rate
- **64 electrodes** following the international 10-20 system

---

## References

**Primary Reference:**
Moerel D., Grootswagers T., Chin J.L.L., Ciardo F., Nijhuis P., Quek G.L., Smit S., Varlet M. (2025). 
Neural decoding of competitive decision-making in Rock-Paper-Scissors. 
bioRxiv preprint. https://doi.org/10.1101/2025.01.09.632285

**Tools Used:**
- MNE-Python (for EEG processing)
- FieldTrip (original methodology reference)
- CoSMoMVPA (for decoding analysis, mentioned in authors' methods)

---

## Quality Checks

The pipeline includes quality control checks:
- Channel variance visualization
- Noisy channel detection and logging
- Epoch counts and metadata
- Processing log files with timestamps

Review the `qc/` directory for detailed quality check outputs.

---

## Requirements

- Python 3.8+
- mne-python >= 1.0
- numpy
- matplotlib (for visualization)

---

## Author

Implementation based on Moerel et al. (2025) methodology  
Created: January 2025
