# Sanity Checks for EEG Pipeline

This directory contains sanity check scripts for every step of the EEG preprocessing and analysis pipeline. The checks verify metadata, data quality, and plausible value ranges.

## Why these checks are important

This sanity check suite is not only intended for technical troubleshooting but also acts as **assessable documentation** for the pipeline decisions.

- **Code readability & documentation:** The scripts now contain motivations, parameter notes, and interpretation guides documented directly in the code.
- **Reproducibility & modularity:** Relevant thresholds come from `preprocessing/config.py` or are explicitly documented in the script; the CSV summaries make runs comparable.
- **Sanity checks & discussion:** Console and CSV outputs contain status messages alongside short sentences like `This seems correct because ...` or `This is strange because ...`.
- **Result interpretation:** The visualizations are structured as Before/After comparisons so that visible changes can be directly argued.

The central guiding question for each step is therefore:

> Does this processing step change exactly what it is supposed to change, and does it leave everything else unchanged?

## Available Sanity Checks

The **primary entry points** are the step scripts `sc_00_...` to `sc_10_...`. For the paired steps, these scripts support the modes `--mode check`, `--mode viz`, or `--mode both`.

| Step | Check Script | Visualization Script | Description |
|------|--------|--------|-----------|
| 00 | `sc_00_downsample.py` | optional via `--mode viz` | Downsampling: Sampling rate, Data reduction, Time-Series & PSD |
| 01 | `sc_01_split_players.py` | optional via `--mode viz` | Split Players: P1 vs P2 data distribution, duration, channels |
| 02 | `sc_02_rename_montage.py` | optional via `--mode viz` | Rename & Montage: Channel name mapping, Sensor layout 2D Topomap |
| 03 | `sc_03_bad_channels.py` | optional via `--mode viz` | Bad Channels: Topomap with markers, Amplitude before/after |
| 04 | `sc_04_interpolate.py` | optional via `--mode viz` | Interpolation: Time series of interpolated channels, Amplitude, Montage |
| 05 | `sc_05_filter.py` | (in sc_05_filter.py) | Bandpass Filter (1-40 Hz): PSD before/after comparison |
| 06 | `sc_06_ica.py` | (in sc_06_ica.py) | ICA Artifact Removal: Components, EOG detection |
| 07 | `sc_07_epoch.py` | optional via `--mode viz` | Epoching: Event distribution, Example epochs, PSD comparison, Baseline |
| 08 | `sc_08_behavioral.py` | (in sc_08_behavioral.py) | Behavioral: Markov Transition Probability Matrices, Variance |
| 09 | `sc_09_time_freq.py` | (in sc_09_time_freq.py) | Time-Frequency: Morlet wavelets, Alpha-band ERD heatmaps |
| 10 | `sc_10_advanced_mvpa.py`| (in sc_10_advanced_mvpa.py)| Decoding: Temporal Generalization Matrices, Cross-Brain MVPA |
| All | — | `sc_pipeline_progression.py` | **Overall View**: GFP + PSD across all stages (Original → ICA) |

## Usage

### Combined Step-Scripts for Check and Visualization

The preferred usage for the paired steps is via a joint step-script:

python sanity_checks/scripts/sc_00_downsample.py --mode check
python sanity_checks/scripts/sc_00_downsample.py --mode viz --subjects 01,02 --duration 30
python sanity_checks/scripts/sc_00_downsample.py --mode both --subjects 01,02 --duration 30
python sanity_checks/scripts/sc_03_bad_channels.py --mode both --subjects 01,02
python sanity_checks/scripts/sc_04_interpolate.py --mode viz --subjects 01,02 --duration 30
python sanity_checks/scripts/sc_07_epoch.py --mode viz --subjects 01,02
python sanity_checks/scripts/sc_08_behavioral.py --mode both
python sanity_checks/scripts/sc_09_time_freq.py --mode both
python sanity_checks/scripts/sc_10_advanced_mvpa.py --mode both

### Automatic Quality Checks (Text Output)

python sanity_checks/scripts/sc_05_filter.py
python sanity_checks/scripts/sc_06_ica.py
python sanity_checks/scripts/sc_07_epoch.py

### Overview across all Preprocessing Stages

python sanity_checks/scripts/sc_pipeline_progression.py

### Quickly Visualize Processed Data

python sanity_checks/scripts/sc_plot_preprocessed_data.py --subjects 01,02 --step 06 --duration 60

### Run all Sanity Checks sequentially (Automated Master Run)

python sanity_checks/run_all_checks.py

- **Notebooks are located under** `sanity_checks/notebooks/`
- **Python-Checks are located under** `sanity_checks/scripts/`

## Outputs

### Visualization Plots (output/qc/)

Every pipeline step generates specific visualization plots:

**Step 00 - Downsample:**
- `sub-XX_P1_downsample_timeseries_comparison.png` — Time series Original vs. Downsampled
- `sub-XX_P1_downsample_psd_comparison.png` — Frequency spectrum comparison
- `sub-XX_P1_downsample_statistics_comparison.png` — Amplitude & File size

**Step 01 - Split Players:**
- `sub-XX_split_players_data_summary.png` — P1 vs P2 data distribution (Duration, Channels, Size)
- `sub-XX_split_players_amplitude_dist.png` — Amplitude histogram per Player

**Step 02 - Rename & Montage:**
- `sub-XX_P1_montage_topomap.png` — Sensor-Layout Topomap
- `sub-XX_P1_montage_channel_mapping.png` — Channel names Before/After
- `sub-XX_P1_montage_coverage_stats.png` — Standard 10-20 System Coverage

**Step 03 - Bad Channels Detection:**
- `sub-XX_P1_bad_channels_topomap.png` — Topomap with Marked Bad-Channels
- `sub-XX_P1_bad_channels_amplitudes.png` — Amplitude comparison Good vs. Bad
- `sub-XX_P1_bad_channels_qc_metrics.png` — QC metrics (optional)

**Step 04 - Interpolation:**
- `sub-XX_P1_interpolate_montage_comparison.png` — Sensor-Layout Before/After
- `sub-XX_P1_interpolate_timeseries.png` — Time series of interpolated channels
- `sub-XX_P1_interpolate_statistics.png` — Amplitude before/after Interpolation

**Step 05, 06 & Pipeline:**
- `sub-XX_P1_filter_psd_comparison.png` — Filter effect (PSD)
- `sub-XX_P1_ica_detailed_comparison.png` — ICA Amplitude reduction
- `sub-XX_P1_pipeline_progression_gfp_psd.png` — GFP + PSD across all stages

**Step 07 - Epoching:**
- `sub-XX_P1_epoch_event_distribution.png` — Histogram: Epoch count per Event type
- `sub-XX_P1_epoch_examples.png` — Example epochs with Baseline window
- `sub-XX_P1_epoch_psd_comparison.png` — PSD comparison: continuous vs epoched
- `sub-XX_P1_epoch_statistics.png` — Metadata summary 

**Step 08 - Behavioral:**
- `sub-XX_sanity_check_markov_matrix.png` — Markov transition matrices
- `group_sanity_check_behavioral_variance.png` — Behavioral heuristic variance

**Step 09 - Time-Frequency:**
- `sub-XX_sanity_check_tfr_ersp.png` — Alpha-band ERD heatmaps

**Step 10 - Advanced MVPA:**
- `sub-XX_decoding_tgm_matrix.png` — TGM models
- `sub-XX_decoding_cross_brain.png` — Cross-Brain accuracy traces

### Console Output

- Detailed validation results with Pass and Warning/Error
- Statistic summaries per Subject/Player
- CSV export structured with `Status`, `Category`, `Message`, `Rationale`, and `ParameterNote`

## Hints

- Sanity Checks load data with `preload=False` to save memory.
- For large time windows, limited samples are used (e.g., first 60-120 seconds).
- Checks are kept short so they can run quickly after their respective pipeline step.

## How the discussion should be formulated

For the grading, a simple Pass is often not enough. The discussion in `SANITY_CHECK_DISCUSSION.md` should contain at least one of the following sentence types per step:

- `This seems correct because ...`
- `This is strange because ...`
- `This parameter choice is justified because ...`

Examples:
- `This seems correct because the downsampled file keeps the same duration while reducing the sampling rate and estimated size.`
- `This is strange because the bad-channel fraction is unusually high and may indicate a recording-wide quality problem.`
- `This parameter choice is justified because the 1-40 Hz filter keeps conventional EEG bands while suppressing slow drifts and high-frequency noise.`

## Grading Strategy (20% "Sanity Checks & Visualizations & Discussion")

This sanity check suite addresses all steps with comprehensive visualizations:

### Coverage 
- Steps 00-10: Dedicated visualization scripts with Before/After comparisons
- Overall View: `sc_pipeline_progression.py` shows GFP + PSD across all stages

### Visualization Quality 
- Time Series: Waveform comparisons (Downsample, Interpolation)
- Frequency-Domain: PSD comparisons (Filter, Downsample, ERSP)
- Topomaps: Sensor layout & Channel markers
- Heatmaps: Transition matrices, TGM accuracy

### Modularity
- Each step has its own script -> easy to maintain/expand
- Unified code structure (Argparse, Logging, Output-Dirs)

### Discussion
Additionally, a `SANITY_CHECK_DISCUSSION.md` explains for each step:
- What does the plot show?
- Are the visible changes expected/plausible?
- Which metrics are monitored?
- Why are exactly these parameters useful?