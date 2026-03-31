# Sanity Check Discussion: Preprocessing and Analysis Pipeline

This document records the quality and plausibility of the visualizations and outputs for each step of the analysis pipeline.

---

## Step 00: Downsample (2048 Hz → 200 Hz)

### Visualized Plots
- `*_downsample_timeseries_comparison.png` — Time Series Original vs. Downsampled
- `*_downsample_psd_comparison.png` — Power Spectral Density Before/After
- `*_downsample_statistics_comparison.png` — Amplitude & File Size

### Expected Effects
| Metric | Expected | Observation |
|--------|----------|-------------|
| Sampling Rate Factor | ~10x Reduction (2048→200 Hz) | ✓ |
| File Size | ~10% of Original | ✓ |
| High-Frequency Power (>100 Hz) | Completely removed | ✓ |
| Low-Frequency Power (<100 Hz) | Preserved | ✓ |
| Waveform Quality | Recognizable but with fewer details | ✓ |

### Discussion
Downsampling reduces the data footprint without significant information loss for EEG analyses in the target frequency bands. The PSD comparisons show that after downsampling, frequencies above the new Nyquist frequency (100 Hz) are successfully removed.

**This seems correct because** the goal of this step is not to alter the biological signal content, but to drastically reduce the data volume while preserving interpretable EEG dynamics. The choice of 200 Hz is motivated by the need to make subsequent machine learning and ICA steps memory-efficient, without destroying the low- and mid-frequency bands required for structural analyses.

---

## Step 01: Split Players (Sub-Level → Per-Player)

### Visualized Plots
- `*_split_players_data_summary.png` — P1 vs P2 Data Distribution
- `*_split_players_amplitude_dist.png` — Amplitude Histograms per Player

### Expected Effects
| Metric | Expected | Observation |
|--------|----------|-------------|
| Data Amount per Player | Approximately split 50/50 | ✓ |
| Channel Count | Identical per player | ✓ |
| Amplitudes | Similar distributions for good channels | ✓ |

### Discussion
The split function isolates the multi-player recording into two dedicated, single-player arrays. The nearly identical data sizes and channel counts confirm that both players were recorded symmetrically. Minor amplitude differences are expected and result from differing electrode impedances or physical placement.

**This seems correct because** the split should only perform a structural separation of the arrays; duration, sampling rate, and status information should remain largely identical. **This is strange if** P1/P2 prefixes or incorrect channels remain after the split, as it would cause subsequent scripts to leak data across players.

---

## Step 02: Rename & Set Montage

### Visualized Plots
- `*_montage_topomap.png` — Sensor layout with all channel names
- `*_montage_channel_mapping.png` — Channel names Before (BioSemi) → After (10-20)
- `*_montage_coverage_stats.png` — Standard 10-20 System Coverage

### Expected Effects
| Metric | Expected | Observation |
|--------|----------|-------------|
| Channel Names | BioSemi → Standard 10-20 | ✓ |
| Electrode Positions | Present & spatially plausible | ✓ |
| Montage Type | Biosemi 64 (projected to standard) | ✓ |
| Standard 10-20 Channels | At least 12-20 of the 64 mapped properly | ✓ |

### Discussion
Setting the montage ensures that every EEG channel holds a strict 3D physical coordinate mapped to the skull. This is essential for spatial analyses (e.g., topomaps, source localization). The topomap validation shows all 64 channels evenly distributed across the scalp surface.

**This seems correct because** renaming and applying a montage does not artificially generate new electrical data; it simply makes the existing continuous arrays spatially interpretable, allowing future spatial interpolation algorithms to function properly.

---

## Step 03: Bad Channels Detect

### Visualized Plots
- `*_bad_channels_topomap.png` — Topomap with flagged Bad Channels
- `*_bad_channels_amplitudes.png` — Amplitude Before/After, Bad-Channel markers
- `*_bad_channels_qc_metrics.png` — QC statistics overview

### Expected Effects
| Metric | Expected | Observation |
|--------|----------|-------------|
| Bad Channel Count | 1-5 of 64 channels (typical scale) | ✓ |
| Bad Channel Positions | Mostly peripheral or erratically noisy channels | ✓ |
| Bad Channel Amplitudes | Visibly higher variance than intact channels | ✓ |
| Info Registration | Properly logged into MNE `raw.info['bads']` | ✓ |

### Discussion
Automated bad channel detection flags electrodes suffering from poor scalp contact, severe motion artifacts, or technical malfunctions. Typically, frontal (Fp1, Fp2) or peripheral electrodes are most vulnerable in behavioral tasks.

**This seems correct because** individual problematic electrodes distinctly break away from the spatial variance distribution of their neighbors. **This is strange if** a massive portion of the array (e.g., 30+ channels) is flagged, which would indicate a global recording failure rather than isolated electrode malfunction. 

---

## Step 04: Interpolate Bad Channels

### Visualized Plots
- `*_interpolate_montage_comparison.png` — Sensor layout Before/After
- `*_interpolate_timeseries.png` — Time series of the interpolated channels
- `*_interpolate_statistics.png` — Amplitude & Channel Status recovery

### Expected Effects
| Metric | Expected | Observation |
|--------|----------|-------------|
| Interpolation Method | Spherical Spline | ✓ |
| Bad Channel Integrity | Reconstructed based on local neighbors | ✓ |
| Interpolated Amplitude | Visually smooth and similar to neighbors | ✓ |
| Artifact Generation | Minimal to zero high-frequency injections | ✓ |

### Discussion
Spherical spline interpolation mathematically reconstructs broken signals using weighted data from surrounding intact channels. The time series comparisons confirm that the newly generated arrays lack artificial jitter and seamlessly integrate into the global signal phase. 

**This seems correct because** interpolation leaves the total channel count entirely unchanged, simply replacing corrupted variances with topologically valid approximations.

---

## Step 05: Filter (Bandpass 1-40 Hz)

### Visualized Plots
- `*_filter_psd_comparison.png` — Power Spectral Density Before/After

### Expected Effects
| Metric | Expected | Observation |
|--------|----------|-------------|
| Target Frequency Band | 1-40 Hz passed seamlessly | ✓ |
| Sub-1 Hz Power | Strongly attenuated (drift removed) | ✓ |
| Over-40 Hz Power | Strongly attenuated (noise removed) | ✓ |
| Waveform Shape | Smoother, lacking 50Hz/60Hz line hum | ✓ |

### Discussion
A bandpass filter restricts analyses strictly to the cognitive frequency bands of interest (Theta, Alpha, Beta) relevant for motor execution and decision-making. 

**This parameter choice is justified because** a 1.0 - 40.0 Hz bandpass aggressively suppresses slow electrical drift while naturally neutralizing generic upper-frequency bounds (like 50 Hz power line noise) without needing a phase-altering notch filter. **This seems correct because** the stopband power drops off a cliff in the PSD plot, while passband power is perfectly preserved.

---

## Step 06: ICA (Artifact Removal)

### Visualized Plots
- `*_ica_components.png` — Component mappings and raw amplitude reduction

### Expected Effects
| Metric | Expected | Observation |
|--------|----------|-------------|
| Fitted ICA Components | Typically 30-40 for an interpolated 64 array | ✓ |
| Identified EOG Components | 1-4 components (Blinks + Eye movements) | ✓ |
| Amplitude Reduction | Modest (~10-20%) noise reduction | ✓ |
| Output Topography | Cleaner frontal lobe activity post-cleanup | ✓ |

### Discussion
Independent Component Analysis strictly dissociates ocular artifacts (blinks, saccades) from genuine cerebral EEG signals. 

**This parameter choice is justified because** components are not dropped manually or subjectively; they are algorithmically routed through `mne-icalabel`, utilizing a neural network to systematically calculate biological artifact probabilities. **This seems correct because** the rejected components display distinct dipole topographies indicative of eye movement, and their removal does not aggressively flatline the raw overarching signal.

---

## Step 07: Epoching

### Visualized Plots
- `*_epoch_overview.png` — Sample Epochs and Event distributions

### Expected Effects
| Metric | Expected | Observation |
|--------|----------|-------------|
| Isolated Event Types | Decision, Response, Feedback | ✓ |
| Balanced Distributions | Similar epoch counts per player/condition | ✓ |
| Epoch Boundaries | E.g., -0.2s before to 2.0s after trigger | ✓ |

### Discussion
Epoching slices the continuous datasets into strictly defined, event-locked time matrices. Given the design of the Rock-Paper-Scissors paradigm, large symmetrical distributions of epochs matching the behavioral TSV logs must be generated.

**This seems correct because** the extracted time windows and baseline zones create the precise, uniform structures required for advanced trial-based machine learning. **This is strange if** extreme mismatches occur in epoch counts, pointing directly to a failure in TSV trigger parsing rather than an actual behavioral anomaly.

---

## Step 08: Behavioral Processing & Markov Modeling

### Visualized Plots
- `*_sanity_check_markov_matrix.png` — Markov Transition Probability Matrix
- `*_sanity_check_behavioral_variance.png` — Group-Level Variance of Heuristics

### Expected Effects
| Metric | Expected | Observation |
|--------|----------|-------------|
| Matrix Row Sums | Exactly 1.0 (100%) per operational state | ✓ |
| Randomness Baseline | Deviations away from pure 33.3% uniformity | ✓ |
| Trial Integrity | No missing or corrupted logic loops | ✓ |

### Discussion
Inspecting the behavioral data verifies that the raw BIDS `events.tsv` outputs were mathematically structured into valid conditional probability architectures reflecting participant heuristics.

**This seems correct because** the rows of the 3x3 Markov matrix scale perfectly to exactly 1.0. This mathematical absolute proves that relative transition frequencies (Win-Stay, Lose-Shift) are properly normalized and safely avoid distortions caused by slightly varying match lengths across different subject dyads.

---

## Step 09: Time-Frequency Representation (ERSP/TFR)

### Visualized Plots
- `*_sanity_check_tfr_ersp.png` — TFR Heatmap (Alpha-band ERD)

### Expected Effects
| Metric | Expected | Observation |
|--------|----------|-------------|
| Metric Scaling | Log-ratio baseline correction (Decibels) | ✓ |
| Baseline Stability | Flat ~0 dB variance preceding 0.0s | ✓ |
| Spectral Resolution | 4-40 Hz well resolved via Morlet wavelets | ✓ |
| Biological Signature | Clear Alpha (8-13 Hz) Event-Related Desynchronization | ✓ |

### Discussion
This check validates the execution of the dynamic Morlet wavelet convolution and the resulting spatial mean computation spanning all 64 channels. 

**This seems correct because** the resting state variance remains stable immediately prior to the stimulus trigger, and the anticipated Alpha-band desynchronization (highlighted by distinct deep blue ERD clusters) physiologically manifests throughout the active decision phase. **This is strange if** absolute raw power values were utilized without log-ratio baselining; the natural 1/f inverse drop-off inherent to raw EEG signals would completely mask and drown out these cognitive fluctuations.

---

## Step 10: Advanced Neural Decoding (TGM & Cross-Brain MVPA)

### Visualized Plots
- `*_decoding_tgm_matrix.png` — Temporal Generalization Matrix
- `*_decoding_cross_brain.png` — Cross-Brain Model (P1 predicting P2)

### Expected Effects
| Metric | Expected | Observation |
|--------|----------|-------------|
| Decoding Baseline | Hovers strictly near chance (~33.3% for 3 classes) | ✓ |
| Generalization Structure | Extended off-diagonal activation contours | ✓ |
| Anti-Data Leakage | Absence of flat 100% predictive spikes | ✓ |

### Discussion
These visualizations ensure the machine learning backend (5-Fold Stratified Cross Validation, Standard Scaler, and Linear Discriminant Analysis) generates stable, unbiased test boundaries, and correctly drops structural triggers.

**This seems correct because** the cross-brain spatial representations remain fiercely tethered near theoretical chance (33.3%) throughout the trial timeline, perfectly reflecting the inherently unpredictable nature of a functional zero-sum competitive game. **This is strange if** the cross-brain MVPA inexplicably decoded opponent maneuvers at 90% accuracy prior to execution; such a spike would inherently confirm catastrophic data leakage (e.g., failing to drop non-biological hardware synchronization triggers before model training).

---