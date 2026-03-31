"""
Signal-processing helpers shared across sanity-check scripts.

REASONING:
- Purpose: avoid duplicated EEG crop + PSD calculations across multiple step scripts.
- Reproducibility: one implementation keeps Welch parameters consistent between checks.
"""

from __future__ import annotations

import mne
import numpy as np
from mne.time_frequency import psd_array_welch


def prepare_eeg_crop(raw, duration_s):
    """Return an EEG-only cropped copy or None if data is too short."""
    eeg_picks = mne.pick_types(raw.info, eeg=True, exclude=[])
    if len(eeg_picks) == 0:
        return None

    t_end = min(float(duration_s), float(raw.times[-1]))
    if t_end <= 1.0:
        return None

    return raw.copy().pick(eeg_picks).crop(tmin=0.0, tmax=t_end)


def compute_psd(raw, duration_s, fmax=60.0):
    """Compute mean/channel PSD on a short EEG crop using Welch."""
    raw_eeg = prepare_eeg_crop(raw, duration_s)
    if raw_eeg is None:
        return None, None, None, None

    data = raw_eeg.get_data()
    sfreq = float(raw_eeg.info["sfreq"])
    n_fft = min(int(round(sfreq * 4.0)), data.shape[1])
    if n_fft < 32:
        return None, None, None, None

    n_per_seg = min(n_fft, data.shape[1])
    n_overlap = n_per_seg // 2

    psd, freqs = psd_array_welch(
        data,
        sfreq=sfreq,
        fmin=0.0,
        fmax=fmax,
        n_fft=n_fft,
        n_per_seg=n_per_seg,
        n_overlap=n_overlap,
        average="mean",
        verbose=False,
    )
    return freqs, np.mean(psd, axis=0), psd, raw_eeg

