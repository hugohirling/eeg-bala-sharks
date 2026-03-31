"""
Configuration for the sanity-check suite.

REASONING:
- Purpose: provide a single configuration module for sanity-check paths, step mappings, and suite-level defaults.
- Reproducibility: centralizing step filenames and orchestration metadata avoids drift between runner scripts and individual step entrypoints.
- Parameter notes: this file intentionally stores constants and mappings, while CLI parsing helpers live in a separate module.
"""

from __future__ import annotations

import sys
from pathlib import Path

from matplotlib.colors import LinearSegmentedColormap


BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
SANITY_DIR = BASE_DIR / "sanity_checks"
SCRIPTS_DIR = SANITY_DIR / "scripts"

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))


DEFAULT_VIS_SUBJECT_LIMIT = 2
DEFAULT_PERSONS = ("P1", "P2")


STEP_CHECK_FILES = {
    "00": "sc_00_downsample.py",
    "01": "sc_01_split_players.py",
    "02": "sc_02_rename_montage.py",
    "03": "sc_03_bad_channels_detect.py",
    "04": "sc_04_interpolate.py",
    "05": "sc_05_filter.py",
    "06": "sc_06_ica.py",
    "07": "sc_07_epoch.py",
    "08": "sc_08_behavioral.py",
    "09": "sc_09_tf.py",
    "10": "sc_10_neuralDecoding.py",
}


COMBINED_MODE_STEPS = {"00", "01", "02", "03", "04", "07"}


PROGRESSION_SCRIPT = "sc_08_pipeline_progression_plots.py"
ERG_CHANNEL_PLOTS_SCRIPT = "sc_erg_channel_plots.py"
ERG_EOG_VIABILITY_SCRIPT = "sc_erg_eog_viability.py"
PLOT_PREPROCESSED_SCRIPT = "sc_plot_preprocessed_data.py"


VIZ_NEUTRAL = {
    "text_dark": "#111111",
    "text_mid": "#444444",
    "line_mid": "#4d4d4d",
    "line_soft": "#777777",
    "marker_edge": "#555555",
    "box_edge_light": "#cccccc",
    "point_gray": "#7a7a7a",
    "panel_gray": "#f5f5f5",
    "white": "#ffffff",
    "black": "#000000",
}


DOWNSAMPLE_VIZ = {
    "before": "#1f77b4",
    "after": "#ff7f0e",
    "before_edge": "darkblue",
    "after_edge": "darkred",
    "before_face": "#f0f8ff",
    "after_face": "#fff5e6",
    "expected": "#9ecae1",
}


SPLIT_VIZ = {
    "p1": "#1f77b4",
    "p2": "#ff7f0e",
    "expected": "#9ecae1",
}


MONTAGE_VIZ = {
    "sensor": "#1f77b4",
    "before": "#ff7f0e",
    "after": "#2ca02c",
    "pass": "#2ca02c",
    "fail": "#d62728",
}


BAD_CHANNEL_VIZ = {
    "good": "black",
    "manual": "#ff7f0e",
    "auto": "#d62728",
    "overlap": "#1f77b4",
    "threshold": "#e377c2",
    "neighbor": "#4d4d4d",
    "good_strong": "#2ca02c",
    "good_hist": "#1f77b4",
    "after_hist": "#ff7f0e",
    "bad_edge": "darkred",
    "bad_box_face": "#ffe6e6",
    "bad_box_edge": "#d62728",
    "good_box_face": "#e6ffe6",
    "good_box_edge": "#2ca02c",
}


INTERPOLATE_VIZ = {
    "before": "#d62728",
    "after": "#006d2c",
    "neighbor": "#4d4d4d",
    "delta": "#5a5a5a",
    "good": "black",
    "before_soft": "#ff7f0e",
    "after_soft": "#1f77b4",
    "after_strong": "#2ca02c",
    "trace_even_face": "#f5f5f5",
    "trace_odd_face": "#ffffff",
    "note_face": "#ffffff",
    "note_edge": "#777777",
    "success_face": "#e6ffe6",
    "success_edge": "#2ca02c",
    "info_face": "#e8f4f8",
    "info_edge": "#1f77b4",
}


FILTER_VIZ = {
    "before": "#c44e52",
    "after": "#2a7f62",
    "mean": "#111111",
    "delta": "#4d4d4d",
    "threshold": "#7a7a7a",
    "passband": "#d8ead3",
    "stopband": "#f6d6d6",
}


ICA_VIZ = {
    "before": "#c44e52",
    "after": "#2a7f62",
    "mean": "#111111",
    "bad": "#e74c3c",
    "good": "#3498db",
    "zero_line": "#777777",
    "passband": "#d8ead3",
    "stopband": "#f6d6d6",
}


EPOCH_VIZ = {
    "continuous": "#1f77b4",
    "epoched": "#ff7f0e",
    "example_title": "#2ca02c",
    "event_palette": ["#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3", "#a6d854", "#ffd92f"],
    "event_face": "#f5f5f5",
    "example_face": "#f0fff0",
    "continuous_face": "#f0f8ff",
    "epoched_face": "#fff5e6",
}


FIXED_ABS_SCALE_UV = 50.949631

SCALP_CMAP = LinearSegmentedColormap.from_list(
    "scalp_positions",
    [
        "#c51b7d",
        "#7b2cbf",
        "#2251d1",
        "#1192e8",
        "#00bfa5",
        "#24a148",
        "#ff6b35",
        "#c51b7d",
    ],
    N=256,
)
