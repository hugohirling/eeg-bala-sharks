"""
Plotting utilities for the EEG decoding pipeline.

This module encapsulates all of the code related to visualizing
and saving decoding accuracy results.  By keeping plotting logic
separate we keep the main pipeline script focused on data handling
and computation.

Functions:
- plot_decoding_accuracy
- save_plot_data
- load_plot_data
- plot_comparison
- plot_only

Dependencies are limited to numpy, matplotlib and the shared
`terminal_log` helper from logging_utils.

Plotting was created using AI assistance (VS Code Copilot).
"""

import os
import pickle
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets
from scipy import interpolate

from logging_utils import terminal_log


# -----------------------------------------------------------------------------
# Serialization helpers
# -----------------------------------------------------------------------------

def save_plot_data(times: np.ndarray, scores: np.ndarray, filepath: str):
    """
    Persist times/scores pair to disk for later plotting.

    Args:
        times: 1‑D array of time points
        scores: 1‑D array of decoding accuracies
        filepath: destination pickle filename
    """
    terminal_log(f"Saving plot data to {filepath}...")
    with open(filepath, "wb") as f:
        pickle.dump({"times": times, "scores": scores}, f)
    terminal_log("Plot data saved successfully.")


def load_plot_data(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a previously saved time/scores pair.

    Returns a tuple (times, scores).
    """
    terminal_log(f"Loading plot data from {filepath}...")
    with open(filepath, "rb") as f:
        data = pickle.load(f)
    terminal_log("Plot data loaded successfully.")
    return data["times"], data["scores"]


# -----------------------------------------------------------------------------
# Visualization helpers
# -----------------------------------------------------------------------------

def plot_decoding_accuracy(
    times: np.ndarray,
    scores: np.ndarray,
    title: str,
    show_plot: bool = True,
    n_bins: int = 50,
):
    """
    Create a smooth accuracy curve with variance band and interactive slider.

    The raw scores are binned along the time axis, and within each
    bin we compute mean/min/max.  A cubic spline is used to draw a
    smooth line and shaded region representing the min–max range.

    A slider allows adjusting the number of bins (granularity) in real-time.

    Args:
        times: time vector corresponding to `scores`
        scores: accuracy values
        title: figure title
        show_plot: if False the figure is not displayed (useful for
            headless runs)
        n_bins: initial number of bins used for aggregation
    """
    if not show_plot:
        return

    # Create figure with subplots for plot and slider
    fig, (ax, ax_slider) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [4, 1]})

    # Initial n_bins
    initial_n_bins = n_bins

    # Function to update plot
    def update_plot(n_bins_val):
        ax.clear()
        n_bins_val = min(n_bins_val, len(times) // 10 or 1)
        idx_groups = np.array_split(np.arange(len(times)), n_bins_val)

        bin_times = []
        bin_means = []
        bin_mins = []
        bin_maxs = []
        for grp in idx_groups:
            if grp.size > 0:
                bin_times.append(times[grp].mean())
                bin_means.append(scores[grp].mean())
                bin_mins.append(scores[grp].min())
                bin_maxs.append(scores[grp].max())
        bin_times = np.array(bin_times)
        bin_means = np.array(bin_means)
        bin_mins = np.array(bin_mins)
        bin_maxs = np.array(bin_maxs)

        # smooth curves via interpolation
        f_mean = interpolate.interp1d(
            bin_times, bin_means, kind="cubic", fill_value="extrapolate"
        )
        f_min = interpolate.interp1d(
            bin_times, bin_mins, kind="cubic", fill_value="extrapolate"
        )
        f_max = interpolate.interp1d(
            bin_times, bin_maxs, kind="cubic", fill_value="extrapolate"
        )

        smooth_t = np.linspace(times.min(), times.max(), 300)
        smooth_mean = f_mean(smooth_t)
        smooth_min = f_min(smooth_t)
        smooth_max = f_max(smooth_t)

        ax.fill_between(smooth_t, smooth_min, smooth_max, alpha=0.3, color="blue", label="Variance (min-max)")
        ax.plot(smooth_t, smooth_mean, linewidth=2.5, color="blue", label="Mean accuracy")
        ax.scatter(bin_times, bin_means, s=20, color="darkblue", zorder=5, alpha=0.6)
        ax.axhline(1 / 3, color="red", linestyle="--", linewidth=1.5, label="Chance level (33%)")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Decoding Accuracy")
        ax.set_title(title)
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
        fig.canvas.draw_idle()

    # Slider
    slider = widgets.Slider(ax_slider, 'Number of Bins', 10, 100, valinit=initial_n_bins, valstep=1)
    slider.on_changed(update_plot)

    # Initial plot
    update_plot(initial_n_bins)

    plt.tight_layout()
    plt.show()

    # print simple statistics
    terminal_log("\n" + "=" * 40)
    terminal_log("STATISTICS:")
    terminal_log("=" * 40)
    terminal_log(f"Mean accuracy: {scores.mean():.4f}")
    terminal_log(f"Std accuracy: {scores.std():.4f}")
    terminal_log(f"Max accuracy: {scores.max():.4f}")
    terminal_log(f"Min accuracy: {scores.min():.4f}")
    terminal_log(f"Time of max: {times[scores.argmax()]:.3f}s")


def plot_comparison(
    times1: np.ndarray,
    scores1: np.ndarray,
    times2: np.ndarray,
    scores2: np.ndarray,
    title: str = "Comparison",
    show_plot: bool = True,
):
    """
    Plot two decoding curves on the same axes with interactive slider and report stats.

    A slider allows adjusting the number of points in the common time axis.
    """
    if not show_plot:
        return

    # Create figure with subplots for plot and slider
    fig, (ax, ax_slider) = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={'height_ratios': [4, 1]})

    # Initial number of points
    initial_n_points = len(times1)

    # Function to update plot
    def update_plot(n_points):
        ax.clear()
        common = np.linspace(0, min(times1[-1], times2[-1]), n_points)
        sc1 = np.interp(common, times1, scores1)
        sc2 = np.interp(common, times2, scores2)

        ax.plot(common, sc1, linewidth=2, color="blue", label="Option 1", marker='o', markersize=3, alpha=0.7)
        ax.plot(common, sc2, linewidth=2, color="red", label="Option 2", marker='s', markersize=3, alpha=0.7)
        ax.axhline(1 / 3, color="k", linestyle="--", label="Chance level (33%)")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Decoding Accuracy")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.canvas.draw_idle()

    # Slider
    slider = widgets.Slider(ax_slider, 'Number of Points', 50, 500, valinit=initial_n_points, valstep=10)
    slider.on_changed(update_plot)

    # Initial plot
    update_plot(initial_n_points)

    plt.tight_layout()
    plt.show()

    # Compute stats on full resolution
    common = np.linspace(0, min(times1[-1], times2[-1]), len(times1))
    sc1 = np.interp(common, times1, scores1)
    sc2 = np.interp(common, times2, scores2)

    terminal_log("\n" + "=" * 60)
    terminal_log("STATISTICS:")
    terminal_log("=" * 60)
    terminal_log(f"Option 1 — Mean: {sc1.mean():.4f}, Std: {sc1.std():.4f}")
    terminal_log(f"Option 2 — Mean: {sc2.mean():.4f}, Std: {sc2.std():.4f}")
    terminal_log(f"Difference: {abs(sc1.mean() - sc2.mean()):.4f}")


def plot_only(option: str, plot_file: str = "plot_data"):
    """
    Entry point for "plot-only" CLI mode.
    """
    terminal_log("=" * 60)
    terminal_log("PLOT-ONLY MODE: Loading saved plot data")
    terminal_log("=" * 60)

    if option == "compare":
        f1 = f"{plot_file}_option1.pkl"
        f2 = f"{plot_file}_option2.pkl"
        if not os.path.exists(f1) or not os.path.exists(f2):
            terminal_log(f"✗ Error: Missing plot data files {f1} or {f2}. Run with --option compare first.")
            return
        t1, s1 = load_plot_data(f1)
        t2, s2 = load_plot_data(f2)
        plot_comparison(t1, s1, t2, s2)

    else:
        if not os.path.exists(plot_file):
            terminal_log(f"✗ Error: {plot_file} not found.")
            return
        t, s = load_plot_data(plot_file)
        plot_decoding_accuracy(t, s, "Decoding accuracy")
