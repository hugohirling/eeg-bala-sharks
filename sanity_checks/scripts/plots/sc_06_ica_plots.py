"""
Internal plot helpers for step 06 (ICA).

Entry point:
    python sanity_checks/scripts/sc_06_ica.py --mode viz
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import mne
import numpy as np
from mne_icalabel import label_components

from preprocessing import config
from helpers.sc_config import ICA_VIZ, VIZ_NEUTRAL
from helpers.sc_plot_io import save_figure
from helpers.sc_signal import compute_psd as _compute_psd

COLOR_BEFORE = ICA_VIZ["before"]
COLOR_AFTER = ICA_VIZ["after"]
COLOR_BAD = ICA_VIZ["bad"]
COLOR_GOOD = ICA_VIZ["good"]
COLOR_PASSBAND = ICA_VIZ["passband"]
COLOR_STOPBAND = ICA_VIZ["stopband"]
PSD_FMAX = 60.0


def _normalize_iclabel_name(label):
    return str(label).strip().lower().replace("_", " ")


def collect_component_metadata(raw, ica):
    artifact_labels = {_normalize_iclabel_name(label) for label in config.ICA_ARTIFACT_LABELS}
    metadata = []

    try:
        raw_crop = raw.copy().load_data().filter(l_freq=1.0, h_freq=None).crop(tmin=0.0, tmax=min(60.0, float(raw.times[-1])))
        label_result = label_components(raw_crop, ica, method=config.ICA_LABEL_METHOD)
        labels = [_normalize_iclabel_name(label) for label in label_result["labels"]]
        probabilities = [float(probability) for probability in label_result["y_pred_proba"]]
    except Exception:
        labels = ["unknown"] * int(ica.n_components)
        probabilities = [float("nan")] * int(ica.n_components)

    for idx in range(int(ica.n_components)):
        label = labels[idx] if idx < len(labels) else "unknown"
        probability = probabilities[idx] if idx < len(probabilities) else float("nan")
        predicted_artifact = label in artifact_labels and np.isfinite(probability) and probability >= float(config.ICA_LABEL_MIN_PROBA)
        removed = idx in set(int(component) for component in ica.exclude)
        metadata.append(
            {
                "index": idx,
                "label": label,
                "probability": probability,
                "removed": removed,
                "predicted_artifact": predicted_artifact,
            }
        )
    return metadata


def component_summary_text(component_meta, limit=5):
    removed = [
        f"C{item['index']} {item['label']} ({item['probability']:.2f})"
        for item in component_meta
        if item["removed"]
    ]
    if not removed:
        return "No components excluded"
    if len(removed) <= limit:
        return "; ".join(removed)
    return "; ".join(removed[:limit]) + f"; ... +{len(removed) - limit} more"


def plot_component_topomaps(ica, raw_before, component_meta, subject_id, person, output_dir):
    if ica.n_components < 1:
        return

    n_comps_show = int(ica.n_components)
    n_rows = (n_comps_show + 3) // 4

    fig, axes = plt.subplots(n_rows, 4, figsize=(16.5, n_rows * 3.3), constrained_layout=True)
    axes = axes.reshape(1, -1) if n_rows == 1 else axes.reshape(n_rows, -1)
    axes = axes.flatten()

    for idx in range(n_comps_show):
        ax = axes[idx]
        meta = component_meta[idx]
        is_bad = meta["removed"]
        probability_text = f"p={meta['probability']:.2f}" if np.isfinite(meta["probability"]) else "p=n/a"
        decision_text = "excluded" if is_bad else "kept"

        try:
            ica.plot_components(picks=[idx], show=False, axes=ax, sphere="eeglab")
        except Exception:
            try:
                ica.plot_components(picks=[idx], show=False, axes=ax, sphere="auto")
            except Exception:
                ax.text(0.5, 0.5, f"Component {idx}\n(plot failed)", ha="center", va="center", fontsize=10)
                ax.set_xlim(-0.1, 0.1)
                ax.set_ylim(-0.1, 0.1)

        title_color = COLOR_BAD if is_bad else COLOR_GOOD
        title_weight = "bold" if is_bad else "normal"
        ax.set_title(f"C{idx} | {meta['label']}\n{probability_text} -> {decision_text}", fontsize=9.5, fontweight=title_weight, color=title_color)

        if is_bad:
            for spine in ax.spines.values():
                spine.set_edgecolor(COLOR_BAD)
                spine.set_linewidth(2.5)

    for idx in range(n_comps_show, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(
        f"sub-{subject_id} {person} - ICA Component Topomaps\n"
        f"Total: {ica.n_components} components | Excluded (red): {len(ica.exclude)} | Kept (blue): {ica.n_components - len(ica.exclude)}",
        fontsize=14,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.012,
        "Component = one independent source estimated from the mixed EEG sensors. Topomap = how strongly that source projects to each scalp channel.\n"
        f"Reasoning rule: exclude if ICLabel predicts one of {', '.join(config.ICA_ARTIFACT_LABELS)} with p >= {float(config.ICA_LABEL_MIN_PROBA):.2f}.",
        ha="center",
        va="bottom",
        fontsize=9,
        color=VIZ_NEUTRAL["text_mid"],
    )

    plot_path = save_figure(fig, output_dir, f"sub-{subject_id}_{person}_ica_components.png", dpi=150)
    plt.close(fig)
    print(f"  [OK] Component topomaps saved: {plot_path.name}")


def plot_variance_explained(ica, raw_data, component_meta, subject_id, person, output_dir):
    if ica.n_components < 1:
        return

    try:
        raw_crop = raw_data.copy().crop(tmin=0, tmax=min(60.0, raw_data.times[-1]))
        raw_crop.load_data()
        sources = ica.get_sources(raw_crop).get_data()
        explained_var = np.var(sources, axis=1)
        explained_var = explained_var / np.sum(explained_var)
    except Exception:
        explained_var = np.abs(ica.mixing_matrix_).mean(axis=0)
        explained_var = explained_var / explained_var.sum()

    fig, ax = plt.subplots(figsize=(14, 6), constrained_layout=True)
    x = np.arange(ica.n_components)
    colors = [COLOR_BAD if item["removed"] else COLOR_GOOD for item in component_meta]
    ax.bar(x, explained_var * 100, color=colors, alpha=0.75, edgecolor="black", linewidth=1.0)

    ax.set_xlabel("Component Index", fontsize=12, fontweight="bold")
    ax.set_ylabel("Relative component energy (%)", fontsize=12, fontweight="bold")
    ax.set_title("Component energy estimated from ICA sources", fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.28, linestyle=":")
    ax.set_ylim(bottom=0.0)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLOR_GOOD, edgecolor="black", label=f"Good ({ica.n_components - len(ica.exclude)})"),
        Patch(facecolor=COLOR_BAD, edgecolor="black", label=f"Bad ({len(ica.exclude)})"),
    ]
    ax.legend(handles=legend_elements, fontsize=10, loc="upper right")

    for item in component_meta:
        if item["removed"]:
            ax.text(item["index"], explained_var[item["index"]] * 100 + 0.35, item["label"], ha="center", va="bottom", rotation=70, fontsize=8, color=COLOR_BAD, fontweight="bold")

    bad_variance = sum(explained_var[ica.exclude] * 100) if len(ica.exclude) > 0 else 0.0
    fig.suptitle(
        f"sub-{subject_id} {person} - ICA Component Energy\n"
        f"Relative source energy total: 100% | Excluded components account for: {bad_variance:.1f}%",
        fontsize=13,
        fontweight="bold",
    )
    ax.text(
        0.01,
        0.98,
        "A component is one latent source recovered by ICA. Higher bars mean that source carries more variance in the ICA activations.\n"
        "Red labels mark components excluded because ICLabel classified them as artifacts above threshold.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        color=VIZ_NEUTRAL["text_mid"],
        bbox={"facecolor": VIZ_NEUTRAL["white"], "edgecolor": VIZ_NEUTRAL["box_edge_light"], "alpha": 0.9, "boxstyle": "round,pad=0.4"},
    )

    plot_path = save_figure(fig, output_dir, f"sub-{subject_id}_{person}_ica_variance.png", dpi=150)
    plt.close(fig)
    print(f"  [OK] Variance explained saved: {plot_path.name}")


def plot_bad_component_timeseries(ica, raw, component_meta, subject_id, person, output_dir):
    if len(ica.exclude) == 0:
        return

    sources = ica.get_sources(raw).get_data()
    sfreq = float(raw.info["sfreq"])
    times = raw.times
    t_end = min(2.0, float(times[-1]))
    n_samples = int(t_end * sfreq)

    n_bad = len(ica.exclude)
    n_cols = min(3, n_bad)
    n_rows = (n_bad + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4.5, n_rows * 3.2), constrained_layout=True)
    if n_bad == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    else:
        axes = axes.reshape(n_rows, -1)
    axes = axes.flatten()

    for plot_idx, comp_idx in enumerate(ica.exclude):
        ax = axes[plot_idx]
        component_data = sources[comp_idx, :n_samples]
        component_times = times[:n_samples]
        meta = component_meta[int(comp_idx)]
        probability_text = f"p={meta['probability']:.2f}" if np.isfinite(meta["probability"]) else "p=n/a"

        ax.plot(component_times, component_data, color=COLOR_BAD, linewidth=1.2)
        ax.axhline(0, color=ICA_VIZ["zero_line"], linestyle="--", linewidth=0.8)
        ax.set_title(f"C{int(comp_idx)} | {meta['label']}\n{probability_text} -> excluded", fontsize=10, fontweight="bold", color=COLOR_BAD)
        ax.set_ylabel("Amplitude (a.u.)")
        ax.grid(alpha=0.22, linestyle=":")

        if plot_idx >= (n_rows - 1) * n_cols:
            ax.set_xlabel("Time (s)")

    for idx in range(n_bad, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(f"sub-{subject_id} {person} - Bad ICA Component Time Series\nFirst 2 seconds | {n_bad} component(s) marked for removal", fontsize=13, fontweight="bold")
    fig.text(
        0.5,
        0.015,
        "Component time series = the activation of one ICA source over time. These traces are subtracted from the sensor data when the component is excluded.",
        ha="center",
        va="bottom",
        fontsize=9,
        color=VIZ_NEUTRAL["text_mid"],
    )

    plot_path = save_figure(fig, output_dir, f"sub-{subject_id}_{person}_ica_bad_timeseries.png", dpi=150)
    plt.close(fig)
    print(f"  [OK] Bad component time series saved: {plot_path.name}")


def plot_psd_comparison_ica(raw_before, raw_after, component_meta, subject_id, person, duration, output_dir):
    freqs_before, mean_before, psd_before, raw_before_eeg = _compute_psd(raw_before, duration)
    freqs_after, mean_after, psd_after, _ = _compute_psd(raw_after, duration)
    if freqs_before is None or freqs_after is None or raw_before_eeg is None:
        return

    mean_before_db = 10.0 * np.log10(np.maximum(mean_before, np.finfo(float).tiny))
    mean_after_db = 10.0 * np.log10(np.maximum(mean_after, np.finfo(float).tiny))
    psd_before_db = 10.0 * np.log10(np.maximum(psd_before, np.finfo(float).tiny))
    psd_after_db = 10.0 * np.log10(np.maximum(psd_after, np.finfo(float).tiny))

    fig, (ax_before, ax_after) = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True, sharey=True)
    y_min = float(min(np.min(psd_before_db), np.min(psd_after_db))) - 2.0
    y_max = float(max(np.max(psd_before_db), np.max(psd_after_db))) + 2.0

    ax_before.axvspan(0.0, config.FREQ_LOWER, color=COLOR_STOPBAND, alpha=0.65, lw=0)
    ax_before.axvspan(config.FREQ_LOWER, config.FREQ_UPPER, color=COLOR_PASSBAND, alpha=0.8, lw=0)
    ax_before.axvspan(config.FREQ_UPPER, PSD_FMAX, color=COLOR_STOPBAND, alpha=0.65, lw=0)
    ax_before.plot(freqs_before, mean_before_db, color=COLOR_BEFORE, linewidth=2.5, label="Mean PSD")
    for ch_idx in range(psd_before_db.shape[0]):
        ax_before.plot(freqs_before, psd_before_db[ch_idx], color=COLOR_BEFORE, linewidth=0.7, alpha=0.15)
    ax_before.set_xlim(0.0, PSD_FMAX)
    ax_before.set_ylim(y_min, y_max)
    ax_before.set_title("Before ICA", fontsize=12, fontweight="bold")
    ax_before.set_xlabel("Frequency (Hz)")
    ax_before.set_ylabel("PSD per channel (dB)")
    ax_before.grid(alpha=0.22, linestyle=":")
    ax_before.legend(fontsize=10)

    ax_after.axvspan(0.0, config.FREQ_LOWER, color=COLOR_STOPBAND, alpha=0.65, lw=0)
    ax_after.axvspan(config.FREQ_LOWER, config.FREQ_UPPER, color=COLOR_PASSBAND, alpha=0.8, lw=0)
    ax_after.axvspan(config.FREQ_UPPER, PSD_FMAX, color=COLOR_STOPBAND, alpha=0.65, lw=0)
    ax_after.plot(freqs_after, mean_after_db, color=COLOR_AFTER, linewidth=2.5, label="Mean PSD")
    for ch_idx in range(psd_after_db.shape[0]):
        ax_after.plot(freqs_after, psd_after_db[ch_idx], color=COLOR_AFTER, linewidth=0.7, alpha=0.15)
    ax_after.set_xlim(0.0, PSD_FMAX)
    ax_after.set_title("After ICA", fontsize=12, fontweight="bold")
    ax_after.set_xlabel("Frequency (Hz)")
    ax_after.grid(alpha=0.22, linestyle=":")
    ax_after.legend(fontsize=10)

    passband_before = np.mean(psd_before[:, (freqs_before >= config.FREQ_LOWER) & (freqs_before <= config.FREQ_UPPER)])
    passband_after = np.mean(psd_after[:, (freqs_after >= config.FREQ_LOWER) & (freqs_after <= config.FREQ_UPPER)])
    passband_change = ((passband_after - passband_before) / passband_before * 100.0) if passband_before > 0 else np.nan

    fig.suptitle(
        f"sub-{subject_id} {person} - ICA PSD Comparison\n"
        f"Passband (1-40 Hz) change: {passband_change:+.1f}% | Removed: {component_summary_text(component_meta)}",
        fontsize=13,
        fontweight="bold",
    )

    plot_path = save_figure(fig, output_dir, f"sub-{subject_id}_{person}_ica_psd_comparison.png", dpi=150)
    plt.close(fig)
    print(f"  [OK] PSD comparison saved: {plot_path.name}")

