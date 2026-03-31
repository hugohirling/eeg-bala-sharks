# Comments in this file were added with the help of GitHub Copilot (GPT-5.3-Codex).
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from rich.console import Console
from rich.live import Live
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

CURRENT_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = CURRENT_DIR.parent
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from preprocessing import config

console = Console()
progress = Progress(
    TextColumn("[progress.description]{task.description}"),
    BarColumn(),
    MofNCompleteColumn(),
    TextColumn("•"),
    TimeElapsedColumn(),
    TextColumn("•"),
    TimeRemainingColumn(),
    console=console,
    transient=False,
    refresh_per_second=10,
)

RESP_CODE_TO_NAME = {1: "rock", 2: "paper", 3: "scissors"}
TARGET_CHOICES = {
    "current_self": "Current own decision",
    "previous_self": "Previous own decision",
    "previous_other": "Previous opponent decision",
    "current_other": "Current opponent decision",
}


def _resolve_subjects(subjects_arg: str | None) -> list[str]:
    """
    Resolves the CLI subject argument into a normalized list of subject IDs.

    If no explicit argument is provided, all configured subjects are returned.

    Args:
        subjects_arg (str | None): Comma-separated subject IDs or None.

    Returns:
        list[str]: Subject IDs to process.
    """
    if subjects_arg:
        return [part.strip() for part in subjects_arg.split(",") if part.strip()]
    return list(config.SUBJECTS)


def _resolve_targets(targets_arg: str | None) -> list[str]:
    """
    Resolves and validates requested decoding targets from CLI input.

    Args:
        targets_arg (str | None): Comma-separated target keys or None.

    Returns:
        list[str]: Valid target keys to decode.

    Raises:
        ValueError: If one or more targets are not supported.
    """
    if not targets_arg:
        return ["current_self"]

    targets = [part.strip() for part in targets_arg.split(",") if part.strip()]
    invalid = sorted(set(targets) - set(TARGET_CHOICES))
    if invalid:
        raise ValueError(f"Unsupported targets: {invalid}. Choose from {sorted(TARGET_CHOICES)}")
    return targets


def _get_events_tsv_path(subject_id: str) -> Path:
    """
    Builds the expected BIDS events TSV path for one subject.

    Args:
        subject_id (str): Subject identifier without the sub- prefix.

    Returns:
        Path: Absolute/relative path to the subject events TSV.
    """
    return (
        Path(config.BIDS_ROOT)
        / f"sub-{subject_id}"
        / "eeg"
        / f"sub-{subject_id}_task-RPS_events.tsv"
    )


def _get_epoch_path(subject_id: str, person: str) -> Path:
    """
    Builds the expected preprocessed epoch-file path for one subject/player.

    Args:
        subject_id (str): Subject identifier without the sub- prefix.
        person (str): Player label, typically P1 or P2.

    Returns:
        Path: Path to the epoch FIF file.
    """
    return Path(config.OUTPUT_DIR) / f"sub-{subject_id}_{person}_epoch.fif"


def _player_response_column(person: str) -> str:
    """
    Maps a player identity to the corresponding response column in events TSV.

    Args:
        person (str): Player label, typically P1 or P2.

    Returns:
        str: Column name (player1_resp or player2_resp).

    Raises:
        ValueError: If the configured player prefix is unexpected.
    """
    prefix = config.PLAYER_PREFIX_MAP[person]
    if prefix.startswith("1"):
        return "player1_resp"
    if prefix.startswith("2"):
        return "player2_resp"
    raise ValueError(f"Unexpected player prefix for {person}: {prefix}")


def _other_player_response_column(person: str) -> str:
    """
    Returns the opponent response column for a given player.

    Args:
        person (str): Player label, typically P1 or P2.

    Returns:
        str: Opponent response column name.
    """
    current_col = _player_response_column(person)
    return "player2_resp" if current_col == "player1_resp" else "player1_resp"


def _load_events_df(subject_id: str) -> pd.DataFrame:
    """
    Loads the subject-level events TSV as a pandas dataframe.

    Args:
        subject_id (str): Subject identifier without the sub- prefix.

    Returns:
        pd.DataFrame: Events table for the subject.

    Raises:
        FileNotFoundError: If the events TSV does not exist.
    """
    events_path = _get_events_tsv_path(subject_id)
    if not events_path.exists():
        raise FileNotFoundError(f"Missing events file: {events_path}")
    return pd.read_csv(events_path, sep="\t")


def _load_decision_epochs(subject_id: str, person: str) -> mne.Epochs:
    """
    Loads preprocessed epochs and restricts them to the decision window.

    The method crops each epoch to 0.0-2.0 s and keeps EEG channels only,
    matching the assumptions of the searchlight decoding stage.

    Args:
        subject_id (str): Subject identifier without the sub- prefix.
        person (str): Player label, typically P1 or P2.

    Returns:
        mne.Epochs: Decision-window EEG epochs.

    Raises:
        FileNotFoundError: If the epoch file does not exist.
    """
    epoch_path = _get_epoch_path(subject_id, person)
    if not epoch_path.exists():
        raise FileNotFoundError(f"Missing epoch file: {epoch_path}")

    epochs = mne.read_epochs(str(epoch_path), preload=True)
    decision_epochs = epochs.copy().crop(tmin=0.0, tmax=2.0)
    decision_epochs.pick("eeg")
    return decision_epochs


def _previous_trial(labels: np.ndarray) -> np.ndarray:
    """
    Shifts labels by one trial to create previous-trial targets.

    The first trial is filled with -1 because no previous trial exists.

    Args:
        labels (np.ndarray): Current-trial labels.

    Returns:
        np.ndarray: Previous-trial labels with sentinel value on first trial.
    """
    shifted = np.full(labels.shape, fill_value=-1, dtype=int)
    if len(labels) > 1:
        shifted[1:] = labels[:-1]
    return shifted


def _load_target_labels(subject_id: str, person: str, target: str) -> np.ndarray:
    """
    Builds the label vector for a requested target definition.

    Supported targets include current/previous self and current/previous other.

    Args:
        subject_id (str): Subject identifier without the sub- prefix.
        person (str): Player label, typically P1 or P2.
        target (str): Target key to construct.

    Returns:
        np.ndarray: Label vector aligned to subject events.

    Raises:
        ValueError: If the target key is unsupported.
    """
    events_df = _load_events_df(subject_id)
    own_col = _player_response_column(person)
    other_col = _other_player_response_column(person)

    if target == "current_self":
        labels = events_df[own_col].to_numpy(dtype=int)
    elif target == "current_other":
        labels = events_df[other_col].to_numpy(dtype=int)
    elif target == "previous_self":
        labels = _previous_trial(events_df[own_col].to_numpy(dtype=int))
    elif target == "previous_other":
        labels = _previous_trial(events_df[other_col].to_numpy(dtype=int))
    else:
        raise ValueError(f"Unsupported target: {target}")

    return labels


def _make_bin_edges(times: np.ndarray, n_bins: int = 8) -> tuple[np.ndarray, list[tuple[float, float]]]:
    """
    Creates evenly spaced temporal bins and human-readable bin windows.

    Args:
        times (np.ndarray): Epoch time vector in seconds.
        n_bins (int): Number of bins to generate.

    Returns:
        tuple[np.ndarray, list[tuple[float, float]]]:
            - edges: Integer index edges for slicing time points.
            - labels: (start_s, end_s) windows for reporting/plot titles.
    """
    edges = np.linspace(0, len(times), n_bins + 1, dtype=int)
    labels = []
    for start, stop in zip(edges[:-1], edges[1:]):
        stop_idx = max(start, stop - 1)
        labels.append((float(times[start]), float(times[stop_idx])))
    return edges, labels


def _make_searchlight_clusters(info: mne.Info, n_neighbors: int) -> list[np.ndarray]:
    """
    Builds channel-wise spatial searchlight clusters from EEG positions.

    For each center channel, this function finds the nearest neighbors in
    Euclidean 3D sensor space and returns sorted channel-index clusters.

    Args:
        info (mne.Info): Channel metadata including sensor locations.
        n_neighbors (int): Number of nearest neighbors besides center channel.

    Returns:
        list[np.ndarray]: One integer index array per center channel.
    """
    positions = np.array([channel["loc"][:3] for channel in info["chs"]], dtype=float)
    clusters: list[np.ndarray] = []

    for center_idx in range(len(info["ch_names"])):
        center_pos = positions[center_idx]
        distances = np.linalg.norm(positions - center_pos, axis=1)
        ordered = np.argsort(distances)
        cluster = np.sort(ordered[: n_neighbors + 1])
        clusters.append(cluster)

    return clusters


def _prepare_topomap_info(info: mne.Info) -> mne.Info:
    """
    Prepares an EEG-only info object suitable for topomap rendering.

    If digitization points are missing, a BioSemi64 fallback montage is applied
    to ensure sensor positions are available for plotting/interpolation.

    Args:
        info (mne.Info): Original info object from epochs.

    Returns:
        mne.Info: EEG-only info with usable sensor geometry.

    Raises:
        RuntimeError: If no EEG channels are available.
    """
    info_plot = info.copy()
    eeg_picks = mne.pick_types(info_plot, eeg=True, exclude=[])
    if len(eeg_picks) == 0:
        raise RuntimeError("No EEG channels available for topomap plotting.")

    info_plot = mne.pick_info(info_plot, eeg_picks, copy=True)
    has_dig = info_plot.get("dig") is not None and len(info_plot["dig"]) > 0
    if not has_dig:
        # Match sanity-check fallback to a standard BioSemi64 head geometry.
        montage = mne.channels.make_standard_montage("biosemi64", head_size=0.105)
        info_plot.set_montage(montage, match_case=False, on_missing="ignore")
    return info_plot


def _prepare_subject_person(
    subject_id: str,
    person: str,
    target: str,
) -> tuple[np.ndarray, np.ndarray, mne.Info, np.ndarray]:
    """
    Loads and aligns feature/label data for one subject-player-target combination.

    The function trims feature and label lengths to a shared minimum and removes
    non-RPS labels so downstream 3-class decoding is valid.

    Args:
        subject_id (str): Subject identifier without the sub- prefix.
        person (str): Player label, typically P1 or P2.
        target (str): Target key (current/previous self/other).

    Returns:
        tuple[np.ndarray, np.ndarray, mne.Info, np.ndarray]:
            - X: Trial x channel x time feature array.
            - y: Filtered label vector.
            - info: Copy of epochs info.
            - times: Copy of epochs time vector.

    Raises:
        ValueError: If fewer than three classes remain after filtering.
    """
    epochs = _load_decision_epochs(subject_id, person)
    X = epochs.get_data(copy=True)
    y = _load_target_labels(subject_id, person, target)

    n = min(len(X), len(y))
    X = X[:n]
    y = y[:n]

    valid_mask = np.isin(y, [1, 2, 3])
    X = X[valid_mask]
    y = y[valid_mask]

    if len(np.unique(y)) < 3:
        raise ValueError(
            f"Not enough classes for sub-{subject_id} {person} target={target}. "
            f"Classes found: {sorted(set(y.tolist()))}"
        )

    return X, y, epochs.info.copy(), epochs.times.copy()


def _decode_searchlight_maps(
    X: np.ndarray,
    y: np.ndarray,
    clusters: list[np.ndarray],
    n_splits: int,
    random_state: int,
) -> np.ndarray:
    """
    Computes time-bin-by-channel searchlight decoding accuracy maps.

    Each channel's local spatial cluster is decoded independently per time bin.
    The result is a 2D matrix compatible with topographic plotting.

    Args:
        X (np.ndarray): Input features shaped (trials, channels, bins).
        y (np.ndarray): Class labels per trial.
        clusters (list[np.ndarray]): Searchlight channel-index clusters.
        n_splits (int): Number of stratified CV folds.
        random_state (int): Random seed for fold shuffling.

    Returns:
        np.ndarray: Accuracy map with shape (n_bins, n_channels).
    """
    clf = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("lda", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
        ]
    )
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    n_bins = X.shape[2]
    scores = np.zeros((n_bins, len(clusters)), dtype=float)

    for bin_idx in range(n_bins):
        X_bin = X[:, :, bin_idx]
        for channel_idx, cluster in enumerate(clusters):
            # Decode each searchlight neighborhood independently.
            X_cluster = X_bin[:, cluster]
            cv_scores = cross_val_score(clf, X_cluster, y, cv=cv, scoring="accuracy")
            scores[bin_idx, channel_idx] = float(np.mean(cv_scores))

    return scores


def _save_topomap_grid(
    data: np.ndarray,
    info: mne.Info,
    windows: list[tuple[float, float]],
    title: str,
    out_path: Path,
) -> Path:
    """
    Saves an 8-panel (2x4) topomap figure covering all 250 ms bins.

    The color scale is shared across panels for direct visual comparison.

    Args:
        data (np.ndarray): Accuracy map with shape (8, n_channels).
        info (mne.Info): EEG channel geometry.
        windows (list[tuple[float, float]]): Time windows for panel titles.
        title (str): Figure-level title.
        out_path (Path): Output PNG path.

    Returns:
        Path: Path to the saved figure.
    """
    vmin = float(np.min(data))
    vmax = float(np.max(data))
    fig, axes = plt.subplots(2, 4, figsize=(15, 7))
    axes = axes.ravel()
    image = None

    for idx, ax in enumerate(axes):
        try:
            image, _ = mne.viz.plot_topomap(
                data[idx],
                info,
                axes=ax,
                show=False,
                contours=0,
                cmap="viridis",
                vlim=(vmin, vmax),
                sphere="eeglab",
            )
        except Exception:
            image, _ = mne.viz.plot_topomap(
                data[idx],
                info,
                axes=ax,
                show=False,
                contours=0,
                cmap="viridis",
                vlim=(vmin, vmax),
                sphere="auto",
            )
        start_s, end_s = windows[idx]
        ax.set_title(f"{start_s:.2f}-{end_s:.2f} s")

    fig.suptitle(title)
    fig.subplots_adjust(left=0.04, right=0.90, top=0.90, bottom=0.05, wspace=0.18, hspace=0.22)
    cax = fig.add_axes([0.92, 0.16, 0.015, 0.68])
    colorbar = fig.colorbar(image, cax=cax)
    colorbar.set_label("Decoding accuracy")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _save_collapsed_topomaps(
    data: np.ndarray,
    info: mne.Info,
    title: str,
    out_path: Path,
) -> Path:
    """
    Saves two collapsed topomaps by averaging early and late decision bins.

    This creates one map for 0-1 s and one for 1-2 s to summarize temporal
    dynamics in a compact figure.

    Args:
        data (np.ndarray): Accuracy map with shape (8, n_channels).
        info (mne.Info): EEG channel geometry.
        title (str): Figure-level title.
        out_path (Path): Output PNG path.

    Returns:
        Path: Path to the saved figure.
    """
    collapsed = np.vstack([
        data[:4].mean(axis=0),
        data[4:].mean(axis=0),
    ])
    windows = [(0.00, 0.99), (1.00, 1.99)]
    vmin = float(np.min(collapsed))
    vmax = float(np.max(collapsed))
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
    image = None

    for idx, ax in enumerate(np.atleast_1d(axes)):
        try:
            image, _ = mne.viz.plot_topomap(
                collapsed[idx],
                info,
                axes=ax,
                show=False,
                contours=0,
                cmap="viridis",
                vlim=(vmin, vmax),
                sphere="eeglab",
            )
        except Exception:
            image, _ = mne.viz.plot_topomap(
                collapsed[idx],
                info,
                axes=ax,
                show=False,
                contours=0,
                cmap="viridis",
                vlim=(vmin, vmax),
                sphere="auto",
            )
        start_s, end_s = windows[idx]
        ax.set_title(f"{start_s:.2f}-{end_s:.2f} s")

    fig.suptitle(title)
    fig.subplots_adjust(left=0.06, right=0.86, top=0.86, bottom=0.08, wspace=0.18)
    cax = fig.add_axes([0.88, 0.18, 0.02, 0.65])
    colorbar = fig.colorbar(image, cax=cax)
    colorbar.set_label("Decoding accuracy")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def run_topomaps(
    subjects: list[str],
    targets: list[str],
    n_splits: int,
    n_neighbors: int,
    random_state: int,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], mne.Info, list[tuple[float, float]]]:
    """
    Runs decision-phase searchlight decoding and aggregates topomap outputs.

    The routine iterates over all subject/player/target combinations, logs
    progress with Rich, stores channel-level rows for CSV export, and computes
    per-target group mean maps for visualization.

    Args:
        subjects (list[str]): Subject IDs to process.
        targets (list[str]): Target keys to decode.
        n_splits (int): Number of CV folds.
        n_neighbors (int): Number of neighbors per searchlight.
        random_state (int): Random seed.

    Returns:
        tuple[pd.DataFrame, dict[str, np.ndarray], mne.Info, list[tuple[float, float]]]:
            - Channel-level long dataframe.
            - Group-mean maps per target.
            - Representative info object for plotting.
            - Representative time windows.

    Raises:
        RuntimeError: If no valid decoding/topomap results were produced.
    """
    rows: list[dict] = []
    group_maps: dict[str, list[np.ndarray]] = {target: [] for target in targets}
    representative_info: mne.Info | None = None
    representative_windows: list[tuple[float, float]] | None = None

    with Live(progress, console=console, refresh_per_second=1) as live:
        total_items = len(subjects) * 2 * len(targets)
        task_id = progress.add_task("Decision-phase topomaps", total=total_items)

        for subject_id in subjects:
            for person in ["P1", "P2"]:
                for target in targets:
                    try:
                        progress.update(task_id, description=f"{target}: sub-{subject_id} {person}")
                        live.refresh()

                        X_full, y, info, times = _prepare_subject_person(subject_id, person, target)
                        edges, windows = _make_bin_edges(times, n_bins=8)
                        # Average raw samples inside each bin before searchlight decoding.
                        X_bins = np.stack(
                            [X_full[:, :, start:stop].mean(axis=2) for start, stop in zip(edges[:-1], edges[1:])],
                            axis=2,
                        )
                        clusters = _make_searchlight_clusters(info, n_neighbors=n_neighbors)
                        maps = _decode_searchlight_maps(
                            X=X_bins,
                            y=y,
                            clusters=clusters,
                            n_splits=n_splits,
                            random_state=random_state,
                        )

                        if representative_info is None:
                            representative_info = _prepare_topomap_info(info)
                            representative_windows = windows

                        group_maps[target].append(maps)
                        for bin_idx, (start_s, end_s) in enumerate(windows):
                            for channel_idx, channel_name in enumerate(info["ch_names"]):
                                rows.append(
                                    {
                                        "subject": subject_id,
                                        "person": person,
                                        "target": target,
                                        "bin_index": bin_idx,
                                        "bin_start_s": start_s,
                                        "bin_end_s": end_s,
                                        "channel": channel_name,
                                        "accuracy": float(maps[bin_idx, channel_idx]),
                                        "chance_level": 1.0 / 3.0,
                                        "n_trials_used": int(len(y)),
                                    }
                                )

                        console.print(
                            f"[green]✓[/green] sub-{subject_id} {person} {target}: "
                            f"mean={maps.mean():.3f}, max={maps.max():.3f}, n={len(y)}"
                        )
                    except Exception as exc:
                        console.print(
                            f"[yellow]⚠[/yellow] sub-{subject_id} {person} {target}: skipped ({exc})"
                        )

                    progress.advance(task_id)
                    live.refresh()

    if representative_info is None or representative_windows is None:
        raise RuntimeError("No valid subject/person results were produced.")

    group_mean_maps = {
        target: np.mean(np.stack(maps_list, axis=0), axis=0)
        for target, maps_list in group_maps.items()
        if maps_list
    }
    if not group_mean_maps:
        raise RuntimeError("No valid topomap results were produced.")

    return pd.DataFrame(rows), group_mean_maps, representative_info, representative_windows


def main() -> None:
    """
    CLI entry point for generating decision-phase searchlight topomaps.

    This function parses user arguments, executes decoding/topomap generation,
    writes the channel-level CSV, and saves both detailed and collapsed figures.

    Returns:
        None
    """
    mne.set_config("MNE_LOGGING_LEVEL", "ERROR")

    parser = argparse.ArgumentParser(
        description=(
            "Generate decision-phase channel-searchlight topomaps from the preprocessed epochs. "
            "Default target is the participant's own current decision."
        )
    )
    parser.add_argument("--subjects", type=str, default=None, help="Comma-separated subject IDs, e.g. 01,02,03")
    parser.add_argument(
        "--targets",
        type=str,
        default="current_self",
        help=(
            "Comma-separated targets: current_self, previous_self, previous_other, current_other. "
            "Default: current_self"
        ),
    )
    parser.add_argument("--n-splits", type=int, default=5, help="Number of CV folds (default: 5)")
    parser.add_argument(
        "--n-neighbors",
        type=int,
        default=4,
        help="Number of nearest neighbors per searchlight in addition to the center channel (default: 4)",
    )
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    subjects = _resolve_subjects(args.subjects)
    targets = _resolve_targets(args.targets)

    out_dir = Path(config.OUTPUT_DIR).parent / "decoding"
    out_dir.mkdir(parents=True, exist_ok=True)

    df, group_mean_maps, info, windows = run_topomaps(
        subjects=subjects,
        targets=targets,
        n_splits=args.n_splits,
        n_neighbors=args.n_neighbors,
        random_state=args.random_state,
    )

    csv_path = out_dir / "decision_phase_topomap_channel_accuracy.csv"
    df.to_csv(csv_path, index=False)

    for target, maps in group_mean_maps.items():
        base_name = f"decision_phase_topomap_{target}"
        grid_path = out_dir / f"{base_name}_250ms.png"
        collapsed_path = out_dir / f"{base_name}_1s_collapsed.png"

        _save_topomap_grid(
            data=maps,
            info=info,
            windows=windows,
            title=f"Decision Phase Searchlight Topomaps: {TARGET_CHOICES[target]}",
            out_path=grid_path,
        )
        _save_collapsed_topomaps(
            data=maps,
            info=info,
            title=f"Decision Phase Searchlight Topomaps (1 s collapse): {TARGET_CHOICES[target]}",
            out_path=collapsed_path,
        )
        print(f"Saved 250 ms topomaps: {grid_path}")
        print(f"Saved 1 s collapsed topomaps: {collapsed_path}")

    print(f"Saved channel-level accuracies: {csv_path}")


if __name__ == "__main__":
    main()
