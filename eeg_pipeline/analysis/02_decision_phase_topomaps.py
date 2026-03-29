from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

CURRENT_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = CURRENT_DIR.parent
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from preprocessing import config

RESP_CODE_TO_NAME = {1: "rock", 2: "paper", 3: "scissors"}
TARGET_CHOICES = {
    "current_self": "Current own decision",
    "previous_self": "Previous own decision",
    "previous_other": "Previous opponent decision",
    "current_other": "Current opponent decision",
}


def _resolve_subjects(subjects_arg: str | None) -> list[str]:
    if subjects_arg:
        return [part.strip() for part in subjects_arg.split(",") if part.strip()]
    return list(config.SUBJECTS)


def _resolve_targets(targets_arg: str | None) -> list[str]:
    if not targets_arg:
        return ["current_self"]

    targets = [part.strip() for part in targets_arg.split(",") if part.strip()]
    invalid = sorted(set(targets) - set(TARGET_CHOICES))
    if invalid:
        raise ValueError(f"Unsupported targets: {invalid}. Choose from {sorted(TARGET_CHOICES)}")
    return targets


def _get_events_tsv_path(subject_id: str) -> Path:
    return (
        Path(config.BIDS_ROOT)
        / f"sub-{subject_id}"
        / "eeg"
        / f"sub-{subject_id}_task-RPS_events.tsv"
    )


def _get_epoch_path(subject_id: str, person: str) -> Path:
    return Path(config.OUTPUT_DIR) / f"sub-{subject_id}_{person}_epoch.fif"


def _player_response_column(person: str) -> str:
    prefix = config.PLAYER_PREFIX_MAP[person]
    if prefix.startswith("1"):
        return "player1_resp"
    if prefix.startswith("2"):
        return "player2_resp"
    raise ValueError(f"Unexpected player prefix for {person}: {prefix}")


def _other_player_response_column(person: str) -> str:
    current_col = _player_response_column(person)
    return "player2_resp" if current_col == "player1_resp" else "player1_resp"


def _load_events_df(subject_id: str) -> pd.DataFrame:
    events_path = _get_events_tsv_path(subject_id)
    if not events_path.exists():
        raise FileNotFoundError(f"Missing events file: {events_path}")
    return pd.read_csv(events_path, sep="\t")


def _load_decision_epochs(subject_id: str, person: str) -> mne.Epochs:
    epoch_path = _get_epoch_path(subject_id, person)
    if not epoch_path.exists():
        raise FileNotFoundError(f"Missing epoch file: {epoch_path}")

    epochs = mne.read_epochs(str(epoch_path), preload=True, verbose=False)
    decision_epochs = epochs.copy().crop(tmin=0.0, tmax=2.0)
    decision_epochs.pick("eeg")
    return decision_epochs


def _previous_trial(labels: np.ndarray) -> np.ndarray:
    shifted = np.full(labels.shape, fill_value=-1, dtype=int)
    if len(labels) > 1:
        shifted[1:] = labels[:-1]
    return shifted


def _load_target_labels(subject_id: str, person: str, target: str) -> np.ndarray:
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
    edges = np.linspace(0, len(times), n_bins + 1, dtype=int)
    labels = []
    for start, stop in zip(edges[:-1], edges[1:]):
        stop_idx = max(start, stop - 1)
        labels.append((float(times[start]), float(times[stop_idx])))
    return edges, labels


def _make_searchlight_clusters(info: mne.Info, n_neighbors: int) -> list[np.ndarray]:
    positions = np.array([channel["loc"][:3] for channel in info["chs"]], dtype=float)
    clusters: list[np.ndarray] = []

    for center_idx in range(len(info["ch_names"])):
        center_pos = positions[center_idx]
        distances = np.linalg.norm(positions - center_pos, axis=1)
        ordered = np.argsort(distances)
        cluster = np.sort(ordered[: n_neighbors + 1])
        clusters.append(cluster)

    return clusters


def _prepare_subject_person(
    subject_id: str,
    person: str,
    target: str,
) -> tuple[np.ndarray, np.ndarray, mne.Info, np.ndarray]:
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
    vmin = float(np.min(data))
    vmax = float(np.max(data))
    fig, axes = plt.subplots(2, 4, figsize=(15, 7))
    axes = axes.ravel()
    image = None

    for idx, ax in enumerate(axes):
        image, _ = mne.viz.plot_topomap(
            data[idx],
            info,
            axes=ax,
            show=False,
            contours=0,
            cmap="viridis",
            vlim=(vmin, vmax),
        )
        start_s, end_s = windows[idx]
        ax.set_title(f"{start_s:.2f}-{end_s:.2f} s")

    fig.suptitle(title)
    colorbar = fig.colorbar(image, ax=axes.tolist(), shrink=0.85)
    colorbar.set_label("Decoding accuracy")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _save_collapsed_topomaps(
    data: np.ndarray,
    info: mne.Info,
    title: str,
    out_path: Path,
) -> Path:
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
        image, _ = mne.viz.plot_topomap(
            collapsed[idx],
            info,
            axes=ax,
            show=False,
            contours=0,
            cmap="viridis",
            vlim=(vmin, vmax),
        )
        start_s, end_s = windows[idx]
        ax.set_title(f"{start_s:.2f}-{end_s:.2f} s")

    fig.suptitle(title)
    colorbar = fig.colorbar(image, ax=np.atleast_1d(axes).tolist(), shrink=0.85)
    colorbar.set_label("Decoding accuracy")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
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
    rows: list[dict] = []
    group_maps: dict[str, list[np.ndarray]] = {target: [] for target in targets}
    representative_info: mne.Info | None = None
    representative_windows: list[tuple[float, float]] | None = None

    for subject_id in subjects:
        for person in ["P1", "P2"]:
            for target in targets:
                try:
                    X_full, y, info, times = _prepare_subject_person(subject_id, person, target)
                    edges, windows = _make_bin_edges(times, n_bins=8)
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
                        representative_info = info
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

                    print(
                        f"sub-{subject_id} {person} {target}: "
                        f"mean={maps.mean():.3f}, max={maps.max():.3f}, n={len(y)}"
                    )
                except Exception as exc:
                    print(f"sub-{subject_id} {person} {target}: skipped ({exc})")

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

    out_dir = Path(config.OUTPUT_DIR).parent / "analysis"
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
