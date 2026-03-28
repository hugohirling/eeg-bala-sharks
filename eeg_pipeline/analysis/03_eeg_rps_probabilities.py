from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

CURRENT_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = CURRENT_DIR.parent
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

import config

RESP_CODE_TO_NAME = {1: "rock", 2: "paper", 3: "scissors"}


def _balanced_accuracy_from_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Macro-averaged recall over rock/paper/scissors (class-balanced accuracy)."""
    recalls: list[float] = []
    for cls in [1, 2, 3]:
        mask = y_true == cls
        if np.any(mask):
            recalls.append(float(np.mean(y_pred[mask] == y_true[mask])))
    return float(np.mean(recalls)) if recalls else float("nan")


def _resolve_subjects(subjects_arg: str | None) -> list[str]:
    if subjects_arg:
        return [part.strip() for part in subjects_arg.split(",") if part.strip()]
    return list(config.SUBJECTS)


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


def _load_labels(subject_id: str, person: str) -> np.ndarray:
    events_path = _get_events_tsv_path(subject_id)
    if not events_path.exists():
        raise FileNotFoundError(f"Missing events file: {events_path}")

    events_df = pd.read_csv(events_path, sep="\t")
    response_col = _player_response_column(person)
    if response_col not in events_df.columns:
        raise ValueError(f"Column '{response_col}' not found in {events_path}")

    return events_df[response_col].to_numpy(dtype=int)


def _load_decision_features(subject_id: str, person: str) -> np.ndarray:
    epoch_path = _get_epoch_path(subject_id, person)
    if not epoch_path.exists():
        raise FileNotFoundError(f"Missing epoch file: {epoch_path}")

    epochs = mne.read_epochs(str(epoch_path), preload=True, verbose=False)
    decision_epochs = epochs.copy().crop(tmin=0.0, tmax=2.0)
    decision_epochs.pick("eeg")
    return decision_epochs.get_data(copy=True)


def _get_decision_time_bins(n_bins: int = 8) -> tuple[list[float], list[float], list[float]]:
    starts = [0.25 * i for i in range(n_bins)]
    ends = [start + 0.25 for start in starts]
    centers = [(start + end) / 2.0 for start, end in zip(starts, ends)]
    return starts, ends, centers


def _decode_timecourse_by_bin(
    X_full: np.ndarray,
    y: np.ndarray,
    subject_id: str,
    person: str,
    actual_splits: int,
    random_state: int,
) -> pd.DataFrame:
    """Compute bin-wise decoding accuracy overall and per decision class."""
    n_times = X_full.shape[2]
    n_bins = 8
    bin_edges = np.linspace(0, n_times, n_bins + 1, dtype=int)
    starts, ends, centers = _get_decision_time_bins(n_bins=n_bins)

    clf = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "lda",
                LinearDiscriminantAnalysis(
                    solver="lsqr",
                    shrinkage="auto",
                    priors=[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
                ),
            ),
        ]
    )
    cv = StratifiedKFold(n_splits=actual_splits, shuffle=True, random_state=random_state)

    rows: list[dict] = []
    for i in range(n_bins):
        start_idx = bin_edges[i]
        end_idx = bin_edges[i + 1]
        X_bin = X_full[:, :, start_idx:end_idx].mean(axis=2)

        proba_bin = cross_val_predict(clf, X_bin, y, cv=cv, method="predict_proba", n_jobs=1)
        pred_idx = np.argmax(proba_bin, axis=1)
        pred_codes = np.array([1, 2, 3], dtype=int)[pred_idx]

        overall_acc = float(np.mean(pred_codes == y))

        mask_rock = y == 1
        mask_paper = y == 2
        mask_scissors = y == 3
        acc_rock = float(np.mean(pred_codes[mask_rock] == y[mask_rock])) if np.any(mask_rock) else np.nan
        acc_paper = float(np.mean(pred_codes[mask_paper] == y[mask_paper])) if np.any(mask_paper) else np.nan
        acc_scissors = (
            float(np.mean(pred_codes[mask_scissors] == y[mask_scissors])) if np.any(mask_scissors) else np.nan
        )

        balanced_acc = _balanced_accuracy_from_predictions(y_true=y, y_pred=pred_codes)

        rows.append(
            {
                "subject": subject_id,
                "person": person,
                "bin_index": int(i),
                "bin_start_s": float(starts[i]),
                "bin_end_s": float(ends[i]),
                "bin_center_s": float(centers[i]),
                "accuracy_overall": overall_acc,
                "accuracy_balanced": balanced_acc,
                "accuracy_rock": acc_rock,
                "accuracy_paper": acc_paper,
                "accuracy_scissors": acc_scissors,
                "chance_level": 1.0 / 3.0,
            }
        )

    return pd.DataFrame(rows)


def _decode_probabilities_for_subject_person(
    subject_id: str,
    person: str,
    n_splits: int,
    random_state: int,
) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """
    EEG-based probability decoder.

    Uses out-of-fold predict_proba so probabilities are computed on unseen trials.
    """
    X_full = _load_decision_features(subject_id, person)
    y_full = _load_labels(subject_id, person)

    n = min(len(X_full), len(y_full))
    X_full = X_full[:n]
    y = y_full[:n]

    mask = np.isin(y, [1, 2, 3])
    X_full = X_full[mask]
    y = y[mask]

    if len(y) < 30:
        raise ValueError(f"Too few valid trials for sub-{subject_id} {person}: {len(y)}")

    classes, class_counts = np.unique(y, return_counts=True)
    if len(classes) < 3:
        raise ValueError(
            f"Not enough classes for sub-{subject_id} {person}. Classes found: {classes.tolist()}"
        )

    smallest_class = int(class_counts.min())
    actual_splits = min(n_splits, smallest_class)
    if actual_splits < 2:
        raise ValueError(
            f"Insufficient per-class samples for CV in sub-{subject_id} {person}."
        )

    # Feature extraction: channel means in 250 ms bins over decision window (0-2s).
    # This keeps temporal structure while reducing dimensionality.
    n_times = X_full.shape[2]
    n_bins = 8
    bin_edges = np.linspace(0, n_times, n_bins + 1, dtype=int)
    binned = [X_full[:, :, bin_edges[i] : bin_edges[i + 1]].mean(axis=2) for i in range(n_bins)]
    X = np.concatenate(binned, axis=1)

    clf = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "lda",
                LinearDiscriminantAnalysis(
                    solver="lsqr",
                    shrinkage="auto",
                    priors=[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
                ),
            ),
        ]
    )

    cv = StratifiedKFold(n_splits=actual_splits, shuffle=True, random_state=random_state)

    # Out-of-fold probabilities (each trial predicted by model that has not seen that trial).
    proba = cross_val_predict(clf, X, y, cv=cv, method="predict_proba", n_jobs=1)

    clf_classes = [1, 2, 3]
    if proba.shape[1] != len(clf_classes):
        raise RuntimeError(
            f"Unexpected proba shape for sub-{subject_id} {person}: {proba.shape}"
        )

    pred_idx = np.argmax(proba, axis=1)
    pred_codes = np.array([clf_classes[idx] for idx in pred_idx], dtype=int)

    trial_df = pd.DataFrame(
        {
            "subject": subject_id,
            "person": person,
            "trial_index": np.arange(len(y), dtype=int),
            "true_choice_code": y.astype(int),
            "true_choice": [RESP_CODE_TO_NAME[int(code)] for code in y],
            "p_rock_eeg": proba[:, 0].astype(float),
            "p_paper_eeg": proba[:, 1].astype(float),
            "p_scissors_eeg": proba[:, 2].astype(float),
            "pred_choice_code": pred_codes,
            "pred_choice": [RESP_CODE_TO_NAME[int(code)] for code in pred_codes],
            "pred_confidence": np.max(proba, axis=1).astype(float),
            "correct": (pred_codes == y).astype(int),
        }
    )

    accuracy = float(np.mean(trial_df["correct"].to_numpy(dtype=float)))
    balanced_accuracy = _balanced_accuracy_from_predictions(y_true=y, y_pred=pred_codes)

    summary = {
        "subject": subject_id,
        "person": person,
        "n_trials_used": int(len(y)),
        "cv_splits_used": int(actual_splits),
        "chance_level": 1.0 / 3.0,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "mean_p_rock_eeg": float(trial_df["p_rock_eeg"].mean()),
        "mean_p_paper_eeg": float(trial_df["p_paper_eeg"].mean()),
        "mean_p_scissors_eeg": float(trial_df["p_scissors_eeg"].mean()),
        "mean_pred_confidence": float(trial_df["pred_confidence"].mean()),
        "n_rock": int(np.sum(y == 1)),
        "n_paper": int(np.sum(y == 2)),
        "n_scissors": int(np.sum(y == 3)),
    }

    timecourse_df = _decode_timecourse_by_bin(
        X_full=X_full,
        y=y,
        subject_id=subject_id,
        person=person,
        actual_splits=actual_splits,
        random_state=random_state,
    )

    return trial_df, summary, timecourse_df


def run_eeg_probability_analysis(
    subjects: list[str],
    n_splits: int,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict, pd.DataFrame, pd.DataFrame]:
    all_trial_rows: list[pd.DataFrame] = []
    all_summary_rows: list[dict] = []
    all_timecourse_rows: list[pd.DataFrame] = []

    for subject_id in subjects:
        for person in ["P1", "P2"]:
            try:
                trial_df, summary, timecourse_df = _decode_probabilities_for_subject_person(
                    subject_id=subject_id,
                    person=person,
                    n_splits=n_splits,
                    random_state=random_state,
                )
                all_trial_rows.append(trial_df)
                all_summary_rows.append(summary)
                all_timecourse_rows.append(timecourse_df)
                print(
                    f"sub-{subject_id} {person}: "
                    f"acc={summary['accuracy']:.3f}, "
                    f"bal_acc={summary['balanced_accuracy']:.3f}, "
                    f"mean probs (R/P/S)=({summary['mean_p_rock_eeg']:.3f}, "
                    f"{summary['mean_p_paper_eeg']:.3f}, {summary['mean_p_scissors_eeg']:.3f}), "
                    f"n={summary['n_trials_used']}"
                )
            except Exception as exc:
                print(f"sub-{subject_id} {person}: skipped ({exc})")

    if not all_trial_rows:
        raise RuntimeError("No valid subject/person results were produced.")

    trials_df = pd.concat(all_trial_rows, ignore_index=True)
    summary_df = pd.DataFrame(all_summary_rows)
    timecourse_df = pd.concat(all_timecourse_rows, ignore_index=True)

    group_timecourse_df = (
        timecourse_df.groupby(["bin_index", "bin_start_s", "bin_end_s", "bin_center_s"], as_index=False)
        .agg(
            accuracy_overall=("accuracy_overall", "mean"),
            accuracy_balanced=("accuracy_balanced", "mean"),
            accuracy_rock=("accuracy_rock", "mean"),
            accuracy_paper=("accuracy_paper", "mean"),
            accuracy_scissors=("accuracy_scissors", "mean"),
            chance_level=("chance_level", "mean"),
        )
        .sort_values("bin_index")
    )

    group_summary = {
        "n_subject_person": int(len(summary_df)),
        "n_trials_total": int(len(trials_df)),
        "mean_accuracy": float(summary_df["accuracy"].mean()),
        "mean_balanced_accuracy": float(summary_df["balanced_accuracy"].mean()),
        "std_accuracy": float(summary_df["accuracy"].std(ddof=0)),
        "std_balanced_accuracy": float(summary_df["balanced_accuracy"].std(ddof=0)),
        "chance_level": 1.0 / 3.0,
        "mean_p_rock_eeg": float(summary_df["mean_p_rock_eeg"].mean()),
        "mean_p_paper_eeg": float(summary_df["mean_p_paper_eeg"].mean()),
        "mean_p_scissors_eeg": float(summary_df["mean_p_scissors_eeg"].mean()),
        "mean_pred_confidence": float(summary_df["mean_pred_confidence"].mean()),
    }

    return trials_df, summary_df, group_summary, timecourse_df, group_timecourse_df


def _save_timecourse_accuracy_plot(group_timecourse_df: pd.DataFrame, out_dir: Path) -> Path:
    """Plot accuracy over time for each decision class plus overall."""
    x = group_timecourse_df["bin_center_s"].to_numpy(dtype=float)
    overall = group_timecourse_df["accuracy_overall"].to_numpy(dtype=float)
    balanced = group_timecourse_df["accuracy_balanced"].to_numpy(dtype=float)
    rock = group_timecourse_df["accuracy_rock"].to_numpy(dtype=float)
    paper = group_timecourse_df["accuracy_paper"].to_numpy(dtype=float)
    scissors = group_timecourse_df["accuracy_scissors"].to_numpy(dtype=float)
    chance = float(group_timecourse_df["chance_level"].iloc[0])

    fig, ax = plt.subplots(figsize=(10, 5.8))
    ax.plot(x, overall, marker="o", linewidth=2.2, color="#1f77b4", label="Overall")
    ax.plot(x, balanced, marker="o", linewidth=2.2, color="#9467bd", label="Balanced")
    ax.plot(x, rock, marker="o", linewidth=1.8, color="#d62728", label="Rock")
    ax.plot(x, paper, marker="o", linewidth=1.8, color="#2ca02c", label="Paper")
    ax.plot(x, scissors, marker="o", linewidth=1.8, color="#ff7f0e", label="Scissors")
    ax.axhline(chance, linestyle="--", color="black", linewidth=1.2, label=f"Chance = {chance:.3f}")

    ax.set_title("Decision-Phase Accuracy Over Time (per decision)")
    ax.set_xlabel("Time from decision onset (s)")
    ax.set_ylabel("Accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{start:.2f}-{end:.2f}" for start, end in zip(group_timecourse_df["bin_start_s"], group_timecourse_df["bin_end_s"])],
        rotation=45,
        ha="right",
    )
    y_min = min(
        chance - 0.05,
        float(np.nanmin([overall.min(), balanced.min(), rock.min(), paper.min(), scissors.min()])) - 0.03,
    )
    y_max = max(
        0.40,
        float(np.nanmax([overall.max(), balanced.max(), rock.max(), paper.max(), scissors.max()])) + 0.03,
    )
    ax.set_ylim(y_min, y_max)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()

    plot_path = out_dir / "eeg_rps_accuracy_timecourse_per_decision.png"
    fig.savefig(plot_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return plot_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate Rock/Paper/Scissors probabilities from EEG (decision phase), "
            "using out-of-fold classifier probabilities."
        )
    )
    parser.add_argument("--subjects", type=str, default=None, help="Comma-separated subject IDs, e.g. 01,02,03")
    parser.add_argument("--n-splits", type=int, default=10, help="Number of CV folds (default: 10)")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    subjects = _resolve_subjects(args.subjects)
    trials_df, summary_df, group_summary, timecourse_df, group_timecourse_df = run_eeg_probability_analysis(
        subjects=subjects,
        n_splits=args.n_splits,
        random_state=args.random_state,
    )

    out_dir = Path(config.OUTPUT_DIR).parent / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    trial_path = out_dir / "eeg_rps_probabilities_per_trial.csv"
    summary_path = out_dir / "eeg_rps_probabilities_per_subject.csv"
    timecourse_path = out_dir / "eeg_rps_accuracy_timecourse_per_subject.csv"
    group_timecourse_path = out_dir / "eeg_rps_accuracy_timecourse_group.csv"
    group_path = out_dir / "eeg_rps_probabilities_group_summary.json"

    trials_df.to_csv(trial_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    timecourse_df.to_csv(timecourse_path, index=False)
    group_timecourse_df.to_csv(group_timecourse_path, index=False)
    group_path.write_text(json.dumps(group_summary, indent=2), encoding="utf-8")
    plot_path = _save_timecourse_accuracy_plot(group_timecourse_df=group_timecourse_df, out_dir=out_dir)

    print("\n=== EEG Probability Summary ===")
    print(f"N subject/person: {group_summary['n_subject_person']}")
    print(f"Total trials: {group_summary['n_trials_total']}")
    print(
        f"Mean EEG-decoding accuracy: {group_summary['mean_accuracy']:.3f} "
        f"(chance={group_summary['chance_level']:.3f})"
    )
    print(
        "Mean EEG-decoding balanced accuracy: "
        f"{group_summary['mean_balanced_accuracy']:.3f} "
        f"(chance={group_summary['chance_level']:.3f})"
    )
    print(
        "Mean EEG probabilities (R/P/S): "
        f"{group_summary['mean_p_rock_eeg']:.3f}, "
        f"{group_summary['mean_p_paper_eeg']:.3f}, "
        f"{group_summary['mean_p_scissors_eeg']:.3f}"
    )
    print(f"Saved per-trial probabilities: {trial_path}")
    print(f"Saved per-subject summary: {summary_path}")
    print(f"Saved timecourse (subject/person): {timecourse_path}")
    print(f"Saved timecourse (group): {group_timecourse_path}")
    print(f"Saved group summary: {group_path}")
    print(f"Saved timecourse plot: {plot_path}")


if __name__ == "__main__":
    main()
