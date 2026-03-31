from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, cross_val_score, permutation_test_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

CURRENT_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = CURRENT_DIR.parent
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from preprocessing import config

from _rps_decoding_utils import (
    STRATEGY_CHOICES,
    STRATEGY_DISPLAY_NAMES,
    STRATEGY_LABELS,
    build_strategy_labels,
    get_phase_bin_windows,
    load_events_df,
    load_phase_features,
    match_status,
    resolve_csv_argument,
    resolve_subjects,
)


def _classifier() -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("lda", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
        ]
    )


def _prepare_data(X_full: np.ndarray, y_full: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = min(len(X_full), len(y_full))
    X = X_full[:n]
    y = y_full[:n]
    mask = np.isin(y, [1, 2])
    return X[mask], y[mask]


def _decode_subject_person(
    subject_id: str,
    person: str,
    target: str,
    n_splits: int,
    n_permutations: int,
    random_state: int,
) -> dict:
    events_df = load_events_df(subject_id)
    labels = build_strategy_labels(events_df, person, target)
    features = load_phase_features(subject_id, person, ["decision"])["decision"]
    match_state = match_status(events_df, person)

    X, y = _prepare_data(features, labels)
    classes, class_counts = np.unique(y, return_counts=True)
    if len(classes) < 2:
        raise ValueError(f"Not enough classes for sub-{subject_id} {person} {target}: {classes.tolist()}")

    actual_splits = min(n_splits, int(class_counts.min()))
    if actual_splits < 2:
        raise ValueError(f"Insufficient per-class samples for sub-{subject_id} {person} {target}.")

    clf = _classifier()
    cv = StratifiedKFold(n_splits=actual_splits, shuffle=True, random_state=random_state)
    n_bins = 8
    n_times = X.shape[2]
    bin_edges = np.linspace(0, n_times, n_bins + 1, dtype=int)
    bin_starts_s, bin_ends_s, bin_centers_s = get_phase_bin_windows("decision")

    bin_scores: list[float] = []
    bin_perm_pvalues: list[float] = []
    for index in range(n_bins):
        start = bin_edges[index]
        stop = bin_edges[index + 1]
        X_bin = X[:, :, start:stop].mean(axis=2)

        cv_scores = cross_val_score(clf, X_bin, y, cv=cv, scoring="balanced_accuracy", n_jobs=-1)
        bin_scores.append(float(np.mean(cv_scores)))

        _score, _perm_scores, pvalue = permutation_test_score(
            clf,
            X_bin,
            y,
            scoring="balanced_accuracy",
            cv=cv,
            n_permutations=n_permutations,
            random_state=random_state,
            n_jobs=-1,
        )
        bin_perm_pvalues.append(float(pvalue))

    class_count_map = {
        STRATEGY_LABELS[int(code)]: int(count)
        for code, count in zip(classes, class_counts)
    }
    chance_level = 0.5
    return {
        "subject": subject_id,
        "person": person,
        "phase": "decision",
        "target": target,
        "match_status": match_state,
        "n_trials_used": int(len(y)),
        "class_counts": class_count_map,
        "cv_accuracy_mean": float(np.mean(bin_scores)),
        "cv_accuracy_std": float(np.std(bin_scores)),
        "chance_level": chance_level,
        "above_chance": bool(np.mean(bin_scores) > chance_level),
        "permutation_pvalue": float(np.min(bin_perm_pvalues)),
        "n_permutations": int(n_permutations),
        "n_splits_used": int(actual_splits),
        "bin_starts_s": bin_starts_s,
        "bin_ends_s": bin_ends_s,
        "bin_centers_s": bin_centers_s,
        "bin_scores": [float(score) for score in bin_scores],
        "bin_permutation_pvalues": [float(pvalue) for pvalue in bin_perm_pvalues],
    }


def run_decoding(
    subjects: list[str],
    target: str,
    n_splits: int,
    n_permutations: int,
    random_state: int,
) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    for subject_id in subjects:
        for person in ["P1", "P2"]:
            try:
                rows.append(
                    _decode_subject_person(
                        subject_id=subject_id,
                        person=person,
                        target=target,
                        n_splits=n_splits,
                        n_permutations=n_permutations,
                        random_state=random_state,
                    )
                )
            except Exception as exc:
                print(f"sub-{subject_id} {person}: skipped ({exc})")

    if not rows:
        raise RuntimeError(f"No valid rows produced for strategy target={target}.")

    acc = np.asarray([row["cv_accuracy_mean"] for row in rows], dtype=float)
    chance = 0.5
    t_stat, p_two_tailed = ttest_1samp(acc, popmean=chance)
    summary = {
        "target": target,
        "target_label": STRATEGY_DISPLAY_NAMES[target],
        "chance_level": chance,
        "n_subject_person": int(len(rows)),
        "mean_accuracy": float(np.mean(acc)),
        "std_accuracy": float(np.std(acc)),
        "n_above_chance": int(np.sum(acc > chance)),
        "group_t_stat": float(t_stat),
        "group_pvalue_two_tailed": float(p_two_tailed),
        "group_pvalue_one_tailed_gt_chance": float(p_two_tailed / 2.0) if t_stat > 0 else 1.0,
    }
    return rows, summary


def _to_dataframe(rows: list[dict]) -> pd.DataFrame:
    flat_rows = []
    for row in rows:
        flat = dict(row)
        class_counts = flat.pop("class_counts", {})
        flat.pop("bin_starts_s", None)
        flat.pop("bin_ends_s", None)
        flat.pop("bin_centers_s", None)
        flat.pop("bin_scores", None)
        flat.pop("bin_permutation_pvalues", None)
        flat["n_target"] = int(class_counts.get("target", 0))
        flat["n_other"] = int(class_counts.get("other", 0))
        flat_rows.append(flat)
    return pd.DataFrame(flat_rows)


def _to_timecourse_dataframe(rows: list[dict]) -> pd.DataFrame:
    records: list[dict] = []
    for row in rows:
        for bin_index, (start_s, end_s, center_s, score, pvalue) in enumerate(
            zip(
                row["bin_starts_s"],
                row["bin_ends_s"],
                row["bin_centers_s"],
                row["bin_scores"],
                row["bin_permutation_pvalues"],
            )
        ):
            records.append(
                {
                    "subject": row["subject"],
                    "person": row["person"],
                    "target": row["target"],
                    "match_status": row["match_status"],
                    "bin_index": int(bin_index),
                    "bin_start_s": float(start_s),
                    "bin_end_s": float(end_s),
                    "bin_center_s": float(center_s),
                    "accuracy": float(score),
                    "permutation_pvalue": float(pvalue),
                    "chance_level": float(row["chance_level"]),
                    "n_trials_used": int(row["n_trials_used"]),
                }
            )
    return pd.DataFrame(records)


def _save_accuracy_plot(df: pd.DataFrame, summary: dict, out_dir: Path) -> None:
    labels = [f"sub-{row.subject}_{row.person}" for row in df.itertuples(index=False)]
    values = df["cv_accuracy_mean"].to_numpy(dtype=float)
    errors = df["cv_accuracy_std"].to_numpy(dtype=float)
    chance = float(summary["chance_level"])

    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(labels))
    ax.bar(x, values, yerr=errors, capsize=4, color="#1f77b4", edgecolor="black", linewidth=0.8)
    ax.axhline(chance, color="#c62828", linestyle="--", linewidth=1.5, label=f"Chance = {chance:.3f}")
    ax.axhline(float(summary["mean_accuracy"]), color="#2e7d32", linestyle=":", linewidth=1.5, label="Mean")
    ax.set_title(f"{summary['target_label']} · Decision phase")
    ax.set_ylabel("Balanced accuracy")
    ax.set_xlabel("Subject / player")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "decision_strategy_accuracy.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def _save_timecourse_plot(timecourse_df: pd.DataFrame, summary: dict, out_dir: Path) -> None:
    grouped = (
        timecourse_df.groupby(["bin_index", "bin_center_s", "bin_start_s", "bin_end_s"], as_index=False)
        .agg(mean_accuracy=("accuracy", "mean"), std_accuracy=("accuracy", "std"))
        .sort_values("bin_index")
    )
    grouped["std_accuracy"] = grouped["std_accuracy"].fillna(0.0)

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    x = grouped["bin_center_s"].to_numpy(dtype=float)
    y = grouped["mean_accuracy"].to_numpy(dtype=float)
    yerr = grouped["std_accuracy"].to_numpy(dtype=float)
    ax.plot(x, y, color="#1f77b4", marker="o", linewidth=2.0, label="Mean balanced accuracy")
    ax.fill_between(x, y - yerr, y + yerr, color="#1f77b4", alpha=0.18, label="±1 SD")
    ax.axhline(float(summary["chance_level"]), color="#c62828", linestyle="--", linewidth=1.5, label="Chance")
    ax.set_title(f"{summary['target_label']} · Decision time course")
    ax.set_xlabel("Time from decision onset (s)")
    ax.set_ylabel("Balanced accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{start:.2f}-{end:.2f}" for start, end in zip(grouped["bin_start_s"], grouped["bin_end_s"])],
        rotation=45,
        ha="right",
    )
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "decision_strategy_timecourse.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    mne.set_config("MNE_LOGGING_LEVEL", "ERROR")

    parser = argparse.ArgumentParser(description="Decision-phase EEG decoding of strategic RPS states.")
    parser.add_argument("--subjects", type=str, default=None, help="Comma-separated subject IDs")
    parser.add_argument(
        "--targets",
        type=str,
        default=",".join(STRATEGY_CHOICES),
        help="Comma-separated strategy targets",
    )
    parser.add_argument("--n-splits", type=int, default=10, help="Maximum number of CV folds")
    parser.add_argument("--n-permutations", type=int, default=200, help="Number of permutations")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    subjects = resolve_subjects(args.subjects)
    targets = resolve_csv_argument(
        args.targets,
        allowed=STRATEGY_CHOICES,
        default=list(STRATEGY_CHOICES),
    )

    base_out_dir = Path(config.OUTPUT_DIR).parent / "decoding" / "strategy"
    base_out_dir.mkdir(parents=True, exist_ok=True)

    for target in targets:
        rows, summary = run_decoding(
            subjects=subjects,
            target=target,
            n_splits=args.n_splits,
            n_permutations=args.n_permutations,
            random_state=args.random_state,
        )

        out_dir = base_out_dir / target
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "strategy_decoding_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

        df = _to_dataframe(rows)
        timecourse_df = _to_timecourse_dataframe(rows)
        df.to_csv(out_dir / "decision_strategy_results.csv", index=False)
        timecourse_df.to_csv(out_dir / "decision_strategy_timecourse.csv", index=False)
        _save_accuracy_plot(df, summary, out_dir)
        _save_timecourse_plot(timecourse_df, summary, out_dir)

        print(f"\n=== {summary['target_label']} ===")
        print(f"Output directory: {out_dir}")
        print(f"Mean balanced accuracy: {summary['mean_accuracy']:.3f}")
        print(f"One-tailed p > chance: {summary['group_pvalue_one_tailed_gt_chance']:.6f}")


if __name__ == "__main__":
    main()