from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from rich.console import Console
from rich.live import Live
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn, TimeRemainingColumn
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
    PHASE_SPECS,
    RESP_CODE_TO_NAME,
    TARGET_CHOICES,
    TARGET_DISPLAY_NAMES,
    build_target_labels,
    get_phase_bin_windows,
    load_events_df,
    load_phase_features,
    match_status,
    resolve_csv_argument,
    resolve_subjects,
    target_output_dir,
)

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


def _classifier() -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("lda", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
        ]
    )


def _prepare_phase_data(X_full: np.ndarray, y_full: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = min(len(X_full), len(y_full))
    X = X_full[:n]
    y = y_full[:n]
    mask = np.isin(y, [1, 2, 3])
    return X[mask], y[mask]


def _decode_phase(
    subject_id: str,
    person: str,
    phase: str,
    target: str,
    match_state: str,
    X_full: np.ndarray,
    y_full: np.ndarray,
    n_splits: int,
    n_permutations: int,
    random_state: int,
) -> dict:
    X, y = _prepare_phase_data(X_full, y_full)
    classes, class_counts = np.unique(y, return_counts=True)
    if len(classes) < 3:
        raise ValueError(
            f"Not enough classes for sub-{subject_id} {person} {target} {phase}: {classes.tolist()}"
        )

    actual_splits = min(n_splits, int(class_counts.min()))
    if actual_splits < 2:
        raise ValueError(
            f"Insufficient per-class samples for sub-{subject_id} {person} {target} {phase}."
        )

    class_count_map = {
        RESP_CODE_TO_NAME[int(code)]: int(count)
        for code, count in zip(classes, class_counts)
    }

    clf = _classifier()
    cv = StratifiedKFold(n_splits=actual_splits, shuffle=True, random_state=random_state)
    n_bins = int(PHASE_SPECS[phase]["n_bins"])
    n_times = X.shape[2]
    bin_edges = np.linspace(0, n_times, n_bins + 1, dtype=int)
    bin_starts_s, bin_ends_s, bin_centers_s = get_phase_bin_windows(phase)

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

    chance_level = 1.0 / 3.0
    return {
        "subject": subject_id,
        "person": person,
        "phase": phase,
        "target": target,
        "match_status": match_state,
        "n_trials_used": int(len(y)),
        "class_counts": class_count_map,
        "cv_accuracy_mean": float(np.mean(bin_scores)),
        "cv_accuracy_std": float(np.std(bin_scores)),
        "n_time_bins": int(n_bins),
        "chance_level": chance_level,
        "above_chance": bool(np.mean(bin_scores) > chance_level),
        "permutation_accuracy": float(np.mean(bin_scores)),
        "permutation_pvalue": float(np.min(bin_perm_pvalues)),
        "n_permutations": int(n_permutations),
        "n_splits_used": int(actual_splits),
        "bin_starts_s": bin_starts_s,
        "bin_ends_s": bin_ends_s,
        "bin_centers_s": bin_centers_s,
        "bin_scores": [float(score) for score in bin_scores],
        "bin_permutation_pvalues": [float(pvalue) for pvalue in bin_perm_pvalues],
    }


def _decode_subject_person(
    subject_id: str,
    person: str,
    target: str,
    phases: list[str],
    n_splits: int,
    n_permutations: int,
    random_state: int,
) -> dict[str, dict]:
    events_df = load_events_df(subject_id)
    labels = build_target_labels(events_df, person, target)
    match_state = match_status(events_df, person)
    features = load_phase_features(subject_id, person, phases)

    results: dict[str, dict] = {}
    for phase in phases:
        results[phase] = _decode_phase(
            subject_id=subject_id,
            person=person,
            phase=phase,
            target=target,
            match_state=match_state,
            X_full=features[phase],
            y_full=labels,
            n_splits=n_splits,
            n_permutations=n_permutations,
            random_state=random_state,
        )
    return results


def run_decoding(
    subjects: list[str],
    target: str,
    phases: list[str],
    n_splits: int,
    n_permutations: int,
    random_state: int,
) -> tuple[dict[str, list[dict]], dict]:
    rows_by_phase = {phase: [] for phase in phases}

    with Live(progress, console=console, refresh_per_second=1) as live:
        total_items = len(subjects) * 2
        task_id = progress.add_task(f"Decoding {target}", total=total_items)

        for subject_id in subjects:
            for person in ["P1", "P2"]:
                try:
                    progress.update(task_id, description=f"{target}: sub-{subject_id} {person}")
                    live.refresh()

                    person_results = _decode_subject_person(
                        subject_id=subject_id,
                        person=person,
                        target=target,
                        phases=phases,
                        n_splits=n_splits,
                        n_permutations=n_permutations,
                        random_state=random_state,
                    )
                    for phase, row in person_results.items():
                        rows_by_phase[phase].append(row)

                    status_line = ", ".join(
                        f"{phase}={person_results[phase]['cv_accuracy_mean']:.3f}"
                        for phase in phases
                    )
                    console.print(f"[green]✓[/green] sub-{subject_id} {person}: {status_line}")
                except Exception as exc:
                    console.print(f"[yellow]⚠[/yellow] sub-{subject_id} {person}: skipped ({exc})")

                progress.advance(task_id)
                live.refresh()

    if not any(rows_by_phase.values()):
        raise RuntimeError(f"No valid results produced for target={target}.")

    chance = 1.0 / 3.0
    summary: dict[str, object] = {
        "target": target,
        "target_label": TARGET_DISPLAY_NAMES[target],
        "chance_level": chance,
        "n_subject_person": int(max((len(rows) for rows in rows_by_phase.values()), default=0)),
        "phases": phases,
    }
    for phase in phases:
        rows = rows_by_phase[phase]
        acc = np.asarray([row["cv_accuracy_mean"] for row in rows], dtype=float)
        t_stat, p_two_tailed = ttest_1samp(acc, popmean=chance)
        summary[f"mean_accuracy_{phase}"] = float(np.mean(acc))
        summary[f"std_accuracy_{phase}"] = float(np.std(acc))
        summary[f"n_above_chance_{phase}"] = int(np.sum(acc > chance))
        summary[f"group_t_stat_{phase}"] = float(t_stat)
        summary[f"group_pvalue_two_tailed_{phase}"] = float(p_two_tailed)
        summary[f"group_pvalue_one_tailed_gt_chance_{phase}"] = float(p_two_tailed / 2.0) if t_stat > 0 else 1.0

    return rows_by_phase, summary


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
        for cls_name in ["rock", "paper", "scissors"]:
            flat[f"n_{cls_name}"] = int(class_counts.get(cls_name, 0))
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
                    "phase": row["phase"],
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


def _save_accuracy_plot(df: pd.DataFrame, summary: dict, out_dir: Path, phase: str) -> Path:
    labels = [f"sub-{row.subject}_{row.person}" for row in df.itertuples(index=False)]
    values = df["cv_accuracy_mean"].to_numpy(dtype=float)
    errors = df["cv_accuracy_std"].to_numpy(dtype=float)
    chance = float(summary["chance_level"])
    mean_acc = float(summary[f"mean_accuracy_{phase}"])

    colors = ["#1f77b4" if value > chance else "#9aa0a6" for value in values]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(labels))
    ax.bar(x, values, yerr=errors, capsize=4, color=colors, edgecolor="black", linewidth=0.8)
    ax.axhline(chance, color="#c62828", linestyle="--", linewidth=1.5, label=f"Chance = {chance:.3f}")
    ax.axhline(mean_acc, color="#2e7d32", linestyle=":", linewidth=1.5, label=f"Mean = {mean_acc:.3f}")
    ax.set_title(f"{summary['target_label']} · {phase.capitalize()} phase")
    ax.set_ylabel("Balanced accuracy")
    ax.set_xlabel("Subject / player")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylim(max(0.0, float(np.min(values - errors)) - 0.03), max(0.45, float(np.max(values + errors)) + 0.03))
    ax.grid(axis="y", alpha=0.25)
    ax.legend()

    subtitle = (
        f"One-tailed group p = {summary[f'group_pvalue_one_tailed_gt_chance_{phase}']:.4f}, "
        f"above chance = {summary[f'n_above_chance_{phase}']}/{summary['n_subject_person']}"
    )
    fig.text(0.5, 0.01, subtitle, ha="center", va="bottom")
    fig.tight_layout(rect=(0, 0.04, 1, 1))

    plot_path = out_dir / f"{phase}_phase_decoding_accuracy.png"
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return plot_path


def _save_timecourse_plot(timecourse_df: pd.DataFrame, summary: dict, out_dir: Path, phase: str) -> Path:
    grouped = (
        timecourse_df.groupby(["bin_index", "bin_center_s", "bin_start_s", "bin_end_s"], as_index=False)
        .agg(mean_accuracy=("accuracy", "mean"), std_accuracy=("accuracy", "std"))
        .sort_values("bin_index")
    )
    grouped["std_accuracy"] = grouped["std_accuracy"].fillna(0.0)

    x = grouped["bin_center_s"].to_numpy(dtype=float)
    y = grouped["mean_accuracy"].to_numpy(dtype=float)
    yerr = grouped["std_accuracy"].to_numpy(dtype=float)
    chance = float(summary["chance_level"])

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    ax.plot(x, y, color="#1f77b4", marker="o", linewidth=2.0, label="Mean balanced accuracy")
    ax.fill_between(x, y - yerr, y + yerr, color="#1f77b4", alpha=0.18, label="±1 SD")
    ax.axhline(chance, color="#c62828", linestyle="--", linewidth=1.5, label=f"Chance = {chance:.3f}")
    ax.set_title(f"{summary['target_label']} · {phase.capitalize()} time course")
    ax.set_xlabel(f"Time from {phase} onset (s)")
    ax.set_ylabel("Balanced accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{start:.2f}-{end:.2f}" for start, end in zip(grouped["bin_start_s"], grouped["bin_end_s"])],
        rotation=45,
        ha="right",
    )
    ax.set_ylim(max(0.2, float(np.min(y - yerr)) - 0.03), max(0.38, float(np.max(y + yerr)) + 0.03))
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()

    plot_path = out_dir / f"{phase}_phase_decoding_timecourse.png"
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return plot_path


def _save_timecourse_plot_by_person(
    timecourse_df: pd.DataFrame,
    summary: dict,
    out_dir: Path,
    person: str,
    phase: str,
) -> Path | None:
    person_df = timecourse_df[timecourse_df["person"] == person].copy()
    if person_df.empty:
        return None

    grouped = (
        person_df.groupby(["bin_index", "bin_center_s", "bin_start_s", "bin_end_s"], as_index=False)
        .agg(mean_accuracy=("accuracy", "mean"), std_accuracy=("accuracy", "std"))
        .sort_values("bin_index")
    )
    grouped["std_accuracy"] = grouped["std_accuracy"].fillna(0.0)

    x = grouped["bin_center_s"].to_numpy(dtype=float)
    y = grouped["mean_accuracy"].to_numpy(dtype=float)
    yerr = grouped["std_accuracy"].to_numpy(dtype=float)
    chance = float(summary["chance_level"])

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    ax.plot(x, y, color="#1f77b4", marker="o", linewidth=2.0, label=f"Mean balanced accuracy ({person})")
    ax.fill_between(x, y - yerr, y + yerr, color="#1f77b4", alpha=0.18, label="±1 SD")
    ax.axhline(chance, color="#c62828", linestyle="--", linewidth=1.5, label=f"Chance = {chance:.3f}")
    ax.set_title(f"{summary['target_label']} · {phase.capitalize()} time course ({person})")
    ax.set_xlabel(f"Time from {phase} onset (s)")
    ax.set_ylabel("Balanced accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{start:.2f}-{end:.2f}" for start, end in zip(grouped["bin_start_s"], grouped["bin_end_s"])],
        rotation=45,
        ha="right",
    )
    ax.set_ylim(max(0.2, float(np.min(y - yerr)) - 0.03), max(0.38, float(np.max(y + yerr)) + 0.03))
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()

    plot_path = out_dir / f"{phase}_phase_decoding_timecourse_{person}.png"
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return plot_path


def _save_target_outputs(
    out_dir: Path,
    rows_by_phase: dict[str, list[dict]],
    summary: dict,
    *,
    plot_only: bool,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path_summary = out_dir / "decoding_summary.json"

    if not plot_only:
        json_path_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    for phase, rows in rows_by_phase.items():
        df = _to_dataframe(rows)
        timecourse_df = _to_timecourse_dataframe(rows)
        csv_path = out_dir / f"{phase}_phase_decoding_results.csv"
        timecourse_csv_path = out_dir / f"{phase}_phase_decoding_timecourse.csv"

        if not plot_only:
            df.to_csv(csv_path, index=False)
            timecourse_df.to_csv(timecourse_csv_path, index=False)
        else:
            df = pd.read_csv(csv_path)
            timecourse_df = pd.read_csv(timecourse_csv_path)

        _save_accuracy_plot(df, summary, out_dir, phase)
        _save_timecourse_plot(timecourse_df, summary, out_dir, phase)
        _save_timecourse_plot_by_person(timecourse_df, summary, out_dir, "P1", phase)
        _save_timecourse_plot_by_person(timecourse_df, summary, out_dir, "P2", phase)


def main() -> None:
    mne.set_config("MNE_LOGGING_LEVEL", "ERROR")

    parser = argparse.ArgumentParser(
        description="All-phase EEG decoding for current and previous self/other RPS decisions."
    )
    parser.add_argument("--subjects", type=str, default=None, help="Comma-separated subject IDs, e.g. 01,02,03")
    parser.add_argument(
        "--targets",
        type=str,
        default="current_self",
        help="Comma-separated targets: current_self,current_other,previous_self,previous_other",
    )
    parser.add_argument(
        "--phases",
        type=str,
        default="decision,response,feedback",
        help="Comma-separated phases: decision,response,feedback",
    )
    parser.add_argument("--n-splits", type=int, default=10, help="Maximum number of CV folds")
    parser.add_argument(
        "--n-permutations",
        type=int,
        default=200,
        help="Number of label permutations for the above-chance test",
    )
    parser.add_argument("--plot-only", action="store_true", help="Reuse existing CSV/JSON results and only replot")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    subjects = resolve_subjects(args.subjects)
    targets = resolve_csv_argument(
        args.targets,
        allowed=TARGET_CHOICES,
        default=["current_self"],
    )
    phases = resolve_csv_argument(
        args.phases,
        allowed=tuple(PHASE_SPECS.keys()),
        default=["decision", "response", "feedback"],
    )

    base_out_dir = Path(config.OUTPUT_DIR).parent / "decoding"
    base_out_dir.mkdir(parents=True, exist_ok=True)

    for target in targets:
        out_dir = target_output_dir(base_out_dir, target)

        if args.plot_only:
            summary_path = out_dir / "decoding_summary.json"
            if not summary_path.exists():
                raise FileNotFoundError(f"Summary not found for target={target}: {summary_path}")
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            rows_by_phase = {phase: [] for phase in phases}
        else:
            rows_by_phase, summary = run_decoding(
                subjects=subjects,
                target=target,
                phases=phases,
                n_splits=args.n_splits,
                n_permutations=args.n_permutations,
                random_state=args.random_state,
            )

        _save_target_outputs(out_dir, rows_by_phase, summary, plot_only=args.plot_only)

        print(f"\n=== {TARGET_DISPLAY_NAMES[target]} ===")
        print(f"Output directory: {out_dir}")
        print(f"N subject/person: {summary['n_subject_person']}")
        for phase in phases:
            print(
                f"{phase}: mean={summary[f'mean_accuracy_{phase}']:.3f}, "
                f"one-tailed p={summary[f'group_pvalue_one_tailed_gt_chance_{phase}']:.6f}"
            )


if __name__ == "__main__":
    main()
