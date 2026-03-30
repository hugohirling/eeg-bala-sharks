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

# Set up Rich console and progress bar
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


def _get_decision_bin_windows(n_bins: int = 8) -> tuple[list[float], list[float], list[float]]:
    bin_starts = [0.25 * index for index in range(n_bins)]
    bin_ends = [start + 0.25 for start in bin_starts]
    bin_centers = [(start + end) / 2.0 for start, end in zip(bin_starts, bin_ends)]
    return bin_starts, bin_ends, bin_centers

def _get_response_bin_windows(n_bins: int = 8) -> tuple[list[float], list[float], list[float]]:
    bin_starts = [0.25 * index for index in range(n_bins)]
    bin_ends = [start + 0.25 for start in bin_starts]
    bin_centers = [(start + end) / 2.0 for start, end in zip(bin_starts, bin_ends)]
    return bin_starts, bin_ends, bin_centers

def _get_feedback_bin_windows(n_bins: int = 4) -> tuple[list[float], list[float], list[float]]:
    bin_starts = [0.25 * index for index in range(n_bins)]
    bin_ends = [start + 0.25 for start in bin_starts]
    bin_centers = [(start + end) / 2.0 for start, end in zip(bin_starts, bin_ends)]
    return bin_starts, bin_ends, bin_centers


def _resolve_subjects(subjects_arg: str | None) -> list[str]:
    if subjects_arg:
        return [part.strip() for part in subjects_arg.split(",") if part.strip()]
    print(list(config.SUBJECTS))
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
    # In this dataset, P1/P2 EEG streams map to prefixes "2-"/"1-".
    # This tells us which behavioral player column to decode.
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

    epochs = mne.read_epochs(str(epoch_path), preload=True)

    # Decision phase from paper: 2 s after decision screen onset.
    # We decode 0-2 s to focus on decision formation and avoid pre-onset baseline.
    decision_epochs = epochs.copy().crop(tmin=0.0, tmax=2.0)
    decision_epochs.pick("eeg")

    return decision_epochs.get_data(copy=True)

def _load_response_features(subject_id: str, person: str) -> np.ndarray:
    epoch_path = _get_epoch_path(subject_id, person)
    if not epoch_path.exists():
        raise FileNotFoundError(f"Missing epoch file: {epoch_path}")

    epochs = mne.read_epochs(str(epoch_path), preload=True)

    # We decode 2-4 s to focus on response formation and avoid pre-onset baseline.
    response_epochs = epochs.copy().crop(tmin=2.0, tmax=4.0)
    response_epochs.pick("eeg")

    return response_epochs.get_data(copy=True)

def _load_feedback_features(subject_id: str, person: str) -> np.ndarray:
    epoch_path = _get_epoch_path(subject_id, person)
    if not epoch_path.exists():
        raise FileNotFoundError(f"Missing epoch file: {epoch_path}")

    epochs = mne.read_epochs(str(epoch_path), preload=True)

    # We decode 4-5s to focus on feedback formation and avoid pre-onset baseline.
    feedback_epochs = epochs.copy().crop(tmin=4.0, tmax=5.0)
    feedback_epochs.pick("eeg")

    return feedback_epochs.get_data(copy=True)


def _decode_subject_person(
    subject_id: str,
    person: str,
    n_splits: int,
    n_permutations: int,
    random_state: int,
) -> dict:
    
    X_decision_full = _load_decision_features(subject_id, person)
    X_response_full = _load_response_features(subject_id, person)
    X_feedback_full = _load_feedback_features(subject_id, person)

    y_full = _load_labels(subject_id, person)

    # Keep only the overlap if lengths differ.
    #Decision
    n_decision = min(len(X_decision_full), len(y_full))
    X_decision_full = X_decision_full[:n_decision]
    y_decision = y_full[:n_decision]
    #Response
    n_response = min(len(X_response_full), len(y_full))
    X_response_full = X_response_full[:n_response]
    y_response = y_full[:n_response]
    #Feedback
    n_feedback = min(len(X_feedback_full), len(y_full))
    X_feedback_full = X_feedback_full[:n_feedback]
    y_feedback = y_full[:n_feedback]

    # Remove no-response trials (0), decode only rock/paper/scissors.
    mask = np.isin(y_full, [1, 2, 3])
    X_decision_full = X_decision_full[mask]
    X_response_full = X_response_full[mask]
    X_feedback_full = X_feedback_full[mask]
    y_decision = y_decision[mask]
    y_response = y_response[mask]
    y_feedback = y_feedback[mask]

    if len(np.unique(y_decision)) < 3 or len(np.unique(y_response)) < 3 or len(np.unique(y_feedback)) < 3:
        raise ValueError(
            f"Not enough classes for sub-{subject_id} {person}. Classes found: {sorted(set(y_decision.tolist()))}"
        )

    class_counts = {RESP_CODE_TO_NAME[int(code)]: int(np.sum(y_decision == code)) for code in sorted(np.unique(y_decision))}

    clf = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("lda", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
        ]
    )

    def _calculate_bin_scores(X_full: np.ndarray, y: np.ndarray, bin_func: callable, n_bins: int) -> tuple[list[float], list[float]]:
        # Match paper idea: decode in 250 ms bins. We use the mean EEG value inside each
        # bin per channel, then aggregate across bins in the 0-2 s decision window.
        n_times = X_full.shape[2]
        bin_edges = np.linspace(0, n_times, n_bins + 1, dtype=int)
        bin_starts_s, bin_ends_s, bin_centers_s = bin_func(n_bins)

        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        bin_scores: list[float] = []
        bin_perm_pvalues: list[float] = []

        for i in range(n_bins):
            start = bin_edges[i]
            stop = bin_edges[i + 1]
            X_bin = X_full[:, :, start:stop].mean(axis=2)

            cv_scores_bin = cross_val_score(clf, X_bin, y, cv=cv, scoring="accuracy")
            bin_scores.append(float(np.mean(cv_scores_bin)))

            observed_acc, _perm_scores, pvalue = permutation_test_score(
                clf,
                X_bin,
                y,
                scoring="accuracy",
                cv=cv,
                n_permutations=n_permutations,
                random_state=random_state,
                n_jobs=1,
            )
            # Keep both the mean CV score and permutation score diagnostics.
            if abs(observed_acc - bin_scores[-1]) > 0.2:
                pass
            bin_perm_pvalues.append(float(pvalue))

        mean_acc = float(np.mean(bin_scores))
        min_perm_pvalue = float(np.min(bin_perm_pvalues))

        return {
            "subject": subject_id,
            "person": person,
            "n_trials_used": int(len(y)),
            "class_counts": class_counts,
            "cv_accuracy_mean": mean_acc,
            "cv_accuracy_std": float(np.std(bin_scores)),
            "n_time_bins": int(n_bins),
            "chance_level": 1.0 / 3.0,
            "above_chance": bool(mean_acc > (1.0 / 3.0)),
            "permutation_accuracy": mean_acc,
            "permutation_pvalue": min_perm_pvalue,
            "n_permutations": int(n_permutations),
            "bin_starts_s": bin_starts_s,
            "bin_ends_s": bin_ends_s,
            "bin_centers_s": bin_centers_s,
            "bin_scores": [float(score) for score in bin_scores],
            "bin_permutation_pvalues": [float(pvalue) for pvalue in bin_perm_pvalues],
        }
    
    decision_results = _calculate_bin_scores(
        X_decision_full, y_decision, _get_decision_bin_windows, 8)
    response_results = _calculate_bin_scores(
        X_response_full, y_response, _get_response_bin_windows, 8)
    feedback_results = _calculate_bin_scores(
        X_feedback_full, y_feedback, _get_feedback_bin_windows, 4)
    
    return {
        "decision": decision_results,
        "response": response_results,
        "feedback": feedback_results,
    }

    
def run_decoding(
    subjects: list[str],
    n_splits: int,
    n_permutations: int,
    random_state: int,
) -> tuple[list[dict], dict]:
    rows: list[dict] = []

    

    with Live(progress, console=console, refresh_per_second=1) as live:
        # Calculate total items to process (subjects × persons)
        total_items = len(subjects) * 2
        task_id = progress.add_task("Decoding subjects", total=total_items)

        for subject_id in subjects:
            for person in ["P1", "P2"]:
                try:
                    # Update progress description
                    progress.update(task_id, description=f"Decoding sub-{subject_id} {person}")
                    live.refresh()

                    row = _decode_subject_person(
                        subject_id=subject_id,
                        person=person,
                        n_splits=n_splits,
                        n_permutations=n_permutations,
                        random_state=random_state,
                    )
                    rows.append(row)
                    console.print(
                        f"[green]✓[/green] sub-{subject_id} {person}: "
                        f"decision acc={row['decision']['cv_accuracy_mean']:.3f} (p={row['decision']['permutation_pvalue']:.4f}), "
                        f"response acc={row['response']['cv_accuracy_mean']:.3f} (p={row['response']['permutation_pvalue']:.4f}), "
                        f"feedback acc={row['feedback']['cv_accuracy_mean']:.3f} (p={row['feedback']['permutation_pvalue']:.4f})"
                    )
                except Exception as exc:
                    console.print(f"[yellow]⚠[/yellow] sub-{subject_id} {person}: skipped ({exc})")

                # Advance progress bar
                progress.advance(task_id)
                live.refresh()

    if not rows:
        raise RuntimeError("No valid subject/person results were produced.")

    acc_decision = np.array([row["decision"]["cv_accuracy_mean"] for row in rows], dtype=float)
    acc_response = np.array([row["response"]["cv_accuracy_mean"] for row in rows], dtype=float)
    acc_feedback = np.array([row["feedback"]["cv_accuracy_mean"] for row in rows], dtype=float)
    chance = 1.0 / 3.0
    t_stat_decision, t_p_two_tailed_decision = ttest_1samp(acc_decision, popmean=chance)
    t_stat_response, t_p_two_tailed_response = ttest_1samp(acc_response, popmean=chance)
    t_stat_feedback, t_p_two_tailed_feedback = ttest_1samp(acc_feedback, popmean=chance)

    summary = {
        "n_subject_person": int(len(rows)),
        "mean_accuracy_decision": float(np.mean(acc_decision)),
        "mean_accuracy_response": float(np.mean(acc_response)),
        "mean_accuracy_feedback": float(np.mean(acc_feedback)),
        "std_accuracy_decision": float(np.std(acc_decision)),
        "std_accuracy_response": float(np.std(acc_response)),
        "std_accuracy_feedback": float(np.std(acc_feedback)),
        "chance_level": chance,
        "n_above_chance_decision": int(np.sum(acc_decision > chance)),
        "n_above_chance_response": int(np.sum(acc_response > chance)),
        "n_above_chance_feedback": int(np.sum(acc_feedback > chance)),
        "group_t_stat_decision": float(t_stat_decision),
        "group_t_stat_response": float(t_stat_response),
        "group_t_stat_feedback": float(t_stat_feedback),
        "group_pvalue_two_tailed_decision": float(t_p_two_tailed_decision),
        "group_pvalue_two_tailed_response": float(t_p_two_tailed_response),
        "group_pvalue_two_tailed_feedback": float(t_p_two_tailed_feedback),
        "group_pvalue_one_tailed_gt_chance_decision": float(t_p_two_tailed_decision / 2.0) if t_stat_decision > 0 else 1.0,
        "group_pvalue_one_tailed_gt_chance_response": float(t_p_two_tailed_response / 2.0) if t_stat_response > 0 else 1.0,
        "group_pvalue_one_tailed_gt_chance_feedback": float(t_p_two_tailed_feedback / 2.0) if t_stat_feedback > 0 else 1.0,
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
        for cls_name in ["rock", "paper", "scissors"]:
            flat[f"n_{cls_name}"] = int(class_counts.get(cls_name, 0))
        flat_rows.append(flat)
    return pd.DataFrame(flat_rows)


def _to_timecourse_dataframe(rows: list[dict]) -> pd.DataFrame:
    records = []
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


def _save_accuracy_plot(df: pd.DataFrame, summary: dict, out_dir: Path) -> Path:
    labels = [f"sub-{row.subject}_{row.person}" for row in df.itertuples(index=False)]
    values = df["cv_accuracy_mean"].to_numpy(dtype=float)
    errors = df["cv_accuracy_std"].to_numpy(dtype=float)
    chance = float(summary["chance_level"])
    mean_acc = float(summary["mean_accuracy"])

    colors = ["#1f77b4" if value > chance else "#9aa0a6" for value in values]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(labels))
    ax.bar(x, values, yerr=errors, capsize=4, color=colors, edgecolor="black", linewidth=0.8)
    ax.axhline(chance, color="#c62828", linestyle="--", linewidth=1.5, label=f"Chance = {chance:.3f}")
    ax.axhline(mean_acc, color="#2e7d32", linestyle=":", linewidth=1.5, label=f"Mean = {mean_acc:.3f}")

    ax.set_title("Decision-Phase Decoding Accuracy")
    ax.set_ylabel("Classification accuracy")
    ax.set_xlabel("Subject / player")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylim(0.28, max(0.4, float(np.max(values + errors)) + 0.01))
    ax.grid(axis="y", alpha=0.25)
    ax.legend()

    subtitle = (
        f"One-tailed group p = {summary['group_pvalue_one_tailed_gt_chance']:.4f}, "
        f"above chance = {summary['n_above_chance']}/{summary['n_subject_person']}"
    )
    fig.text(0.5, 0.01, subtitle, ha="center", va="bottom")
    fig.tight_layout(rect=(0, 0.04, 1, 1))

    plot_path = out_dir / "decision_phase_decoding_accuracy.png"
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return plot_path


def _save_timecourse_plot(timecourse_df: pd.DataFrame, summary: dict, out_dir: Path) -> Path:
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
    ax.plot(x, y, color="#1f77b4", marker="o", linewidth=2.0, label="Mean accuracy")
    ax.fill_between(x, y - yerr, y + yerr, color="#1f77b4", alpha=0.18, label="±1 SD")
    ax.axhline(chance, color="#c62828", linestyle="--", linewidth=1.5, label=f"Chance = {chance:.3f}")

    ax.set_title("Decision-Phase Decoding Time Course")
    ax.set_xlabel("Time from decision onset (s)")
    ax.set_ylabel("Classification accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{start:.2f}-{end:.2f}" for start, end in zip(grouped["bin_start_s"], grouped["bin_end_s"])], rotation=45, ha="right")
    ax.set_ylim(min(chance - 0.03, float(np.min(y - yerr)) - 0.01), max(0.4, float(np.max(y + yerr)) + 0.01))
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()

    plot_path = out_dir / "decision_phase_decoding_timecourse.png"
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return plot_path


def _save_timecourse_plot_by_person(
    timecourse_df: pd.DataFrame,
    summary: dict,
    out_dir: Path,
    person: str,
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
    ax.plot(x, y, color="#1f77b4", marker="o", linewidth=2.0, label=f"Mean accuracy ({person})")
    ax.fill_between(x, y - yerr, y + yerr, color="#1f77b4", alpha=0.18, label="±1 SD")
    ax.axhline(chance, color="#c62828", linestyle="--", linewidth=1.5, label=f"Chance = {chance:.3f}")

    ax.set_title(f"Decision-Phase Decoding Time Course ({person})")
    ax.set_xlabel("Time from decision onset (s)")
    ax.set_ylabel("Classification accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{start:.2f}-{end:.2f}" for start, end in zip(grouped["bin_start_s"], grouped["bin_end_s"])],
        rotation=45,
        ha="right",
    )
    ax.set_ylim(min(chance - 0.03, float(np.min(y - yerr)) - 0.01), max(0.4, float(np.max(y + yerr)) + 0.01))
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()

    plot_path = out_dir / f"decision_phase_decoding_timecourse_{person}.png"
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return plot_path


def main() -> None:

    mne.set_config("MNE_LOGGING_LEVEL", "ERROR")  # Ensure MNE logging is configured before any MNE calls.

    parser = argparse.ArgumentParser(
        description=(
            "Decode rock/paper/scissors from EEG during the Decision phase (0-2 s after decision onset)."
        )
    )
    parser.add_argument("--subjects", type=str, default=None, help="Comma-separated subject IDs, e.g. 01,02,03")
    parser.add_argument("--n-splits", type=int, default=10, help="Number of CV folds (default: 10)")
    parser.add_argument(
        "--n-permutations",
        type=int,
        default=200,
        help="Number of label permutations for the above-chance test (default: 200)",
    )
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    subjects = _resolve_subjects(args.subjects)

    rows, summary = run_decoding(
        subjects=subjects,
        n_splits=args.n_splits,
        n_permutations=args.n_permutations,
        random_state=args.random_state,
    )

    out_dir = Path(config.OUTPUT_DIR).parent / "decoding"
    out_dir.mkdir(parents=True, exist_ok=True)

    df_decision = _to_dataframe(rows["decision"])
    df_response = _to_dataframe(rows["response"])
    df_feedback = _to_dataframe(rows["feedback"])
    csv_path_decision = out_dir / "decision_phase_decoding_results.csv"
    csv_path_response = out_dir / "response_phase_decoding_results.csv"
    csv_path_feedback = out_dir / "feedback_phase_decoding_results.csv"
    df_decision.to_csv(csv_path_decision, index=False)
    df_response.to_csv(csv_path_response, index=False)
    df_feedback.to_csv(csv_path_feedback, index=False)

    timecourse_df_decision = _to_timecourse_dataframe(rows["decision"])
    timecourse_df_response = _to_timecourse_dataframe(rows["response"])
    timecourse_df_feedback = _to_timecourse_dataframe(rows["feedback"])
    timecourse_csv_path_decision = out_dir / "decision_phase_decoding_timecourse.csv"
    timecourse_csv_path_response = out_dir / "response_phase_decoding_timecourse.csv"
    timecourse_csv_path_feedback = out_dir / "feedback_phase_decoding_timecourse.csv"
    timecourse_df_decision.to_csv(timecourse_csv_path_decision, index=False)
    timecourse_df_response.to_csv(timecourse_csv_path_response, index=False)
    timecourse_df_feedback.to_csv(timecourse_csv_path_feedback, index=False)

    json_path_decision = out_dir / "decision_phase_decoding_summary.json"
    json_path_response = out_dir / "response_phase_decoding_summary.json"
    json_path_feedback = out_dir / "feedback_phase_decoding_summary.json"
    json_path_decision.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    json_path_response.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    json_path_feedback.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    plot_path_decision = _save_accuracy_plot(df_decision, summary, out_dir)
    plot_path_response = _save_accuracy_plot(df_response, summary, out_dir)
    plot_path_feedback = _save_accuracy_plot(df_feedback, summary, out_dir)
    timecourse_plot_path_decision = _save_timecourse_plot(timecourse_df_decision, summary, out_dir)
    timecourse_plot_path_response = _save_timecourse_plot(timecourse_df_response, summary, out_dir)
    timecourse_plot_path_feedback = _save_timecourse_plot(timecourse_df_feedback, summary, out_dir)
    timecourse_plot_p1_path_decision = _save_timecourse_plot_by_person(timecourse_df_decision, summary, out_dir, "P1")
    timecourse_plot_p1_path_response = _save_timecourse_plot_by_person(timecourse_df_response, summary, out_dir, "P1")
    timecourse_plot_p1_path_feedback = _save_timecourse_plot_by_person(timecourse_df_feedback, summary, out_dir, "P1")
    timecourse_plot_p2_path_decision = _save_timecourse_plot_by_person(timecourse_df_decision, summary, out_dir, "P2")
    timecourse_plot_p2_path_response = _save_timecourse_plot_by_person(timecourse_df_response, summary, out_dir, "P2")
    timecourse_plot_p2_path_feedback = _save_timecourse_plot_by_person(timecourse_df_feedback, summary, out_dir, "P2")

    print("\n=== Group Summary ===")
    print(f"N subject/person: {summary['n_subject_person']}")
    print(f"Mean accuracy (decision): {summary['mean_accuracy_decision']:.3f} (chance: {summary['chance_level']:.3f})")
    print(f"Mean accuracy (response): {summary['mean_accuracy_response']:.3f} (chance: {summary['chance_level']:.3f})")
    print(f"Mean accuracy (feedback): {summary['mean_accuracy_feedback']:.3f} (chance: {summary['chance_level']:.3f})")
    print(f"Above chance count (decision): {summary['n_above_chance_decision']}/{summary['n_subject_person']}")
    print(f"Above chance count (response): {summary['n_above_chance_response']}/{summary['n_subject_person']}")
    print(f"Above chance count (feedback): {summary['n_above_chance_feedback']}/{summary['n_subject_person']}")
    print(
        "One-tailed group p (accuracy decision > chance): "
        f"{summary['group_pvalue_one_tailed_gt_chance_decision']:.6f}"
    )
    print(
        "One-tailed group p (accuracy response > chance): "
        f"{summary['group_pvalue_one_tailed_gt_chance_response']:.6f}"
    )
    print(
        "One-tailed group p (accuracy feedback > chance): "
        f"{summary['group_pvalue_one_tailed_gt_chance_feedback']:.6f}"
    )
    print(f"Saved per-subject results: {csv_path_decision}, {csv_path_response}, {csv_path_feedback}")
    print(f"Saved time-course results: {timecourse_csv_path_decision}, {timecourse_csv_path_response}, {timecourse_csv_path_feedback}")
    print(f"Saved summary: {json_path_decision}, {json_path_response}, {json_path_feedback}")
    print(f"Saved plot: {plot_path_decision}, {plot_path_response}, {plot_path_feedback}")
    print(f"Saved time-course plot: {timecourse_plot_path_decision}, {timecourse_plot_path_response}, {timecourse_plot_path_feedback}")
    if timecourse_plot_p1_path_decision is not None:
        print(f"Saved time-course plot (P1): {timecourse_plot_p1_path_decision}")
    if timecourse_plot_p1_path_response is not None:
        print(f"Saved time-course plot (P1): {timecourse_plot_p1_path_response}")
    if timecourse_plot_p1_path_feedback is not None:
        print(f"Saved time-course plot (P1): {timecourse_plot_p1_path_feedback}")
    if timecourse_plot_p2_path_decision is not None:
        print(f"Saved time-course plot (P2): {timecourse_plot_p2_path_decision}")
    if timecourse_plot_p2_path_response is not None:
        print(f"Saved time-course plot (P2): {timecourse_plot_p2_path_response}")
    if timecourse_plot_p2_path_feedback is not None:
        print(f"Saved time-course plot (P2): {timecourse_plot_p2_path_feedback}")


if __name__ == "__main__":
    main()
