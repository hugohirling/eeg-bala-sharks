from __future__ import annotations

import sys
from pathlib import Path

import mne
import numpy as np
import pandas as pd

CURRENT_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = CURRENT_DIR.parent
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from preprocessing import config

RESP_CODE_TO_NAME = {1: "rock", 2: "paper", 3: "scissors"}
STRATEGY_LABELS = {1: "target", 2: "other"}

TARGET_CHOICES = (
    "current_self",
    "current_other",
    "previous_self",
    "previous_other",
)

TARGET_DISPLAY_NAMES = {
    "current_self": "Current own decision",
    "current_other": "Current opponent decision",
    "previous_self": "Previous own decision",
    "previous_other": "Previous opponent decision",
}

STRATEGY_CHOICES = (
    "stay_vs_switch",
    "win_stay_vs_other",
    "lose_shift_vs_other",
)

STRATEGY_DISPLAY_NAMES = {
    "stay_vs_switch": "Stay vs switch",
    "win_stay_vs_other": "Win-stay vs other",
    "lose_shift_vs_other": "Lose-shift vs other",
}

PHASE_SPECS = {
    "decision": {"tmin": 0.0, "tmax": 2.0, "n_bins": 8},
    "response": {"tmin": 2.0, "tmax": 4.0, "n_bins": 8},
    "feedback": {"tmin": 4.0, "tmax": 5.0, "n_bins": 4},
}


# -----------------------------------------------------------------------------
# Function: resolve_subjects
# Purpose:
#   Parse a comma-separated subject string or fall back to configured subjects.
# Inputs:
#   - subjects_arg: optional CSV string of subject IDs.
# Outputs:
#   - List of normalized subject IDs.
# -----------------------------------------------------------------------------
def resolve_subjects(subjects_arg: str | None) -> list[str]:
    if subjects_arg:
        # Trim whitespace and ignore empty CSV entries.
        return [part.strip() for part in subjects_arg.split(",") if part.strip()]
    # Default behavior uses the global subject list from preprocessing config.
    return list(config.SUBJECTS)


# -----------------------------------------------------------------------------
# Function: resolve_csv_argument
# Purpose:
#   Validate and parse generic comma-separated CLI arguments against allowed values.
# Inputs:
#   - arg_value: optional CSV string from CLI.
#   - allowed: tuple of permitted tokens.
#   - default: fallback list when argument is absent.
# Outputs:
#   - List of parsed argument values.
# -----------------------------------------------------------------------------
def resolve_csv_argument(arg_value: str | None, *, allowed: tuple[str, ...], default: list[str]) -> list[str]:
    if not arg_value:
        return default
    # Split and normalize user-provided CSV values.
    values = [part.strip() for part in arg_value.split(",") if part.strip()]
    # Explicitly reject unsupported tokens to prevent silent misconfiguration.
    invalid = sorted(set(values) - set(allowed))
    if invalid:
        raise ValueError(f"Unsupported values: {invalid}. Allowed: {list(allowed)}")
    return values


# -----------------------------------------------------------------------------
# Function: get_events_tsv_path
# Purpose:
#   Build the canonical BIDS events TSV path for one subject.
# Inputs:
#   - subject_id: subject identifier used in BIDS naming.
# Outputs:
#   - Absolute/relative Path to subject events TSV.
# -----------------------------------------------------------------------------
def get_events_tsv_path(subject_id: str) -> Path:
    return (
        Path(config.BIDS_ROOT)
        / f"sub-{subject_id}"
        / "eeg"
        / f"sub-{subject_id}_task-RPS_events.tsv"
    )


# -----------------------------------------------------------------------------
# Function: get_epoch_path
# Purpose:
#   Build the path to a saved epoch FIF file for one subject and player role.
# Inputs:
#   - subject_id: subject identifier.
#   - person: player key (e.g., player1/player2 style key from config mapping).
# Outputs:
#   - Path to epoch FIF file.
# -----------------------------------------------------------------------------
def get_epoch_path(subject_id: str, person: str) -> Path:
    return Path(config.OUTPUT_DIR) / f"sub-{subject_id}_{person}_epoch.fif"


# -----------------------------------------------------------------------------
# Function: player_response_column
# Purpose:
#   Resolve which events column contains the current person's response.
# Inputs:
#   - person: configured player identity key.
# Outputs:
#   - Either "player1_resp" or "player2_resp".
# -----------------------------------------------------------------------------
def player_response_column(person: str) -> str:
    prefix = config.PLAYER_PREFIX_MAP[person]
    if prefix.startswith("1"):
        return "player1_resp"
    if prefix.startswith("2"):
        return "player2_resp"
    raise ValueError(f"Unexpected player prefix for {person}: {prefix}")


# -----------------------------------------------------------------------------
# Function: other_player_response_column
# Purpose:
#   Return the response column corresponding to the opponent of the given person.
# Inputs:
#   - person: configured player identity key.
# Outputs:
#   - Opponent response column name.
# -----------------------------------------------------------------------------
def other_player_response_column(person: str) -> str:
    current_col = player_response_column(person)
    # Opponent column is always the complement of the current player's column.
    return "player2_resp" if current_col == "player1_resp" else "player1_resp"


# -----------------------------------------------------------------------------
# Function: load_events_df
# Purpose:
#   Read one subject's events TSV into a DataFrame.
# Inputs:
#   - subject_id: subject identifier.
# Outputs:
#   - Parsed events DataFrame.
# Notes:
#   - Raises early if the events file is missing.
# -----------------------------------------------------------------------------
def load_events_df(subject_id: str) -> pd.DataFrame:
    events_path = get_events_tsv_path(subject_id)
    if not events_path.exists():
        raise FileNotFoundError(f"Missing events file: {events_path}")
    return pd.read_csv(events_path, sep="\t")


# -----------------------------------------------------------------------------
# Function: load_phase_features
# Purpose:
#   Load epoch data and extract EEG feature arrays per phase.
# Inputs:
#   - subject_id: subject identifier.
#   - person: player identity key.
#   - phases: list of phase names to extract.
# Outputs:
#   - Dict mapping phase name to ndarray of EEG data.
# -----------------------------------------------------------------------------
def load_phase_features(subject_id: str, person: str, phases: list[str]) -> dict[str, np.ndarray]:
    epoch_path = get_epoch_path(subject_id, person)
    if not epoch_path.exists():
        raise FileNotFoundError(f"Missing epoch file: {epoch_path}")

    # Load all epochs once, then crop phase-specific windows from a copy.
    epochs = mne.read_epochs(str(epoch_path), preload=True)
    features: dict[str, np.ndarray] = {}
    for phase in phases:
        spec = PHASE_SPECS[phase]
        # Crop to the configured phase interval and retain EEG channels only.
        phase_epochs = epochs.copy().crop(tmin=spec["tmin"], tmax=spec["tmax"])
        phase_epochs.pick("eeg")
        features[phase] = phase_epochs.get_data(copy=True)
    return features


# -----------------------------------------------------------------------------
# Function: previous_trial
# Purpose:
#   Shift a label vector by one trial to create previous-trial targets.
# Inputs:
#   - labels: trial-wise labels.
#   - fill_value: value used for the first trial (no previous trial exists).
# Outputs:
#   - Shifted ndarray with same shape as input.
# -----------------------------------------------------------------------------
def previous_trial(labels: np.ndarray, fill_value: int = -1) -> np.ndarray:
    shifted = np.full(labels.shape, fill_value=fill_value, dtype=int)
    if len(labels) > 1:
        # Trial i receives label from trial i-1.
        shifted[1:] = labels[:-1]
    return shifted


# -----------------------------------------------------------------------------
# Function: person_trial_outcomes
# Purpose:
#   Convert raw match outcomes to person-centric outcomes (+1 win, -1 loss, 0 other).
# Inputs:
#   - events_df: events table containing "outcome" codes.
#   - person: player identity key.
# Outputs:
#   - Integer ndarray of person-centric outcomes per trial.
# -----------------------------------------------------------------------------
def person_trial_outcomes(events_df: pd.DataFrame, person: str) -> np.ndarray:
    own_col = player_response_column(person)
    own_is_player1 = own_col == "player1_resp"
    raw = events_df["outcome"].to_numpy(dtype=int)
    outcomes = np.zeros_like(raw, dtype=int)
    if own_is_player1:
        # Outcome code semantics are inverted depending on whether own player is P1/P2.
        outcomes[raw == 2] = 1
        outcomes[raw == 3] = -1
    else:
        outcomes[raw == 3] = 1
        outcomes[raw == 2] = -1
    return outcomes


# -----------------------------------------------------------------------------
# Function: match_status
# Purpose:
#   Summarize overall match performance for one person.
# Inputs:
#   - events_df: events table.
#   - person: player identity key.
# Outputs:
#   - "winner", "loser", or "tied_match".
# -----------------------------------------------------------------------------
def match_status(events_df: pd.DataFrame, person: str) -> str:
    outcomes = person_trial_outcomes(events_df, person)
    wins = int(np.sum(outcomes == 1))
    losses = int(np.sum(outcomes == -1))
    if wins > losses:
        return "winner"
    if losses > wins:
        return "loser"
    return "tied_match"


# -----------------------------------------------------------------------------
# Function: build_target_labels
# Purpose:
#   Build decoding labels for current/previous self/other targets.
# Inputs:
#   - events_df: events table.
#   - person: player identity key.
#   - target: one of TARGET_CHOICES.
# Outputs:
#   - Integer label vector aligned to trials.
# -----------------------------------------------------------------------------
def build_target_labels(events_df: pd.DataFrame, person: str, target: str) -> np.ndarray:
    own_col = player_response_column(person)
    other_col = other_player_response_column(person)

    if target == "current_self":
        return events_df[own_col].to_numpy(dtype=int)
    if target == "current_other":
        return events_df[other_col].to_numpy(dtype=int)
    if target == "previous_self":
        # Previous-trial variant uses one-step shifted current responses.
        return previous_trial(events_df[own_col].to_numpy(dtype=int))
    if target == "previous_other":
        return previous_trial(events_df[other_col].to_numpy(dtype=int))
    raise ValueError(f"Unsupported target: {target}")


# -----------------------------------------------------------------------------
# Function: build_strategy_labels
# Purpose:
#   Build binary strategy labels for specific strategy decoding tasks.
# Inputs:
#   - events_df: events table.
#   - person: player identity key.
#   - strategy_target: one of STRATEGY_CHOICES.
# Outputs:
#   - Integer labels where 1 is "target" class, 2 is "other", -1 invalid.
# -----------------------------------------------------------------------------
def build_strategy_labels(events_df: pd.DataFrame, person: str, strategy_target: str) -> np.ndarray:
    own = events_df[player_response_column(person)].to_numpy(dtype=int)
    prev_own = previous_trial(own)
    prev_outcome = previous_trial(person_trial_outcomes(events_df, person))

    labels = np.full(own.shape, fill_value=-1, dtype=int)
    # Valid pairs require both current and previous responses to be valid R/P/S codes.
    valid_pair = np.isin(own, [1, 2, 3]) & np.isin(prev_own, [1, 2, 3])

    if strategy_target == "stay_vs_switch":
        # Class 1: stay, Class 2: switch.
        labels[valid_pair & (own == prev_own)] = 1
        labels[valid_pair & (own != prev_own)] = 2
        return labels

    if strategy_target == "win_stay_vs_other":
        # Default valid trials to "other", then mark win-stay pattern as target class.
        labels[valid_pair] = 2
        labels[valid_pair & (prev_outcome == 1) & (own == prev_own)] = 1
        return labels

    if strategy_target == "lose_shift_vs_other":
        # Default valid trials to "other", then mark lose-shift pattern as target class.
        labels[valid_pair] = 2
        labels[valid_pair & (prev_outcome == -1) & (own != prev_own)] = 1
        return labels

    raise ValueError(f"Unsupported strategy target: {strategy_target}")


# -----------------------------------------------------------------------------
# Function: get_phase_bin_windows
# Purpose:
#   Return fixed-width 250 ms window boundaries and centers for a phase.
# Inputs:
#   - phase: phase key defined in PHASE_SPECS.
# Outputs:
#   - Tuple of (bin_starts, bin_ends, bin_centers) in seconds, phase-relative.
# -----------------------------------------------------------------------------
def get_phase_bin_windows(phase: str) -> tuple[list[float], list[float], list[float]]:
    n_bins = int(PHASE_SPECS[phase]["n_bins"])
    # Bins are phase-relative and use constant 0.25 s width.
    bin_starts = [0.25 * index for index in range(n_bins)]
    bin_ends = [start + 0.25 for start in bin_starts]
    bin_centers = [(start + end) / 2.0 for start, end in zip(bin_starts, bin_ends)]
    return bin_starts, bin_ends, bin_centers


# -----------------------------------------------------------------------------
# Function: target_output_dir
# Purpose:
#   Resolve output directory layout for decoding targets.
# Inputs:
#   - base_dir: default decoding output directory.
#   - target: one of TARGET_CHOICES.
# Outputs:
#   - Path where outputs for this target should be written/read.
# Notes:
#   - "current_self" is treated as the base legacy/default directory.
# -----------------------------------------------------------------------------
def target_output_dir(base_dir: Path, target: str) -> Path:
    if target == "current_self":
        return base_dir
    return base_dir / target
