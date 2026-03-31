# This file has been commented using GitHub Copilot with the Grok Code Fast 1 model.

from pathlib import Path
import sys

import mne
from mne_icalabel import label_components

CURRENT_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = CURRENT_DIR.parent
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s"
)

LOGGER = logging.getLogger(__name__)

from preprocessing import config
from helper.general.helper_functions import get_step_io_files, save_current_step_file


def _normalize_iclabel_name(label):
    """
    Normalizes the ICA label name by converting to lowercase and replacing underscores with spaces.

    Args:
        label (str): The original label name.

    Returns:
        str: The normalized label name.
    """
    return str(label).strip().lower().replace("_", " ")


def _classify_artifact_components(raw, ica):
    """
    Classifies ICA components as artifacts using ICLabel and determines which to exclude.

    This function uses the ICLabel method to label ICA components and excludes those
    that match artifact labels with sufficient probability.

    Args:
        raw (mne.io.Raw): The raw EEG data.
        ica (mne.preprocessing.ICA): The fitted ICA object.

    Returns:
        tuple: (excluded_components, labels, probabilities)
            - excluded_components (list): Indices of components to exclude.
            - labels (list): Labels for each component.
            - probabilities (list): Probabilities for each label.
    """
    result = label_components(raw, ica, method=config.ICA_LABEL_METHOD)
    labels = result["labels"]
    probabilities = result["y_pred_proba"]
    # Create a set of normalized artifact labels for comparison
    artifact_labels = {_normalize_iclabel_name(label) for label in config.ICA_ARTIFACT_LABELS}

    excluded = []
    label_index_map = {}
    # Loop through each component to classify and decide exclusion
    for component_idx, (label, probability) in enumerate(zip(labels, probabilities)):
        normalized_label = _normalize_iclabel_name(label)
        label_index_map.setdefault(normalized_label, []).append(component_idx)
        LOGGER.info(
            "ICA component %02d labeled as %s (p=%.3f)",
            component_idx,
            normalized_label,
            float(probability),
        )
        # Exclude if it's an artifact with sufficient probability
        if normalized_label in artifact_labels and float(probability) >= float(config.ICA_LABEL_MIN_PROBA):
            excluded.append(component_idx)

    ica.labels_ = label_index_map
    return excluded, labels, probabilities


def run_ica(raw):
    """
    Runs Independent Component Analysis (ICA) on the raw EEG data.

    This function filters the data, fits an ICA model, classifies components,
    and applies the ICA to remove artifacts.

    Args:
        raw (mne.io.Raw): The raw EEG data to process.

    Returns:
        tuple: (cleaned_raw, ica)
            - cleaned_raw (mne.io.Raw): The ICA-cleaned raw data.
            - ica (mne.preprocessing.ICA): The fitted ICA object.
    """
    print("Running ICA")

    # Filter the data for ICA fitting to focus on frequencies above 1 Hz
    raw_for_fit = raw.copy().filter(l_freq=1.0, h_freq=None)

    # Determine the number of ICA components based on available EEG channels
    n_eeg = len(mne.pick_types(raw_for_fit.info, eeg=True, exclude="bads"))
    n_components = min(config.ICA_N_COMPONENTS, n_eeg) if n_eeg > 0 else config.ICA_N_COMPONENTS

    # Initialize and fit the ICA model
    ica = mne.preprocessing.ICA(
        n_components=n_components,
        random_state=97,
        max_iter=config.ICA_MAX_ITER,
        method="fastica",
    )
    ica.fit(raw_for_fit)

    # Classify components and determine which to exclude
    excluded, labels, probabilities = _classify_artifact_components(raw_for_fit, ica)
    ica.exclude = sorted(set(excluded))

    # Log ICA configuration and results
    LOGGER.info("ICA label threshold: %.2f", float(config.ICA_LABEL_MIN_PROBA))
    LOGGER.info("ICA artifact labels: %s", ", ".join(config.ICA_ARTIFACT_LABELS))
    LOGGER.info(
        "ICA component labels: %s",
        ", ".join(
            f"{idx}:{_normalize_iclabel_name(label)}@{float(probability):.2f}"
            for idx, (label, probability) in enumerate(zip(labels, probabilities))
        ),
    )
    LOGGER.info(f"ICA components excluded: {ica.exclude}")
    # Apply ICA to clean the data
    cleaned = raw.copy()
    ica.apply(cleaned)
    return cleaned, ica


def process_subject(subject_id):
    """
    Processes ICA for a given subject, handling both P1 and P2 data.

    This function loads the filtered data, runs ICA, saves the cleaned data
    and the ICA decomposition.

    Args:
        subject_id (str): The subject identifier.

    Returns:
        list: List of tuples (person, out_path, ica_path) for each person.
    """
    outputs = []
    for person in ["P1", "P2"]:
        # Get input and output file paths for the current person
        path_in, _ = get_step_io_files(
            subject_id,
            __file__,
            person=person,
            pipeline_steps=config.PIPELINE_STEPS,
            step_output_suffixes=config.STEP_OUTPUT_SUFFIXES,
        )
        if path_in is None:
            raise ValueError("ICA step requires filtered input files")

        LOGGER.info(f"Loading previous step file ({person}): {path_in}")
        raw = mne.io.read_raw_fif(path_in, preload=True)
        # Run ICA on the loaded data
        cleaned, ica = run_ica(raw)

        # Save the cleaned data
        out_path = save_current_step_file(
            cleaned,
            subject_id,
            __file__,
            person=person,
            step_output_suffixes=config.STEP_OUTPUT_SUFFIXES,
        )
        LOGGER.info(f"Saved ICA-cleaned file ({person}) to: {out_path}")

        # Save the ICA decomposition
        ica_path = config.ICA_DIR / f"sub-{subject_id}_{person}_ica.fif"
        ica.save(ica_path, overwrite=True)
        LOGGER.info(f"Saved ICA decomposition ({person}) to: {ica_path}")

        outputs.append((person, out_path, ica_path))

    return outputs


if __name__ == "__main__":
    for i, subj in enumerate(config.SUBJECTS):
        process_subject(subj)
        LOGGER.info(f"PROGRESS:{i + 1}")
