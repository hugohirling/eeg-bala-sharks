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
    return str(label).strip().lower().replace("_", " ")


def _classify_artifact_components(raw, ica):
    result = label_components(raw, ica, method=config.ICA_LABEL_METHOD)
    labels = result["labels"]
    probabilities = result["y_pred_proba"]
    artifact_labels = {_normalize_iclabel_name(label) for label in config.ICA_ARTIFACT_LABELS}

    excluded = []
    label_index_map = {}
    for component_idx, (label, probability) in enumerate(zip(labels, probabilities)):
        normalized_label = _normalize_iclabel_name(label)
        label_index_map.setdefault(normalized_label, []).append(component_idx)
        LOGGER.info(
            "ICA component %02d labeled as %s (p=%.3f)",
            component_idx,
            normalized_label,
            float(probability),
        )
        if normalized_label in artifact_labels and float(probability) >= float(config.ICA_LABEL_MIN_PROBA):
            excluded.append(component_idx)

    ica.labels_ = label_index_map
    return excluded, labels, probabilities


def run_ica(raw):
    print("Running ICA")

    raw_for_fit = raw.copy().filter(l_freq=1.0, h_freq=None)

    n_eeg = len(mne.pick_types(raw_for_fit.info, eeg=True, exclude="bads"))
    n_components = min(config.ICA_N_COMPONENTS, n_eeg) if n_eeg > 0 else config.ICA_N_COMPONENTS

    ica = mne.preprocessing.ICA(
        n_components=n_components,
        random_state=97,
        max_iter=config.ICA_MAX_ITER,
        method="fastica",
    )
    ica.fit(raw_for_fit)

    excluded, labels, probabilities = _classify_artifact_components(raw_for_fit, ica)
    ica.exclude = sorted(set(excluded))

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
    cleaned = raw.copy()
    ica.apply(cleaned)
    return cleaned, ica


def process_subject(subject_id):
    outputs = []
    for person in ["P1", "P2"]:
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
        cleaned, ica = run_ica(raw)

        out_path = save_current_step_file(
            cleaned,
            subject_id,
            __file__,
            person=person,
            step_output_suffixes=config.STEP_OUTPUT_SUFFIXES,
        )
        LOGGER.info(f"Saved ICA-cleaned file ({person}) to: {out_path}")

        ica_path = config.ICA_DIR / f"sub-{subject_id}_{person}_ica.fif"
        ica.save(ica_path, overwrite=True)
        LOGGER.info(f"Saved ICA decomposition ({person}) to: {ica_path}")

        outputs.append((person, out_path, ica_path))

    return outputs


if __name__ == "__main__":
    for i, subj in enumerate(config.SUBJECTS):
        process_subject(subj)
        LOGGER.info(f"PROGRESS:{i + 1}")
