from pathlib import Path
import sys

import mne

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

    eog_channels = mne.pick_types(raw.info, eog=True)
    if len(eog_channels) > 0:
        for eog_idx in eog_channels:
            eog_name = raw.ch_names[eog_idx]
            bads, _ = ica.find_bads_eog(raw, ch_name=eog_name)
            for component in bads:
                if component not in ica.exclude:
                    ica.exclude.append(component)

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

        ica_path = config.QC_DIR / f"sub-{subject_id}_{person}_ica.fif"
        ica.save(ica_path, overwrite=True)
        LOGGER.info(f"Saved ICA decomposition ({person}) to: {ica_path}")

        outputs.append((person, out_path, ica_path))

    return outputs


if __name__ == "__main__":
    for i, subj in enumerate(config.SUBJECTS):
        process_subject(subj)
        LOGGER.info(f"PROGRESS:{i + 1}")
