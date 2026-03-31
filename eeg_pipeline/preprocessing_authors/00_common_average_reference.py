from pathlib import Path
import sys
import logging

CURRENT_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = CURRENT_DIR.parent
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from preprocessing import config
from helper.general.helper_functions import get_step_io_files, save_current_step_file
from helper.general.helper_functions import load_raw_fif

from helper.authors.authors_helpers import PIPELINE_STEPS, STEP_OUTPUT_SUFFIXES, resolve_initial_input, resolve_output_dir


def apply_common_average_reference(raw):
    raw.set_eeg_reference(ref_channels="average")
    return raw


def process_subject(subject_id):
    output_dir = resolve_output_dir()

    path_in, _ = get_step_io_files(
        subject_id=subject_id,
        current_step=__file__,
        output_dir=output_dir,
        pipeline_steps=PIPELINE_STEPS,
        step_output_suffixes=STEP_OUTPUT_SUFFIXES,
    )

    if path_in is None:
        path_in = resolve_initial_input(subject_id, logger=logging.getLogger("authors_step00"))

    if not Path(path_in).exists():
        raise FileNotFoundError(f"Input not found for sub-{subject_id}: {path_in}")

    print(f"[authors] Loading input: {path_in}")
    raw = load_raw_fif(path_in, preload=True)

    raw = apply_common_average_reference(raw)

    out_path = save_current_step_file(
        raw,
        subject_id,
        __file__,
        output_dir=output_dir,
        step_output_suffixes=STEP_OUTPUT_SUFFIXES,
    )
    print(f"[authors] Saved CAR output: {out_path}")
    return out_path


if __name__ == "__main__":
    for subj in config.SUBJECTS:
        process_subject(subj)
