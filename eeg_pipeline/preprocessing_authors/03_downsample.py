from pathlib import Path
import sys

CURRENT_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = CURRENT_DIR.parent
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from preprocessing import config
from helper.general.helper_functions import get_step_io_files, save_current_step_file
from helper.general.helper_functions import load_raw_fif

from helper.authors.authors_helpers import PIPELINE_STEPS, STEP_OUTPUT_SUFFIXES, resolve_output_dir, resolve_target_sfreq


def downsample_data(raw, target_sfreq=256):
    if float(raw.info["sfreq"]) != float(target_sfreq):
        raw.resample(target_sfreq)
    return raw


def process_subject(subject_id, target_sfreq=256):
    output_dir = resolve_output_dir()

    path_in, _ = get_step_io_files(
        subject_id=subject_id,
        current_step=__file__,
        output_dir=output_dir,
        pipeline_steps=PIPELINE_STEPS,
        step_output_suffixes=STEP_OUTPUT_SUFFIXES,
    )
    if path_in is None or not Path(path_in).exists():
        raise FileNotFoundError(f"Input not found for sub-{subject_id}: {path_in}")

    print(f"[authors] Loading input: {path_in}")
    raw = load_raw_fif(path_in, preload=True)

    raw = downsample_data(raw, target_sfreq=target_sfreq)

    out_path = save_current_step_file(
        raw,
        subject_id,
        __file__,
        output_dir=output_dir,
        step_output_suffixes=STEP_OUTPUT_SUFFIXES,
    )
    print(f"[authors] Saved downsampled output: {out_path}")
    return out_path


if __name__ == "__main__":
    target_sfreq = resolve_target_sfreq()
    for subj in config.SUBJECTS:
        process_subject(subj, target_sfreq=target_sfreq)
