from pathlib import Path
import sys
import csv

import mne

CURRENT_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = CURRENT_DIR.parent
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from preprocessing import config
from helper.general.helper_functions import get_step_io_files, save_current_step_file


def _read_tsv_rows(file_path):
    with file_path.open("r", encoding="utf-8", newline="") as file_obj:
        reader = csv.DictReader(file_obj, delimiter="\t")
        rows = []
        for row in reader:
            rows.append({str(key).strip().lower(): value for key, value in row.items()})
        return rows


def _parse_bad_channel_list(value):
    if value is None:
        return []
    if isinstance(value, str) and value.strip().lower() in {"nan", "na", "none"}:
        return []
    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned:
            return []
        return [entry.strip() for entry in cleaned.replace(";", ",").split(",") if entry.strip()]
    return []


def _get_bad_channels_for_subject(subject_id, person):
    file_path = config.BAD_CHANNELS_FILE
    if file_path is None:
        return []

    table_path = Path(file_path)
    if not table_path.exists():
        print(f"Bad-channels file not found: {table_path}. Skipping interpolation lists.")
        return []

    participants = _read_tsv_rows(table_path)
    if not participants:
        return []

    subject_label = f"sub-{subject_id}"
    if "participant_id" not in participants[0]:
        print("Column 'participant_id' not found in bad-channels table. Skipping interpolation lists.")
        return []

    row = next((entry for entry in participants if entry.get("participant_id") == subject_label), None)
    if row is None:
        return []

    person_columns = [
        f"bad_channels_{person.lower()}",
        f"bad_channels_player{person[-1]}",
        f"bad_{person.lower()}",
        "bad_channels",
    ]

    for column in person_columns:
        if column in row:
            return _parse_bad_channel_list(row.get(column))

    return []


def interpolate_bad_channels(subject_id):
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
            raise ValueError("Interpolation step requires renamed+montaged input files")

        print(f"Loading previous step file ({person}): {path_in}")
        raw = mne.io.read_raw_fif(path_in, preload=True)

        bad_channels = _get_bad_channels_for_subject(subject_id, person)
        if not bad_channels:
            bad_channels = list(raw.info.get("bads", []))
        bad_channels = [ch for ch in bad_channels if ch in raw.ch_names]

        if bad_channels:
            raw.info["bads"] = bad_channels
            raw.interpolate_bads(reset_bads=True)
            print(f"Interpolated bad channels for {subject_id} {person}: {', '.join(bad_channels)}")
        else:
            print(f"No bad channels configured for {subject_id} {person}.")

        out_path = save_current_step_file(
            raw,
            subject_id,
            __file__,
            person=person,
            step_output_suffixes=config.STEP_OUTPUT_SUFFIXES,
        )
        print(f"Saved interpolated file ({person}) to: {out_path}")
        outputs.append((person, out_path))

    return outputs


if __name__ == "__main__":
    for subj in config.SUBJECTS:
        interpolate_bad_channels(subj)
