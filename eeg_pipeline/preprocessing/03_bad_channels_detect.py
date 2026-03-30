from pathlib import Path
import sys
import csv
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed

import mne
import numpy as np

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


_BAD_CHANNEL_ROWS_CACHE = None


def _compute_robust_z(values):
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    if mad == 0:
        return np.zeros_like(values)
    return 0.6745 * (values - median) / mad


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
    global _BAD_CHANNEL_ROWS_CACHE

    file_path = config.BAD_CHANNELS_FILE
    if file_path is None:
        return []

    table_path = Path(file_path)
    if not table_path.exists():
        print(f"Bad-channels file not found: {table_path}. Skipping manual bad-channel merge.")
        return []

    if _BAD_CHANNEL_ROWS_CACHE is None:
        _BAD_CHANNEL_ROWS_CACHE = _read_tsv_rows(table_path)

    participants = _BAD_CHANNEL_ROWS_CACHE
    if not participants:
        return []

    subject_label = f"sub-{subject_id}"
    if "participant_id" not in participants[0]:
        print("Column 'participant_id' not found in bad-channels table. Skipping manual bad-channel merge.")
        return []

    row = next((entry for entry in participants if entry.get("participant_id") == subject_label), None)
    if row is None:
        return []

    person_columns = [
        f"player{person[-1]}_pre_processing_channels_fixed",
        f"bad_channels_{person.lower()}",
        f"bad_channels_player{person[-1]}",
        f"bad_{person.lower()}",
        "bad_channels",
    ]

    for column in person_columns:
        if column in row:
            return _parse_bad_channel_list(row.get(column))

    return []


def _write_qc_report(subject_id, person, channel_names, std_values, z_scores, reasons):
    report_path = config.BAD_CHANNELS_DIR / f"sub-{subject_id}_{person}_bad_channels_detect.tsv"
    with report_path.open("w", encoding="utf-8", newline="") as file_obj:
        file_obj.write("subject_id\tperson\tchannel\tstd\trobust_z\tsuggested\treason\n")
        for channel, std_value, z_value, reason in zip(channel_names, std_values, z_scores, reasons):
            suggested = "yes" if reason else "no"
            reason_text = reason if reason else ""
            file_obj.write(
                f"sub-{subject_id}\t{person}\t{channel}\t{std_value:.12e}\t{z_value:.6f}\t{suggested}\t{reason_text}\n"
            )
    return report_path


def detect_bad_channels(subject_id):
    outputs = []
    for person in ["P1", "P2"]:
        path_in, out_path = get_step_io_files(
            subject_id,
            __file__,
            person=person,
            pipeline_steps=config.PIPELINE_STEPS,
            step_output_suffixes=config.STEP_OUTPUT_SUFFIXES,
        )
        if path_in is None:
            raise ValueError("Bad-channel detection step requires rename+montage input files")

        LOGGER.info(f"Loading previous step file ({person}): {path_in}")
        raw = mne.io.read_raw_fif(path_in, preload=False)

        eeg_picks = mne.pick_types(raw.info, eeg=True, exclude=[])
        if len(eeg_picks) == 0:
            raise RuntimeError(f"No EEG channels available for bad-channel detection in {path_in}")

        channel_names = [raw.ch_names[idx] for idx in eeg_picks]
        channel_data = raw.get_data(picks=eeg_picks, reject_by_annotation="omit")
        std_values = np.std(channel_data, axis=1)
        z_scores = _compute_robust_z(std_values)

        manual_bads = set(_get_bad_channels_for_subject(subject_id, person))
        manual_bads = {channel for channel in manual_bads if channel in raw.ch_names}

        reasons = []
        suggested_bads = []
        auto_suggested_bads = []
        for channel, std_value, z_value in zip(channel_names, std_values, z_scores):
            channel_reasons = []
            if std_value <= config.BAD_CHANNEL_FLAT_STD_THRESHOLD:
                channel_reasons.append("flat")
            if abs(z_value) >= config.BAD_CHANNEL_ZSCORE_THRESHOLD:
                channel_reasons.append("outlier_std")
            if channel in manual_bads:
                channel_reasons.append("manual_tsv")

            reason_text = ",".join(channel_reasons)
            reasons.append(reason_text)
            if reason_text:
                suggested_bads.append(channel)
            if "flat" in channel_reasons or "outlier_std" in channel_reasons:
                auto_suggested_bads.append(channel)

        existing_bads = set(raw.info.get("bads", []))
        merged_bads = sorted(existing_bads.union(suggested_bads))
        raw.info["bads"] = merged_bads

        report_path = _write_qc_report(subject_id, person, channel_names, std_values, z_scores, reasons)
        LOGGER.info(f"Saved bad-channel QC report ({person}) to: {report_path}")
        if manual_bads:
            LOGGER.info(f"Manual bad channels from TSV for {subject_id} {person}: {', '.join(sorted(manual_bads))}")
        if auto_suggested_bads:
            LOGGER.info(f"Auto-suggested bad channels for {subject_id} {person}: {', '.join(auto_suggested_bads)}")
        if suggested_bads:
            LOGGER.info(f"Final bad channels marked for {subject_id} {person}: {', '.join(suggested_bads)}")
        else:

            LOGGER.info(f"No bad-channel suggestions for {subject_id} {person}.")

        if set(merged_bads) == existing_bads:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path_in, out_path)
            LOGGER.info(
                f"No bad-channel metadata change ({person}); copied input file to output without re-saving: {out_path}"
            )
        else:
            out_path = save_current_step_file(
                raw,
                subject_id,
                __file__,
                person=person,
                step_output_suffixes=config.STEP_OUTPUT_SUFFIXES,
            )
        LOGGER.info(f"Saved detection output ({person}) to: {out_path}")
        outputs.append((person, out_path, report_path, suggested_bads))

    return outputs


if __name__ == "__main__":
    subjects = list(config.SUBJECTS)
    n_jobs = max(1, int(getattr(config, "BAD_CHANNEL_N_JOBS", 1)))

    if n_jobs == 1 or len(subjects) <= 1:
        for i, subj in enumerate(subjects):
            detect_bad_channels(subj)
            LOGGER.info(f"PROGRESS:{i + 1}")
    else:
        completed = 0
        LOGGER.info(f"Running bad-channel detection with {n_jobs} workers")
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            future_to_subject = {executor.submit(detect_bad_channels, subj): subj for subj in subjects}
            for future in as_completed(future_to_subject):
                subject = future_to_subject[future]
                try:
                    future.result()
                except Exception:
                    LOGGER.exception(f"Failed bad-channel detection for subject {subject}")
                    raise

                completed += 1
                LOGGER.info(f"PROGRESS:{completed}")
