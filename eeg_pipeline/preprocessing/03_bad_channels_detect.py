from pathlib import Path
import sys

import mne
import numpy as np

CURRENT_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = CURRENT_DIR.parent
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

import config
from helper.helper_functions import get_step_io_files, save_current_step_file


def _compute_robust_z(values):
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    if mad == 0:
        return np.zeros_like(values)
    return 0.6745 * (values - median) / mad


def _write_qc_report(subject_id, person, channel_names, std_values, z_scores, reasons):
    report_path = config.QC_DIR / f"sub-{subject_id}_{person}_bad_channels_detect.tsv"
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
        path_in, _ = get_step_io_files(subject_id, __file__, person=person)
        if path_in is None:
            raise ValueError("Bad-channel detection step requires rename+montage input files")

        print(f"Loading previous step file ({person}): {path_in}")
        raw = mne.io.read_raw_fif(path_in, preload=True)

        eeg_picks = mne.pick_types(raw.info, eeg=True, exclude=[])
        if len(eeg_picks) == 0:
            raise RuntimeError(f"No EEG channels available for bad-channel detection in {path_in}")

        channel_names = [raw.ch_names[idx] for idx in eeg_picks]
        channel_data = raw.get_data(picks=eeg_picks, reject_by_annotation="omit")
        std_values = np.std(channel_data, axis=1)
        z_scores = _compute_robust_z(std_values)

        reasons = []
        suggested_bads = []
        for channel, std_value, z_value in zip(channel_names, std_values, z_scores):
            channel_reasons = []
            if std_value <= config.BAD_CHANNEL_FLAT_STD_THRESHOLD:
                channel_reasons.append("flat")
            if abs(z_value) >= config.BAD_CHANNEL_ZSCORE_THRESHOLD:
                channel_reasons.append("outlier_std")

            reason_text = ",".join(channel_reasons)
            reasons.append(reason_text)
            if reason_text:
                suggested_bads.append(channel)

        existing_bads = set(raw.info.get("bads", []))
        merged_bads = sorted(existing_bads.union(suggested_bads))
        raw.info["bads"] = merged_bads

        report_path = _write_qc_report(subject_id, person, channel_names, std_values, z_scores, reasons)
        print(f"Saved bad-channel QC report ({person}) to: {report_path}")
        if suggested_bads:
            print(f"Suggested bad channels for {subject_id} {person}: {', '.join(suggested_bads)}")
        else:
            print(f"No bad-channel suggestions for {subject_id} {person}.")

        out_path = save_current_step_file(raw, subject_id, __file__, person=person)
        print(f"Saved detection output ({person}) to: {out_path}")
        outputs.append((person, out_path, report_path, suggested_bads))

    return outputs


if __name__ == "__main__":
    for subj in config.SUBJECTS:
        detect_bad_channels(subj)
