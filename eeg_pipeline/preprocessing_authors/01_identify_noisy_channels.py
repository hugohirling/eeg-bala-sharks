from pathlib import Path
import sys

import numpy as np

CURRENT_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = CURRENT_DIR.parent
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from preprocessing import config
from helper.general.helper_functions import get_step_io_files, save_current_step_file
from helper.general.helper_functions import load_raw_fif, save_json

from helper.authors.authors_helpers import (
    PIPELINE_STEPS,
    STEP_OUTPUT_SUFFIXES,
    resolve_interactive_noisy,
    resolve_output_dir,
)


def detect_noisy_channels_automated(raw, z_threshold=3.0):
    data = raw.get_data()
    variances = np.var(data, axis=1)
    mean_var = np.mean(variances)
    std_var = np.std(variances)
    z_scores = np.abs((variances - mean_var) / (std_var if std_var > 0 else 1.0))
    noisy_candidates = np.where(z_scores > z_threshold)[0]
    noisy_channel_names = [raw.ch_names[i] for i in noisy_candidates]
    return noisy_channel_names, variances


def process_subject(subject_id, interactive=False):
    output_dir = resolve_output_dir()
    qc_dir = output_dir / "qc"
    qc_dir.mkdir(parents=True, exist_ok=True)

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

    noisy_channels, variances = detect_noisy_channels_automated(raw)

    try:
        import matplotlib.pyplot as plt

        fig_path = qc_dir / f"sub-{subject_id}_channel_variances_authors.png"
        plt.figure(figsize=(14, 6))
        plt.bar(range(len(variances)), variances)
        plt.xlabel("Channel Index")
        plt.ylabel("Variance")
        plt.title(f"sub-{subject_id}: Channel variances (authors)")
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(fig_path, dpi=150)
        plt.close()
        print(f"[authors] Saved QC plot: {fig_path}")
    except Exception as exc:
        print(f"[authors] Plot skipped: {exc}")

    if interactive:
        raw.plot(duration=30, n_channels=min(32, len(raw.ch_names)), scalings="auto")

    log_file = qc_dir / f"sub-{subject_id}_noisy_channels_authors.json"
    save_json(
        {
            "subject": f"sub-{subject_id}",
            "method": "variance-based + visual inspection",
            "n_noisy_channels": len(noisy_channels),
            "noisy_channels": noisy_channels,
        },
        log_file,
    )
    print(f"[authors] Saved noisy-channel log: {log_file}")

    out_path = save_current_step_file(
        raw,
        subject_id,
        __file__,
        output_dir=output_dir,
        step_output_suffixes=STEP_OUTPUT_SUFFIXES,
    )
    print(f"[authors] Saved noisy-check output: {out_path}")
    return out_path


if __name__ == "__main__":
    interactive = resolve_interactive_noisy()
    for subj in config.SUBJECTS:
        process_subject(subj, interactive=interactive)
