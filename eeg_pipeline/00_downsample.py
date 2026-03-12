def load_data(subject_id):
    print(f"Loading subject {subject_id}")

    bids_path = BIDSPath(
        subject=subject_id,
        task="RPS",
        datatype="eeg",
        suffix="eeg",
        root=config.BIDS_ROOT,
    )

    raw = read_raw_bids(bids_path, verbose=False)
    return raw


def downsample_data(raw, target_sfreq=config.DOWNSAMPLE_SFREQ):
    print(f"Down-sampling data to {target_sfreq} Hz")

    raw.resample(target_sfreq)
    return raw

if __name__ == "__main__":
    for subj in config.SUBJECTS:
        raw = load_data(subj)
        raw = downsample_data(raw)