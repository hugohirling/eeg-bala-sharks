import os
from pathlib import Path


def discover_subjects_from_bids(bids_root: Path) -> list[str]:
    subjects = []
    for path in bids_root.glob("sub-*"):
        if not path.is_dir():
            continue
        subject_id = path.name.replace("sub-", "", 1)
        if subject_id:
            subjects.append(subject_id)
    return sorted(set(subjects))


def subjects_from_env(env_var: str = "EEG_SUBJECTS") -> list[str]:
    raw = os.getenv(env_var, "").strip()
    if not raw:
        return []
    return [entry.strip() for entry in raw.split(",") if entry.strip()]


def resolve_subjects(bids_root: Path, env_var: str = "EEG_SUBJECTS") -> list[str]:
    override = subjects_from_env(env_var=env_var)
    if override:
        return override
    return discover_subjects_from_bids(bids_root)
