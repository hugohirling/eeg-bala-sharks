"""
Shared utilities for sanity check scripts.

Provides:
- Consistent error tracking and reporting
- Before/after numerical comparisons
- Anomaly detection helpers
- CSV summary output with grading-oriented rationale fields

REASONING:
- Purpose: centralize how sanity checks report not only pass/fail status, but also why a result matters.
- Reproducibility: summary CSVs make it easier to compare runs across machines and software environments.
- Key assertions: every result can store category, parameter motivation, and short interpretation text.
"""
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import csv

import numpy as np
import mne


class SanityCheckCollector:
    """Collects pass/fail/warning results across multiple subjects and persons."""

    def __init__(self, step_name: str):
        self.step_name = step_name
        self.results: Dict[str, Dict[str, List[Dict[str, str]]]] = {}
        self.errors = []
        self.warnings = []
        self.step_context = {
            "purpose": "",
            "reproducibility": "",
            "parameter_notes": [],
        }

    def set_step_context(
        self,
        *,
        purpose: str = "",
        reproducibility: str = "",
        parameter_notes: Optional[List[str]] = None,
    ):
        """Attach grading-oriented context that is printed before the per-subject summary."""
        self.step_context = {
            "purpose": purpose.strip(),
            "reproducibility": reproducibility.strip(),
            "parameter_notes": [note.strip() for note in (parameter_notes or []) if str(note).strip()],
        }

    def add_result(
        self,
        subject_id: str,
        person: str,
        status: str,
        message: str,
        *,
        category: str = "general",
        rationale: str = "",
        parameter_note: str = "",
    ):
        """Add a result entry. status in ('âœ“', 'âš ', 'ERROR')."""
        if subject_id not in self.results:
            self.results[subject_id] = {}
        if person not in self.results[subject_id]:
            self.results[subject_id][person] = []

        if status == "ERROR":
            self.errors.append(f"{subject_id}/{person}: {message}")
        elif status == "âš ":
            self.warnings.append(f"{subject_id}/{person}: {message}")

        self.results[subject_id][person].append(
            {
                "status": status,
                "message": message,
                "category": category,
                "rationale": rationale.strip(),
                "parameter_note": parameter_note.strip(),
            }
        )
    
    @staticmethod
    def _console_safe(text: str) -> str:
        return (
            str(text)
            .replace("âœ“", "[OK]")
            .replace("âš ", "[!]")
            .replace("âœ—", "[X]")
        )

    def print_summary(self):
        """Print collected results in unified format."""
        print("\n" + "=" * 80)
        print(f"SANITY CHECK: Step {self.step_name}")
        print("=" * 80)

        if self.step_context["purpose"]:
            print(f"Purpose: {self._console_safe(self.step_context['purpose'])}")
        if self.step_context["reproducibility"]:
            print(f"Reproducibility: {self._console_safe(self.step_context['reproducibility'])}")
        if self.step_context["parameter_notes"]:
            print("Parameter notes:")
            for note in self.step_context["parameter_notes"]:
                print(f"  - {self._console_safe(note)}")

        for subject_id in sorted(self.results.keys()):
            print(f"\n--- Subject {subject_id} ---")
            for person in ["P1", "P2"]:
                if person in self.results[subject_id]:
                    print(f"\n{person}:")
                    for entry in self.results[subject_id][person]:
                        prefix = f"  {entry['status']} [{entry['category']}] {entry['message']}"
                        print(self._console_safe(prefix))
                        if entry["rationale"]:
                            print(self._console_safe(f"    {entry['rationale']}"))
                        if entry["parameter_note"]:
                            print(self._console_safe(f"    Parameter note: {entry['parameter_note']}"))

        print("\n" + "=" * 80)
        if self.errors:
            print(f"ERRORS ({len(self.errors)}):")
            for error in self.errors[:10]:
                print(f"  [X] {error}")
            if len(self.errors) > 10:
                print(f"  ... and {len(self.errors) - 10} more")
        if self.warnings:
            print(f"WARNINGS ({len(self.warnings)}):")
            for warning in self.warnings[:10]:
                print(f"  [!] {warning}")
            if len(self.warnings) > 10:
                print(f"  ... and {len(self.warnings) - 10} more")

        if not self.errors and not self.warnings:
            print("[OK] No errors or warnings detected.")

        print("=" * 80)

    def export_csv(self, output_path: Path):
        """Export summary to CSV."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "Step",
                    "Subject",
                    "Person",
                    "Status",
                    "Category",
                    "Message",
                    "Rationale",
                    "ParameterNote",
                ]
            )
            for subject_id in sorted(self.results.keys()):
                for person in sorted(self.results[subject_id].keys()):
                    for entry in self.results[subject_id][person]:
                        writer.writerow(
                            [
                                self.step_name,
                                subject_id,
                                person,
                                entry["status"],
                                entry["category"],
                                entry["message"],
                                entry["rationale"],
                                entry["parameter_note"],
                            ]
                        )


def seems_correct_because(reason: str) -> str:
    """Return a short interpretation sentence for expected outcomes."""
    reason = str(reason).strip().rstrip(".")
    return f"This seems correct because {reason}." if reason else ""


def strange_because(reason: str) -> str:
    """Return a short interpretation sentence for unexpected outcomes."""
    reason = str(reason).strip().rstrip(".")
    return f"This is strange because {reason}." if reason else ""


def compare_amplitudes(
    raw_before: mne.io.BaseRaw,
    raw_after: mne.io.BaseRaw,
    duration_s: float = 120.0,
    pick_type: str = "eeg",
) -> Tuple[float, float, float]:
    """
    Compare RMS amplitude before/after processing.

    Returns:
        (std_before_uv, std_after_uv, change_pct)
    """
    eeg_picks = mne.pick_types(raw_before.info, **{pick_type: True})
    if len(eeg_picks) == 0:
        return np.nan, np.nan, np.nan

    t_end = min(duration_s, raw_before.times[-1])
    t_idx_end = int(t_end * raw_before.info["sfreq"])

    data_before = raw_before.get_data(picks=eeg_picks, start=0, stop=t_idx_end)
    data_after = raw_after.get_data(picks=eeg_picks, start=0, stop=t_idx_end)

    std_before = np.std(data_before) * 1e6  # Convert to ÂµV
    std_after = np.std(data_after) * 1e6

    change_pct = ((std_after - std_before) / std_before * 100) if std_before != 0 else 0

    return std_before, std_after, change_pct


def check_data_integrity(raw: mne.io.BaseRaw, person: str, collector: SanityCheckCollector):
    """Check for NaN/Inf values, channel count, duration."""
    pass  # Will use in specific checks


def detect_bad_channel_anomalies(bads_list: List[str], eeg_count: int) -> Optional[str]:
    """Check if bad-channel detection seems reasonable."""
    if not bads_list:
        return None

    bad_pct = len(bads_list) / eeg_count * 100
    if bad_pct > 20:
        return f"High bad-channel percentage: {bad_pct:.1f}% ({len(bads_list)}/{eeg_count})"
    return None


def detect_amplitude_anomaly(change_pct: float, threshold_pct: float = 50.0) -> Optional[str]:
    """Check if amplitude change is unusually large."""
    if abs(change_pct) > threshold_pct:
        return f"Large amplitude change: {change_pct:+.1f}%"
    return None

