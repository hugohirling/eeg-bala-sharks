from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd

CURRENT_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = CURRENT_DIR.parent
if str(PIPELINE_DIR) not in sys.path:
	sys.path.insert(0, str(PIPELINE_DIR))

from preprocessing import config

from _rps_decoding_utils import PHASE_SPECS, TARGET_CHOICES, TARGET_DISPLAY_NAMES, target_output_dir


# -----------------------------------------------------------------------------
# Function: _load_phase_timecourse_csvs
# Purpose:
#   Load one CSV per task phase and stack them into a single long-format table.
# Inputs:
#   - out_dir: folder containing per-phase decoding CSV outputs.
#   - phases: phase names to load (e.g., decision/response/feedback).
# Outputs:
#   - Concatenated DataFrame with a "phase" column attached.
# Notes:
#   - Fails fast if any expected CSV is missing so plotting never uses partial data.
# -----------------------------------------------------------------------------
def _load_phase_timecourse_csvs(out_dir: Path, phases: list[str]) -> pd.DataFrame:
	"""
	Loads and concatenates per-phase decoding timecourse CSV files.

	Args:
		out_dir (Path): Target-specific decoding output directory.
		phases (list[str]): Phase keys to load.

	Returns:
		pd.DataFrame: Concatenated long-format timecourse table.

	Raises:
		FileNotFoundError: If one or more phase CSVs are missing.
	"""
	frames: list[pd.DataFrame] = []
	for phase in phases:
		# Build the exact file name convention used by decoding export scripts.
		csv_path = out_dir / f"{phase}_phase_decoding_timecourse.csv"
		if not csv_path.exists():
			raise FileNotFoundError(f"Missing timecourse CSV for phase={phase}: {csv_path}")
		df = pd.read_csv(csv_path)
		# Keep source-phase identity explicit after concatenation.
		df["phase"] = phase
		frames.append(df)

	# Merge all rows across phases into a single table for downstream plotting.
	return pd.concat(frames, axis=0, ignore_index=True)


# -----------------------------------------------------------------------------
# Function: _add_absolute_time
# Purpose:
#   Convert per-phase relative bin times into one shared absolute timeline.
# Inputs:
#   - df: concatenated phase DataFrame containing bin_*_s columns.
# Outputs:
#   - Copy of df with absolute_start_s / absolute_center_s / absolute_end_s.
# Notes:
#   - Uses PHASE_SPECS tmin offsets so each phase appears in its true task slot.
# -----------------------------------------------------------------------------
def _add_absolute_time(df: pd.DataFrame) -> pd.DataFrame:
	"""
	Converts phase-relative bin times to absolute task time in seconds.

	Args:
		df (pd.DataFrame): Combined timecourse dataframe.

	Returns:
		pd.DataFrame: Copy with absolute-time start/center/end columns.
	"""
	out = df.copy()
	# Start with the existing relative times; then shift each phase by its onset.
	out["absolute_start_s"] = out["bin_start_s"].to_numpy(dtype=float)
	out["absolute_center_s"] = out["bin_center_s"].to_numpy(dtype=float)
	out["absolute_end_s"] = out["bin_end_s"].to_numpy(dtype=float)

	for phase, spec in PHASE_SPECS.items():
		mask = out["phase"] == phase
		offset = float(spec["tmin"])
		# Phase-wise shift from local (phase-relative) time to global (task) time.
		out.loc[mask, "absolute_start_s"] = out.loc[mask, "absolute_start_s"] + offset
		out.loc[mask, "absolute_center_s"] = out.loc[mask, "absolute_center_s"] + offset
		out.loc[mask, "absolute_end_s"] = out.loc[mask, "absolute_end_s"] + offset

	return out


# -----------------------------------------------------------------------------
# Function: _save_combined_plot
# Purpose:
#   Save a combined all-phase plot with separate winner vs loser trajectories.
# Inputs:
#   - df: timecourse DataFrame with absolute-time columns.
#   - out_path: destination PNG path.
#   - target: decoding target key used for the title.
# Outputs:
#   - Path to the saved figure.
# Notes:
#   - Introduces tiny x-axis gaps between phases to emphasize segmentation.
# -----------------------------------------------------------------------------
def _save_combined_plot(df: pd.DataFrame, out_path: Path, target: str) -> Path:
	"""
	Saves a single combined plot for decision/response/feedback decoding.

	The plot shows winners vs losers over absolute task time and includes
	phase boundaries plus chance-level reference.

	Args:
		df (pd.DataFrame): Timecourse data with absolute-time columns.
		out_path (Path): Destination PNG path.
		target (str): Target key for title text.

	Returns:
		Path: Saved figure path.
	"""
	# Aggregate to one mean +/- SD curve per class and phase at each time bin.
	grouped = (
		df[df["match_status"].isin(["winner", "loser"]) ]
		.groupby(["match_status", "phase", "absolute_center_s", "absolute_start_s", "absolute_end_s"], as_index=False)
		.agg(mean_accuracy=("accuracy", "mean"), std_accuracy=("accuracy", "std"))
		.sort_values("absolute_center_s")
	)
	# SD can be NaN for singleton groups; treat as zero-width uncertainty ribbon.
	grouped["std_accuracy"] = grouped["std_accuracy"].fillna(0.0)

	# Add tiny visual gaps between phases to match segmented panel-like styling.
	phase_gap_s = 0.06
	phase_offsets = {"decision": 0.0, "response": phase_gap_s, "feedback": 2.0 * phase_gap_s}
	grouped["x_plot"] = grouped["absolute_center_s"] + grouped["phase"].map(phase_offsets).astype(float)

	fig, ax = plt.subplots(figsize=(11.0, 5.2))

	# Light phase backgrounds make segment boundaries easier to parse at a glance.
	ax.axvspan(0.0, 2.0, color="#f0f0f0", alpha=0.9, zorder=0)
	ax.axvspan(2.0 + phase_gap_s, 4.0 + phase_gap_s, color="#e8ecef", alpha=0.9, zorder=0)
	ax.axvspan(4.0 + 2.0 * phase_gap_s, 5.0 + 2.0 * phase_gap_s, color="#f0f0f0", alpha=0.9, zorder=0)

	style_map = {
		"winner": {"label": "Winners", "color": "#4f6ea8"},
		"loser": {"label": "Losers", "color": "#6bc7b4"},
	}

	label_used = {"winner": False, "loser": False}
	for match_status in ["winner", "loser"]:
		for phase in ["decision", "response", "feedback"]:
			# Plot each (class, phase) segment separately to preserve phase boundaries.
			part = grouped[(grouped["match_status"] == match_status) & (grouped["phase"] == phase)].copy()
			if part.empty:
				continue

			x = part["x_plot"].to_numpy(dtype=float)
			y = 100.0 * part["mean_accuracy"].to_numpy(dtype=float)
			yerr = 100.0 * part["std_accuracy"].to_numpy(dtype=float)
			color = style_map[match_status]["color"]
			label = style_map[match_status]["label"] if not label_used[match_status] else None

			ax.plot(x, y, marker="o", linewidth=1.8, markersize=4.0, color=color, label=label, zorder=3)
			# Shade one standard deviation around the mean decoding accuracy.
			ax.fill_between(x, y - yerr, y + yerr, color=color, alpha=0.18, zorder=2)
			label_used[match_status] = True

	# Three-class chance level reference line.
	chance_percent = 100.0 / 3.0
	ax.axhline(chance_percent, color="#555555", linestyle=(0, (3, 3)), linewidth=1.2)

	ax.text(1.0, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 40.0, "Decision", ha="center", va="top", fontsize=10)
	ax.text(3.0 + phase_gap_s, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 40.0, "Response", ha="center", va="top", fontsize=10)
	ax.text(4.5 + 2.0 * phase_gap_s, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 40.0, "Feedback", ha="center", va="top", fontsize=10)

	x_ticks_raw = np.arange(0.0, 5.1, 0.5)
	# Shift tick positions to match the same visual phase offsets used for the data.
	x_ticks_plot = np.array(
		[
			t if t < 2.0 else (t + phase_gap_s if t < 4.0 else t + 2.0 * phase_gap_s)
			for t in x_ticks_raw
		],
		dtype=float,
	)
	ax.set_xlim(0.0, 5.0 + 2.0 * phase_gap_s)
	ax.set_xticks(x_ticks_plot)
	ax.set_xticklabels([f"{t:.1f}" for t in x_ticks_raw])
	ax.set_xlabel("Time (s)")
	ax.set_ylabel("Decoding accuracy (%)")
	ax.set_title(f"Own response decoding across phases: {TARGET_DISPLAY_NAMES[target]}")
	ax.grid(axis="y", alpha=0.25)
	ax.legend(loc="lower right", frameon=False)

	fig.tight_layout()
	fig.savefig(out_path, dpi=220, bbox_inches="tight")
	plt.close(fig)
	return out_path


# -----------------------------------------------------------------------------
# Function: _save_aggregated_plot
# Purpose:
#   Save a combined all-phase plot with one aggregated curve per phase.
# Inputs:
#   - df: timecourse DataFrame with absolute-time columns.
#   - out_path: destination PNG path.
#   - target: decoding target key used for the title.
# Outputs:
#   - Path to the saved figure.
# Notes:
#   - Unlike _save_combined_plot, this ignores winner/loser labels entirely.
# -----------------------------------------------------------------------------
def _save_aggregated_plot(df: pd.DataFrame, out_path: Path, target: str) -> Path:
	"""
	Saves a single aggregated accuracy plot without winner/loser separation.

	The curve is split by phase using distinct colors while keeping one decoding
	signal (mean +- SD across all subject/player rows).

	Args:
		df (pd.DataFrame): Timecourse data with absolute-time columns.
		out_path (Path): Destination PNG path.
		target (str): Target key for title text.

	Returns:
		Path: Saved figure path.
	"""
	# Collapse across all rows to one phase-specific signal (mean +/- SD) per bin.
	grouped = (
		df.groupby(["phase", "absolute_center_s", "absolute_start_s", "absolute_end_s"], as_index=False)
		.agg(mean_accuracy=("accuracy", "mean"), std_accuracy=("accuracy", "std"))
		.sort_values("absolute_center_s")
	)
	grouped["std_accuracy"] = grouped["std_accuracy"].fillna(0.0)

	fig, ax = plt.subplots(figsize=(11.0, 5.2))

	ax.axvspan(0.0, 2.0, color="#f0f0f0", alpha=0.9, zorder=0)
	ax.axvspan(2.0, 4.0, color="#e8ecef", alpha=0.9, zorder=0)
	ax.axvspan(4.0, 5.0, color="#f0f0f0", alpha=0.9, zorder=0)
	ax.axvline(2.0, color="#9e9e9e", linewidth=1.0)
	ax.axvline(4.0, color="#9e9e9e", linewidth=1.0)

	phase_styles = {
		"decision": {"color": "#f0ad7a", "label": "Decision"},
		"response": {"color": "#d95f6a", "label": "Response"},
		"feedback": {"color": "#b05ca8", "label": "Feedback"},
	}

	for phase in ["decision", "response", "feedback"]:
		# Draw each phase independently so color mapping and legend remain explicit.
		part = grouped[grouped["phase"] == phase].copy()
		if part.empty:
			continue

		x = part["absolute_center_s"].to_numpy(dtype=float)
		y = 100.0 * part["mean_accuracy"].to_numpy(dtype=float)
		yerr = 100.0 * part["std_accuracy"].to_numpy(dtype=float)
		color = phase_styles[phase]["color"]

		ax.plot(
			x,
			y,
			marker="o",
			linewidth=1.8,
			markersize=4.0,
			color=color,
			label=phase_styles[phase]["label"],
			zorder=3,
		)
		ax.fill_between(x, y - yerr, y + yerr, color=color, alpha=0.18, zorder=2)

	# Three-class chance level reference line.
	chance_percent = 100.0 / 3.0
	ax.axhline(chance_percent, color="#555555", linestyle=(0, (3, 3)), linewidth=1.2)

	y_top = ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 40.0
	ax.text(1.0, y_top, "Decision", ha="center", va="top", fontsize=10)
	ax.text(3.0, y_top, "Response", ha="center", va="top", fontsize=10)
	ax.text(4.5, y_top, "Feedback", ha="center", va="top", fontsize=10)

	ax.set_xlim(0.0, 5.0)
	ax.set_xticks(np.arange(0.0, 5.1, 0.5))
	ax.set_xlabel("Time (s)")
	ax.set_ylabel("Decoding accuracy (%)")
	ax.set_title(f"Own response decoding across phases: {TARGET_DISPLAY_NAMES[target]}")
	ax.grid(axis="y", alpha=0.25)
	ax.legend(loc="lower right", frameon=False)

	fig.tight_layout()
	fig.savefig(out_path, dpi=220, bbox_inches="tight")
	plt.close(fig)
	return out_path


# -----------------------------------------------------------------------------
# Function: main
# Purpose:
#   CLI entry point that loads phase CSVs, builds absolute-time data, and saves
#   either split winner/loser or aggregated all-phase decoding plot.
# Inputs:
#   - Command-line arguments (--target, --out-name, --split-match-status).
# Outputs:
#   - Side effect: writes one PNG figure in the target decoding output folder.
# -----------------------------------------------------------------------------
def main() -> None:
	"""
	CLI entry point for creating one combined all-phase decoding timecourse plot.

	Returns:
		None
	"""
	# Silence verbose MNE logging so CLI output stays focused on pipeline status.
	mne.set_config("MNE_LOGGING_LEVEL", "ERROR")

	parser = argparse.ArgumentParser(
		description="Create one combined decision/response/feedback decoding plot from existing CSV outputs."
	)
	parser.add_argument(
		"--target",
		type=str,
		default="current_self",
		choices=TARGET_CHOICES,
		help="Target output directory to use (default: current_self)",
	)
	parser.add_argument(
		"--out-name",
		type=str,
		default="all_phases_decoding_timecourse.png",
		help="Output figure file name (saved in the target decoding directory)",
	)
	parser.add_argument(
		"--split-match-status",
		action=argparse.BooleanOptionalAction,
		default=True,
		help=(
			"If true (default), plot separate winner/loser curves. "
			"If false, plot one aggregated decoding curve with phase colors."
		),
	)
	args = parser.parse_args()

	# Fixed phase order matches preprocessing/decoding export conventions.
	phases = ["decision", "response", "feedback"]
	base_out_dir = Path(config.OUTPUT_DIR).parent / "decoding"
	out_dir = target_output_dir(base_out_dir, args.target)
	# Ensure destination exists even when plotter is run before prior steps create it.
	out_dir.mkdir(parents=True, exist_ok=True)

	# Load raw per-phase decoding outputs and project all bins into a shared timeline.
	combined_df = _load_phase_timecourse_csvs(out_dir, phases)
	combined_df = _add_absolute_time(combined_df)

	out_path = out_dir / args.out_name
	# Choose plotting mode based on CLI switch.
	if args.split_match_status:
		saved = _save_combined_plot(combined_df, out_path, args.target)
	else:
		saved = _save_aggregated_plot(combined_df, out_path, args.target)
	# Emit final artifact path for easy terminal confirmation.
	print(f"Saved combined plot: {saved}")


if __name__ == "__main__":
	main()
