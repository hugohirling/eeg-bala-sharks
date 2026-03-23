from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any, Callable, Sequence

import config

DEFAULT_PIPELINE_STEPS = [
	"downsample",
	"split_players",
	"rename_set_montage",
	"bad_channels_detect",
	"interpolate_bad_channels",
	"filter",
	"ica",
	"epoch",
]

DEFAULT_STEP_OUTPUT_SUFFIXES = {
	"downsample": "downsampled",
	"split_players": "split",
	"rename_set_montage": "renamed_montaged",
	"bad_channels_detect": "badchannels_detected",
	"interpolate_bad_channels": "interpolated",
	"filter": "filtered",
	"ica": "ica_cleaned",
	"epoch": "epoch",
}


def normalize_step_name(step: str | Path) -> str:
	step_stem = Path(str(step)).stem
	return re.sub(r"^\d+[_-]*", "", step_stem)


def load_pipeline_steps_from_master(master_pipeline_path: str | Path) -> list[str]:
	master_path = Path(master_pipeline_path)
	module = ast.parse(master_path.read_text(encoding="utf-8"), filename=str(master_path))

	for node in module.body:
		if not isinstance(node, ast.Assign):
			continue
		for target in node.targets:
			if not isinstance(target, ast.Name) or target.id != "pipeline_steps":
				continue
			if not isinstance(node.value, (ast.List, ast.Tuple)):
				break
			return [
				element.value
				for element in node.value.elts
				if isinstance(element, ast.Constant) and isinstance(element.value, str)
			]

	raise ValueError(f"No 'pipeline_steps' list found in {master_path}")


def _resolve_pipeline_steps(
	pipeline_steps: Sequence[str | Path] | None = None,
	master_pipeline_path: str | Path | None = None,
) -> list[str]:
	if pipeline_steps is not None:
		raw_steps = list(pipeline_steps)
	elif master_pipeline_path is not None:
		raw_steps = load_pipeline_steps_from_master(master_pipeline_path)
	else:
		raw_steps = DEFAULT_PIPELINE_STEPS
	return [normalize_step_name(step) for step in raw_steps]


def _resolve_step_suffixes(
	step_output_suffixes: dict[str, str] | None = None,
) -> dict[str, str]:
	if step_output_suffixes is None:
		return dict(DEFAULT_STEP_OUTPUT_SUFFIXES)
	return {normalize_step_name(step): suffix for step, suffix in step_output_suffixes.items()}


def _build_subject_file_path(
	subject_id: str,
	suffix: str,
	person: str | None = None,
	extension: str = ".fif",
	output_dir: str | Path | None = None,
) -> Path:
	base_dir = Path(output_dir) if output_dir is not None else Path(config.OUTPUT_DIR)
	person_part = f"_{person}" if person else ""
	return base_dir / f"sub-{subject_id}{person_part}_{suffix}{extension}"


def get_previous_step_name(
	current_step: str | Path,
	pipeline_steps: Sequence[str | Path] | None = None,
	master_pipeline_path: str | Path | None = None,
) -> str | None:
	normalized_current_step = normalize_step_name(current_step)
	normalized_steps = _resolve_pipeline_steps(pipeline_steps, master_pipeline_path)

	if normalized_current_step not in normalized_steps:
		available_steps = ", ".join(normalized_steps)
		raise ValueError(
			f"Step '{normalized_current_step}' is not in the pipeline order. Available steps: {available_steps}"
		)

	step_index = normalized_steps.index(normalized_current_step)
	if step_index == 0:
		return None
	return normalized_steps[step_index - 1]


def get_previous_step_file(
	subject_id: str,
	current_step: str | Path,
	person: str | None = None,
	extension: str = ".fif",
	output_dir: str | Path | None = None,
	pipeline_steps: Sequence[str | Path] | None = None,
	step_output_suffixes: dict[str, str] | None = None,
	master_pipeline_path: str | Path | None = None,
) -> Path | None:
	previous_step = get_previous_step_name(current_step, pipeline_steps, master_pipeline_path)
	if previous_step is None:
		return None

	suffixes = _resolve_step_suffixes(step_output_suffixes)
	if previous_step not in suffixes:
		raise ValueError(f"No output suffix configured for previous step '{previous_step}'")

	return _build_subject_file_path(subject_id, suffixes[previous_step], person, extension, output_dir)


def get_current_step_output_file(
	subject_id: str,
	current_step: str | Path,
	person: str | None = None,
	extension: str = ".fif",
	output_dir: str | Path | None = None,
	step_output_suffixes: dict[str, str] | None = None,
) -> Path:
	current_step_name = normalize_step_name(current_step)
	suffixes = _resolve_step_suffixes(step_output_suffixes)

	if current_step_name not in suffixes:
		available_steps = ", ".join(sorted(suffixes))
		raise ValueError(
			f"No output suffix configured for step '{current_step_name}'. Available steps: {available_steps}"
		)

	return _build_subject_file_path(subject_id, suffixes[current_step_name], person, extension, output_dir)


def get_step_io_files(
	subject_id: str,
	current_step: str | Path,
	person: str | None = None,
	extension: str = ".fif",
	output_dir: str | Path | None = None,
	pipeline_steps: Sequence[str | Path] | None = None,
	step_output_suffixes: dict[str, str] | None = None,
	master_pipeline_path: str | Path | None = None,
) -> tuple[Path | None, Path]:
	input_file = get_previous_step_file(
		subject_id=subject_id,
		current_step=current_step,
		person=person,
		extension=extension,
		output_dir=output_dir,
		pipeline_steps=pipeline_steps,
		step_output_suffixes=step_output_suffixes,
		master_pipeline_path=master_pipeline_path,
	)
	output_file = get_current_step_output_file(
		subject_id=subject_id,
		current_step=current_step,
		person=person,
		extension=extension,
		output_dir=output_dir,
		step_output_suffixes=step_output_suffixes,
	)
	return input_file, output_file


def save_current_step_file(
	data: Any,
	subject_id: str,
	current_step: str | Path,
	person: str | None = None,
	save_callable: Callable[[Any, Path], None] | None = None,
	extension: str = ".fif",
	output_dir: str | Path | None = None,
	step_output_suffixes: dict[str, str] | None = None,
	overwrite: bool = True,
) -> Path:
	output_file = get_current_step_output_file(
		subject_id=subject_id,
		current_step=current_step,
		person=person,
		extension=extension,
		output_dir=output_dir,
		step_output_suffixes=step_output_suffixes,
	)
	output_file.parent.mkdir(parents=True, exist_ok=True)

	if save_callable is not None:
		save_callable(data, output_file)
		return output_file

	if not hasattr(data, "save"):
		raise TypeError("data must provide a .save(...) method or save_callable must be set")

	data.save(output_file, overwrite=overwrite)
	return output_file
