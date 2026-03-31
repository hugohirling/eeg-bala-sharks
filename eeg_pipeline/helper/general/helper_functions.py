from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any, Callable, Sequence

from preprocessing import config
import mne


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
			if not isinstance(target, ast.Name) or target.id not in {"pipeline_steps", "PIPELINE_STEPS"}:
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
		raise ValueError(
			"pipeline_steps must be provided explicitly (or set master_pipeline_path)"
		)
	return [normalize_step_name(step) for step in raw_steps]


def _resolve_step_suffixes(
	step_output_suffixes: dict[str, str] | None = None,
) -> dict[str, str]:
	if step_output_suffixes is None:
		raise ValueError("step_output_suffixes must be provided explicitly")
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


def ensure_parent_dir(file_path: str | Path) -> Path:
	path = Path(file_path)
	path.parent.mkdir(parents=True, exist_ok=True)
	return path


def _load_with_reader(
	file_path: str | Path,
	reader: Callable[[str], Any],
	*,
	default: Any = None,
	allow_missing: bool = False,
) -> Any:
	path = Path(file_path)
	if allow_missing and not path.exists():
		return default
	return reader(str(path))


def _save_with_writer(file_path: str | Path, writer: Callable[[Path], None]) -> Path:
	path = ensure_parent_dir(file_path)
	writer(path)
	return path


def load_raw_fif(input_file: str | Path, preload: bool = True) -> mne.io.BaseRaw:
	# Use MNE's generic loader so BDF/FIF and other supported raw formats work.
	return _load_with_reader(
		input_file,
		lambda p: mne.io.read_raw(p, preload=preload),
	)


def load_epochs_fif(epochs_file: str | Path, preload: bool = True) -> mne.Epochs:
	return _load_with_reader(
		epochs_file,
		lambda p: mne.read_epochs(p, preload=preload),
	)


def save_epochs_fif(epochs: mne.Epochs, output_file: str | Path, overwrite: bool = True) -> Path:
	return _save_with_writer(
		output_file,
		lambda p: epochs.save(str(p), overwrite=overwrite),
	)


def load_json(json_file: str | Path, default: Any = None) -> Any:
	return _load_with_reader(
		json_file,
		lambda p: json.loads(Path(p).read_text(encoding="utf-8")),
		default=default,
		allow_missing=True,
	)


def save_json(payload: Any, json_file: str | Path) -> Path:
	return _save_with_writer(
		json_file,
		lambda p: p.write_text(json.dumps(payload, indent=2), encoding="utf-8"),
	)
