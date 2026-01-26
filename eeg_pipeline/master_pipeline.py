import subprocess
import os

pipeline_dir = "eeg_pipeline"
pipeline_steps = [
    "00_load_data_per_person.py",
    "01_inspect_and_rename.py",
    "02_re_reference_common_average.py",
    "03_filter.py",
    "04_ica_artifact_removal.py",
    "05_downsample.py",
    "06_epoching.py"
    ]

for step in pipeline_steps:
    step_path = os.path.join(pipeline_dir, step)
    print(f"Running {step_path} ...")
    subprocess.run(["python", step_path])
    print(f"{step} completed.\n")
