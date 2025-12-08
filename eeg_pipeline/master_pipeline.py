import subprocess
import os

pipeline_dir = "eeg_pipeline"
pipeline_steps = [
    "00_load_data_per_person.py",
    "01_re-reference_common_average.py",
    "02_filter.py",
    "03_detect_noisy_channels.py",
    "04_downsample.py",
    "05_epoching.py"
    ]

for step in pipeline_steps:
    step_path = os.path.join(pipeline_dir, step)
    print(f"Running {step_path} ...")
    subprocess.run(["python", step_path])
    print(f"{step} completed.\n")
