import subprocess
import os

pipeline_dir = "eeg_pipeline"
pipeline_steps = [
    "01_load_data_per_person.py",
    "02_re-reference_common_average.py",
]

for step in pipeline_steps:
    step_path = os.path.join(pipeline_dir, step)
    print(f"Running {step_path} ...")
    subprocess.run(["python", step_path])
    print(f"{step} completed.\n")
