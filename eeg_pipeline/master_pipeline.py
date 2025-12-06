import subprocess
import os

pipeline_dir = "eeg_pipeline"
pipeline_steps = [
    "01_load_data.py",
    "02_re-reference.py",
    "03_artifact_detection.py",
    "04_epoching.py",
    "05_ica.py",
    "06_interbrain_analysis.py"
]

for step in pipeline_steps:
    step_path = os.path.join(pipeline_dir, step)
    print(f"Running {step_path} ...")
    subprocess.run(["python", step_path])
    print(f"{step} completed.\n")
