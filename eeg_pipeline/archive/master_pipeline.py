import subprocess
import os
import sys

pipeline_dir = "eeg_pipeline"

# Define pipeline steps in order
pipeline_steps = [
    # Preprocessing (Steps 00-06)
    "00_load_data_per_person.py",
    "01_inspect_and_rename.py",
    "02_re_reference_common_average.py",
    "03_filter.py",
    "04_ica_artifact_removal.py",
    "05_downsample.py",
    "06_epoching.py",
    # Analysis (Steps 07-10)
    "07_predictability_measure.py",      # Behavioral/Predictability analysis
    "08_interbrain_plv_shuffled.py",     # Inter-brain PLV (existing)
    "09_time_frequency_analysis.py",     # Time-frequency analysis
    "10_statistical_testing.py",         # Statistical testing
]

print("="*70)
print("           EEG HYPERSCANNING ANALYSIS PIPELINE")
print("="*70)

for step in pipeline_steps:
    step_path = os.path.join(pipeline_dir, step)
    
    # Check if file exists
    if not os.path.exists(step_path):
        print(f"\n⚠ Warning: {step} not found, skipping...")
        continue
    
    print(f"\n{'='*70}")
    print(f"Running {step_path} ...")
    print("="*70)
    
    result = subprocess.run([sys.executable, step_path])
    
    if result.returncode != 0:
        print(f"\n✗ Error in {step} (return code: {result.returncode})")
        user_input = input("Continue with next step? (y/n): ")
        if user_input.lower() != 'y':
            print("Pipeline stopped.")
            sys.exit(1)
    else:
        print(f"\n✓ {step} completed successfully.")

print("\n" + "="*70)
print("           PIPELINE COMPLETE")
print("="*70)
print("\nOutput directories:")
print("  - 06_epochs/          : Epoched EEG data")
print("  - 07_predictability/  : Behavioral predictability measures")
print("  - 08_plv/             : Inter-brain PLV results")
print("  - 09_time_frequency/  : ERPs, TFRs, and PSD")
print("  - 10_statistics/      : Statistical test results")