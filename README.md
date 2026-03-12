# Authors
- Ayush Batra
- Bahar Kalyoncu
- Hugo Hirling

# Project Structure
```
EEG_Bala Sharks
├─ eeg_pipeline/
├─ milestones/
├─ MNE-sample-data/
│  └─ ds006761/
├─ sanity_checks/
```
# Pipeline
```
Step 1 — Load & split
Step 2 - Rename channels
Step 3 — Filtering
Step 4 — ICA Labeling
Step 5 — Downsampling
Step 6 — Epoching
```


Question:
Do I need to add positions while loading? or sometime after?

Question for Tutor Session:
Should I keep the numbering for each person?


Notes 120326 - Tutor Meeting:
- downsample at the start! (due to memory issues)
- merging accuracy.py with pipeline (simplyfy)
- decode after ica removal
- downsample to 100 or 200 Hz -> should still be clear for frequencys of interest
- bad channel detection etc. before the output
- cleaned data before interpreting
- we might have uncleaned data for example if subject is blinking a lot -> output might look different than expected
- 