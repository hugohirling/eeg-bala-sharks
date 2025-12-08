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
Step 2 — Filtering
Step 3 — Detect noisy channels
Step 4 — Rereference
Step 5 — ICA → ICLabel → Remove bad ICs
Step 6 — Interpolate removed channels
```