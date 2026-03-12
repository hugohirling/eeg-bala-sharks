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
- position mapping of the renaming might be different? -> authors have some coordinates we have to check (Hugo has done some work on this) VERY IMPORTANT for the topoplots!!!!
- rereferencing/filter: when is it good -> power spectrum plot doesn't look too different -> data doesn't get too distorted (if needed look at papers for good rereferencing/filtering)
- ica: rejection criteria -> colorful plots a good sign (look into lecture how to interpret ica plots)
- bad channels to clean data (some measure) further before ica even!!!
- ica before encoding (ica is useful to remove blinks and eye movements) - for our case it matters since we use all channels to decode to clean eye movement and bad channels
- original authors code: switched around the prefixes! (Hugo has done this in his script)
- scope of project: add some other analysis since we drop feedback and reaction phase -> to add to report
- try a different type of classifier was a feedback of the last tutor meeting 

- highly suggest adding more analysis for the scope
- general comment: try to document a lot for the code and report (write about important things, put the sanity checks in and so on, what we find interesting, swapping labels and so on -> a year from now you need to talk about this as a basis for a report)