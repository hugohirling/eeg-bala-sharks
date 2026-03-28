# Preprocessing Authors Pipeline - Setup Complete ✓

## Was wurde erstellt?

Ich habe eine vollständige **Preprocessing-Pipeline** basierend auf dem Paper von **Moerel et al. (2025)** erstellt.

### Verzeichnisstruktur

```
eeg_pipeline/preprocessing_authors/          ← Neue Pipeline
├── 00_common_average_reference.py           (Step 1: CAR)
├── 01_identify_noisy_channels.py            (Step 2: Kanal-Identifikation)
├── 02_interpolate_bad_channels.py           (Step 3: Interpolation)
├── 03_downsample.py                         (Step 4: Downsampling 2048→256 Hz)
├── 04_epoch.py                              (Step 5: Epoching 3 Phasen)
├── 05_baseline_correction_binning.py        (Step 6: Baseline + 250ms Binning)
├── master_pipeline_authors.py               (Master Orchestration)
├── config_authors.py                        (Konfiguration)
├── __init__.py                              (Python Modul)
├── README.md                                (Ausführliche Dokumentation)
└── QUICKSTART.md                            (Schnellanleitung)

output/preprocessing_authors/                ← Output Verzeichnis
├── (Wird während Processing gefüllt)
└── qc/                                      (Quality Control Plots)
```

---

## Implementierte Preprocessing-Schritte der Autoren

### ✓ Step 1: Common Average Reference (CAR)
Referenziert alle Kanäle auf ihren Durchschnitt  
**Ausgabe:** `{subject}_car.fif`

### ✓ Step 2: Noisy Channel Identification
Automatische Varianz-basierte Detektion + Visualisierung  
**Ausgabe:** `{subject}_noisy_channels.json` + QC-Plots

### ✓ Step 3: Bad Channel Interpolation
Rekonstruiert fehlerhafte Kanäle mit sphärischen Splines  
**Ausgabe:** `{subject}_interpolated.fif`

### ✓ Step 4: Downsampling
Reduziert von 2048 Hz auf 256 Hz  
**Ausgabe:** `{subject}_downsampled.fif`

### ✓ Step 5: Epoching (3 Phasen)
Teilt in Decision, Response, Feedback Phasen auf
- **Decision Phase:** -200ms bis +2000ms
- **Response Phase:** -200ms bis +2000ms
- **Feedback Phase:** -200ms bis +1000ms

**Ausgabe:** `{subject}_{decision/response/feedback}-epo.fif`

### ✓ Step 6: Baseline Correction & Time Binning
- Baseline Correction: -200ms bis 0ms
- Time Binning: 250ms Fenster (resultiert in 20 Bins für 0-5000ms)

**Ausgabe:** `{subject}_{phase}_binned-epo.fif` + Metadaten

---

## Wichtige Feature: KEIN Filtering!

Die Autoren schreiben explizit:
> "We did not apply filtering, as this has been shown to cause artefacts or temporally smear the signal"

Diese Pipeline **respektiert diese Entscheidung** und wendet bewusst KEIN Filtering an.

---

## Verwendung

### Quick Start

```python
from eeg_pipeline.preprocessing_authors import main

# Pipeline für einzelne oder mehrere Subjects laufen lassen
main(
    subjects_input_files='path/to/sub-01_raw.fif',
    output_dir='output/preprocessing_authors',
    subject_ids=['sub-01']
)
```

### Oder mit Master Pipeline

```python
from eeg_pipeline.preprocessing_authors import AuthorsPreprocessingPipeline

config = {'output_dir': 'output/preprocessing_authors'}
pipeline = AuthorsPreprocessingPipeline(config)
pipeline.run_pipeline('sub-01', 'path/to/sub-01_raw.fif')
```

---

## Output-Locations

| Schritt | Output File | Pfad |
|---------|------------|------|
| Alle | Logs | `eeg_pipeline/preprocessing_authors/logs/` |
| Schritt 2 | Noisy Channels | `output/preprocessing_authors/qc/` |
| **Final** | **Processed Data** | **`output/preprocessing_authors/`** |

---

## Dokumentation

- **README.md** - Ausführliche technische Dokumentation
- **QUICKSTART.md** - Schnellanleitung
- **config_authors.py** - Alle Parameter konfigurierbar
- **Jede .py Datei** - Ausführliche Docstrings und Kommentare
- **Inline Comments** - Zitate aus dem Original-Paper

---

## Basierend auf

**Moerel D., Grootswagers T., Chin J.L.L., Ciardo F., Nijhuis P., Quek G.L., Smit S., Varlet M. (2025)**  
"Neural decoding of competitive decision-making in Rock-Paper-Scissors"  
bioRxiv preprint. https://doi.org/10.1101/2025.01.09.632285

**Methodologie References:**
- FieldTrip Toolbox (original methodology)
- MNE-Python (implementiert hier)
- CoSMoMVPA (für spätere decoding analysis)

---

## Nächste Schritte

1. **Daten vorbereiten**: Raw EEG `.fif` Dateien mit korrekten Event-Markierungen
2. **Pipeline starten**: `master_pipeline_authors.py` mit Ihrer Konfiguration ausführen
3. **QC überprüfen**: Plots in `qc/` Verzeichnis ansehen
4. **Ergebnisse nutzen**: Binned epochs für decoding-Analysen verwenden

---

## Questions & Troubleshooting

- **Fehlende Events**: Die `04_epoch.py` sucht automatisch nach Stim-Channels oder Annotations. Diese müssen in Ihren Raw-Daten vorhanden sein.
- **Speicher**: Für große Dateien: `MEMORY_EFFICIENT = True` in `config_authors.py` verwenden
- **Custom Parameter**: Alle Parameter sind in `config_authors.py` konfigurierbar

---

## ✓ Ready to Use!

Die Pipeline ist **vollständig implementiert** und **sofort einsatzbereit** mit Ihrer Raw-EEG-Daten.

Viel Erfolg beim Preprocessing! 🧠📊
