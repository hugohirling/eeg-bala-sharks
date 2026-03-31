# Sanity Checks für EEG-Pipeline

Diese Verzeichnis enthält Sanity-Check Scripts für jeden Schritt der EEG-Preprocessing-Pipeline. Die Checks überprüfen Metadaten, Datenqualität und plausible Wertebereiche.

## Warum diese Checks wichtig sind

Diese Sanity-Check-Suite ist nicht nur als technische Fehlersuche gedacht, sondern auch als **bewertbare Dokumentation** für die Pipeline-Entscheidungen.

- **Code readability & documentation:** Die Skripte enthalten jetzt direkt im Code dokumentierte Motivation, Parameternotizen und Interpretationshilfen.
- **Reproducibility & modularity:** Relevante Schwellenwerte kommen aus `preprocessing/config.py` oder sind im Skript explizit dokumentiert; die CSV-Summaries machen Runs vergleichbar.
- **Sanity checks & discussion:** Konsolen- und CSV-Ausgaben enthalten neben Statusmeldungen auch kurze Sätze wie `This seems correct because ...` oder `This is strange because ...`.
- **Result interpretation:** Die Visualisierungen sind als Before/After-Vergleiche aufgebaut, damit sichtbare Änderungen direkt argumentiert werden können.

Die zentrale Leitfrage pro Schritt lautet daher:

> Verändert dieser Preprocessing-Schritt genau das, was er verändern soll, und lässt er alles andere unverändert?

## Verfügbare Sanity Checks

Die **primären Entry-Points** sind jetzt die Step-Skripte `sc_00_...` bis `sc_07_...`. Für die Steps 00, 01, 02, 03, 04 und 07 unterstützen diese Skripte den Modus `--mode check`, `--mode viz` oder `--mode both`.

| Step | Überprüfungs-Script | Visualisierungs-Script | Beschreibung |
|------|--------|--------|-----------|
| 00 | `sc_00_downsample.py` | optional via `--mode viz` | Downsampling: Sampling-Rate, Datenreduktion, Zeit-Serie & PSD Vergleich |
| 01 | `sc_01_split_players.py` | optional via `--mode viz` | Split-Players: P1 vs P2 Datenverteilung, Duration, Kanäle |
| 02 | `sc_02_rename_montage.py` | optional via `--mode viz` | Rename & Montage: Kanal-Namen-Mapping, Sensor-Layout 2D Topomap |
| 03 | `sc_03_bad_channels_detect.py` | optional via `--mode viz` | Bad Channels: Topomap mit Markierungen, Amplitude vor/nach |
| 04 | `sc_04_interpolate.py` | optional via `--mode viz` | Interpolation: Zeitreihen der interpolierten Kanäle, Amplitude, Montage |
| 05 | `sc_05_filter.py` | (in sc_05_filter.py) | Bandpass-Filter (1-40 Hz): PSD vor/nach Vergleich |
| 06 | `sc_06_ica.py` | (in sc_06_ica.py) | ICA Artifact Removal: Komponenten, EOG-Erkennung |
| 07 | `sc_07_epoch.py` | optional via `--mode viz` | Epoching: Event-Verteilung, Beispiel-Epochs, PSD Vergleich, Baseline-Fenster |
| 08 | — | `sc_08_pipeline_progression_plots.py` | **Gesamt-Übersicht**: GFP + PSD über alle Stages (Original → ICA) |

## Verwendung

### Kombinierte Step-Skripte für Check und Visualisierung

Die bevorzugte Nutzung für die gepaarten Steps ist jetzt jeweils ein gemeinsames Step-Skript:

```bash
# Step 00: Text-Check
python sanity_checks/scripts/sc_00_downsample.py --mode check

# Step 00: Nur Visualisierung
python sanity_checks/scripts/sc_00_downsample.py --mode viz --subjects 01,02 --duration 30

# Step 00: Check + Visualisierung
python sanity_checks/scripts/sc_00_downsample.py --mode both --subjects 01,02 --duration 30

# Step 03: Check + Visualisierung
python sanity_checks/scripts/sc_03_bad_channels_detect.py --mode both --subjects 01,02

# Step 04: Nur Visualisierung
python sanity_checks/scripts/sc_04_interpolate.py --mode viz --subjects 01,02 --duration 30

# Step 07: Nur Visualisierung
python sanity_checks/scripts/sc_07_epoch.py --mode viz --subjects 01,02
```

### Automatische Qualitätsüberprüfung (Text-Output)

Weitere Checks (Step 05-07) sind integriert:

```bash
# Step 05: Filter überprüfen
python sanity_checks/scripts/sc_05_filter.py

# Step 06: ICA überprüfen
python sanity_checks/scripts/sc_06_ica.py

# Step 07: Epoching überprüfen
python sanity_checks/scripts/sc_07_epoch.py
```

### Übersicht über alle Preprocessing-Stages

```bash
# GFP + PSD über die gesamte Pipeline (Original → alle verfügbaren Steps)
python sanity_checks/scripts/sc_08_pipeline_progression_plots.py
```

### Verarbeitete Daten schnell visualisieren

```bash
# Gefilterte/ICA-bereinigte Daten plotten (mit Time Series, PSD, Topomaps, Amplituden)
python sanity_checks/scripts/sc_plot_preprocessed_data.py --subjects 01,02 --step 06 --duration 60
```

### Alle Sanity Checks in Folge (optional)
```bash
for script in sanity_checks/scripts/sc_*.py; do python "$script"; done
```

- **Notebooks liegen unter** `sanity_checks/notebooks/`
- **Python-Checks liegen unter** `sanity_checks/scripts/`

## Ausgaben

### Visualisierungs-Plots (output/qc/)

Jeder Preprocessing-Step generiert spezifische Visualisierungs-Plots:

**Step 00 - Downsample:**
- `sub-XX_P1_downsample_timeseries_comparison.png` — Zeitreihen Original vs. Downsampled
- `sub-XX_P1_downsample_psd_comparison.png` — Frequenzspektrum Vergleich
- `sub-XX_P1_downsample_statistics_comparison.png` — Amplitude & Dateigröße

**Step 01 - Split Players:**
- `sub-XX_split_players_data_summary.png` — P1 vs P2 Datenverteilung (Duration, Kanäle, Größe)
- `sub-XX_split_players_amplitude_dist.png` — Amplituden-Histogramm pro Player

**Step 02 - Rename & Montage:**
- `sub-XX_P1_montage_topomap.png` — Sensor-Layout Topomap
- `sub-XX_P1_montage_channel_mapping.png` — Kanal-Namen Vorher/Nachher
- `sub-XX_P1_montage_coverage_stats.png` — Standard 10-20 System Coverage

**Step 03 - Bad Channels Detection:**
- `sub-XX_P1_bad_channels_topomap.png` — Topomap mit Markierten Bad-Channels
- `sub-XX_P1_bad_channels_amplitudes.png` — Amplitude Vergleich Good vs. Bad
- `sub-XX_P1_bad_channels_qc_metrics.png` — QC-Metriken (optional)

**Step 04 - Interpolation:**
- `sub-XX_P1_interpolate_montage_comparison.png` — Sensor-Layout Vorher/Nachher
- `sub-XX_P1_interpolate_timeseries.png` — Zeitreihen interpolierter Kanäle
- `sub-XX_P1_interpolate_statistics.png` — Amplitude vor/nach Interpolation

**Step 07 - Epoching:**
- `sub-XX_P1_epoch_event_distribution.png` — Histogram: Anzahl Epochen pro Event-Typ
- `sub-XX_P1_epoch_examples.png` — Beispiel-Epochs mit Baseline-Fenster
- `sub-XX_P1_epoch_psd_comparison.png` — PSD Vergleich: kontinuierlich vs. epochiert
- `sub-XX_P1_epoch_statistics.png` — Metadaten-Summary (Dimensionen, Baseline, Bad Channels)

**Step 05-08:**
- `sub-XX_P1_filter_psd_comparison.png` — Filter-Effekt (PSD)
- `sub-XX_P1_ica_detailed_comparison.png` — ICA Amplituden-Reduktion
- `sub-XX_P1_epochs_sample.png` — Sample Epochs-Visualisierung
- `sub-XX_P1_pipeline_progression_gfp_psd.png` — GFP + PSD über alle Stages

### Console Output

- Detaillierte Überprüfungs-Ergebnisse mit ✓ (bestanden) und ⚠/ERROR (Warnung/Fehler)
- Statistik-Zusammenfassungen pro Subject/Player

## Hinweise

- Sanity Checks laden Daten mit `preload=False` um Speicher zu sparen
- Für große Zeitfenster werden begrenzte Stichproben verwendet (z.B. erste 60-120 Sekunden)
- Checks sind so kurz gehalten, dass sie nach jedem entsprechenden Pipeline-Schritt schnell laufen können
- Text-Checks exportieren strukturierte CSV-Dateien mit `Status`, `Category`, `Message`, `Rationale` und `ParameterNote`

## Wie die Diskussion formuliert werden sollte

Für die Bewertung reicht ein reines `✓` oft nicht aus. Die Diskussion sollte pro Step mindestens einen der folgenden Satztypen enthalten:

- `This seems correct because ...`
- `This is strange because ...`
- `This parameter choice is justified because ...`

Beispiele:

- `This seems correct because the downsampled file keeps the same duration while reducing the sampling rate and estimated size.`
- `This is strange because the bad-channel fraction is unusually high and may indicate a recording-wide quality problem.`
- `This parameter choice is justified because the 1-40 Hz filter keeps conventional EEG bands while suppressing slow drifts and high-frequency noise.`

## Grading-Strategie (20% "Sanity Checks & Visualizations & Discussion")

Diese Sanity-Check-Suite adressiert alle **8 Preprocessing-Steps** mit umfangreichen Visualisierungen:

### Coverage ✓
- **Steps 00-04:** Dedizierte Visualisierungs-Skripte mit Before/After-Vergleichen
- **Steps 05-07:** Integrierte Plots in bestehenden Überprüfungs-Skripten
- **Gesamt-Übersicht:** `sc_08_pipeline_progression_plots.py` zeigt GFP + PSD über alle Stages

### Visualisierungs-Qualität ✓
- **Time Series:** Wellenform-Vergleiche (z.B. Downsample, Interpolation)
- **Frequenz-Domain:** PSD-Vergleiche (z.B. Filter, Downsample)
- **Topomaps:** Sensor-Layout & Bad-Channel-Markierungen (z.B. Montage, Bad Channels)
- **Statistiken:** Amplitude, Datenverteilung, QC-Metriken pro Step

### Modularität ✓
- Jeder Step hat ein eigenes Visualisierungs-Skript → leicht zu wartbar/erweiterbar
- Einheitliche Code-Struktur (Argparse, Logging, Output-Dirs)
- Wiederverwendbare Plotting-Funktionen

### Diskussion
Zusätzlich sollte eine **SANITY_CHECK_DISCUSSION.md** erstellt werden, die für jeden Step erklärt:
- Was zeigt der Plot?
- Sind die sichtbaren Änderungen erwartbar/plausibel?
- Welche Metriken werden überwacht?
- Warum sind genau diese Parameter oder Schwellenwerte sinnvoll?
- Beispiel:
  ```
  ## Step 00: Downsample
  **Erwartung:** 2048 Hz → 200 Hz (10x Reduktion), Dateigröße ~10% Original, 
  Power oberhalb 100 Hz sollte komplett weg sein
  **Beobachtung:** ✓ Alle Metriken erfüllt
  **Interpretation:** This seems correct because the lower sampling rate reduces storage cost without changing channel count or duration.
  
  ## Step 03: Bad Channels
  **Erwartung:** Typischerweise 1-5 der 64 Kanäle als schlecht markiert
  **Beobachtung:** 2 Kanäle markiert (Fp1, EOG links) — plausibel
  **Interpretation:** This seems correct because the affected channels stand out locally and do not reflect a global amplitude shift across the whole cap.
  ```
