# Sanity Checks für EEG-Pipeline

Diese Verzeichnis enthält Sanity-Check Scripts für jeden Schritt der EEG-Preprocessing-Pipeline. Die Checks überprüfen Metadaten, Datenqualität und plausible Wertebereiche.

## Verfügbare Sanity Checks

| Step | Script | Überprüft |
|------|--------|-----------|
| 00 | `scripts/sc_00_downsample.py` | Downsampling erfolgreich, Sampling-Rate korrekt, Datengröße reduziert |
| 01 | `scripts/sc_01_split_players.py` | Spieler aufgeteilt, Kanäle korrekt zugeordnet, Kanal-Typen gesetzt |
| 02 | `scripts/sc_02_rename_montage.py` | Kanäle korrekt umbenannt (BioSemi → 10-20), Montage gesetzt |
| 03 | `scripts/sc_03_bad_channels_detect.py` | Bad-Channels identifiziert, QC-Reports erstellt |
| 04 | `scripts/sc_04_interpolate.py` | Interpolation durchgeführt, Bad-Listen geleert |
| 05 | `scripts/sc_05_filter.py` | Bandpass-Filter angewendet (1-40 Hz), PSD-Reduktion sichtbar |
| 06 | `scripts/sc_06_ica.py` | ICA Komponenten extrahiert, EOG-Artefakte erkannt |
| 07 | `scripts/sc_07_epoch.py` | Epochs erstellt, Event-Typen korrekt, Zeit-Fenster passen |

## Verwendung

- Notebooks liegen unter `sanity_checks/notebooks/`
- Python-Checks liegen unter `sanity_checks/scripts/`

### Einzelnen Sanity Check ausführen
```bash
python sanity_checks/scripts/sc_00_downsample.py
```

### Alle Sanity Checks in Folge (optional, können später in Master-Pipeline integriert werden)
```bash
for script in sanity_checks/scripts/sc_*.py; do python $script; done
```

## Ausgaben

- **Console Output**: Detaillierte Überprüfungs-Ergebnisse mit ✓ (bestanden) und ⚠/ERROR (Warnung/Fehler)
- **QC-Reports**: TSV-Dateien und PNG-Plots werden in `output/qc/` gespeichert
  - `sub-01_P1_bad_channels_detect.tsv` (aus Step 03)
  - `sub-01_P1_filter_psd_comparison.png` (aus Step 05)
  - `sub-01_P1_ica.fif` (aus Step 06)

## Hinweise

- Sanity Checks laden Daten mit `preload=False` um Speicher zu sparen
- Für große Zeitfenster werden begrenzte Stichproben verwendet (z.B. erste 60-120 Sekunden)
- Checks sind so kurz gehalten, dass sie nach jedem entsprechenden Pipeline-Schritt schnell laufen können

## Integration in Pipeline

Optional können diese Sanity Checks automatisch nach jedem entsprechenden Schritt in `eeg_pipeline/preprocessing/master_pipeline.py` aufgerufen werden.
