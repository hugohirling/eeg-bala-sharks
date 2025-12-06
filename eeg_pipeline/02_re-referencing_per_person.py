# 02_rereference.py
import mne
from utils import load_raw, save_raw
import config

def rereference_subject(file_path):
    """
    Re-referenziert eine Raw-Datei pro Person auf B16 (Cz-Äquivalent bei BioSemi 64).
    Fällt auf Biosemi Default (DRL/CMS) zurück, falls Kanal nicht existiert.
    """
    print(f"\nProcessing {file_path.name}...")
    raw = load_raw(file_path)

    # --- Montage setzen (falls noch nicht gesetzt) ---
    raw.set_montage('biosemi64', on_missing='ignore')

    # --- Re-Referencing ---
    # Bestimme Person aus Dateiname
    person = '1' if '_P1_' in str(file_path) else '2'
    ref_channel = f"{person}-B16"

    if ref_channel in raw.ch_names:
        raw.set_eeg_reference(ref_channels=[ref_channel])
        print(f"Re-referenced using {ref_channel}")
    else:
        print(f"{ref_channel} not found, using Biosemi default (DRL/CMS)")

    # --- Speichern ---
    out_file = file_path.with_name(file_path.stem + '_reref.fif')
    save_raw(raw, out_file)
    print(f"Saved re-referenced data to {out_file}")
    return out_file

if __name__ == "__main__":
    for subj in config.SUBJECTS:
        for person in ['P1', 'P2']:
            file_path = config.OUTPUT_DIR / f"sub-{subj}_{person}_raw.fif"
            rereference_subject(file_path)
