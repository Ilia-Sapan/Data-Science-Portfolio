from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "COVID_Duesseldorf.csv"
FIGURES_DIR = PROJECT_ROOT / "figures"

DATE_COL = "Datum"

CUMULATIVE_COLS = [
    "bestaetigte Faelle kumulativ",
    "Todesfaelle",
    "Genesene",
    "Impfungen gesamt",
]

TARGET_COL = "7-Tages-Inzidenz"

KEY_COLS = [
    "Aktive Faelle",
    "Genesene",
    "Todesfaelle",
    "bestaetigte Faelle kumulativ",
    "7-Tages-Inzidenz",
    "Anzahl in Krankenhäusern",
    "davon auf Intensivstationen",
    "Abstriche gesamt",
    "haeusliche Quarantaene",
    "Impfungen gesamt",
]
