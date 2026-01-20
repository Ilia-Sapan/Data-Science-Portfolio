# Gesundheitliche Infrastruktur in Dortmund

Dieses Projekt analysiert offene Standortdaten zu gesundheitsbezogenen Einrichtungen in Dortmund.
Ziel ist es, räumliche Muster zu erkennen, Korrelationen zwischen Kategorien zu prüfen und
erste Insights auf Ebene der Stadtbezirke abzuleiten.

## Daten
Die Rohdaten liegen lokal in `data/raw/` und umfassen u.a.:
- Gesundheitsamt
- Krankenhäuser
- Hospize
- Kurzzeitpflegeeinrichtungen
- Öffentliche Toiletten
- Barrierefreie öffentliche Toiletten
- Trinkwasserbrunnen

**Hinweis**: Es werden ausschließlich absolute Standortzahlen ausgewertet. Bevölkerungszahlen,
Flächendaten oder Bedarfsindikatoren sind nicht enthalten.

## Struktur
- `data/raw/` – originale CSV-Dateien
- `notebooks/gesund_dortmund_eda.ipynb` – EDA, Korrelationen und Visualisierungen
- `src/` – Platz für spätere Skripte

## Analyse-Highlights (Kurzfassung)
- Zentral gelegene Stadtbezirke zeigen eine höhere Dichte über mehrere Kategorien hinweg.
- Toiletten und Trinkwasserbrunnen korrelieren positiv mit stark frequentierten Bezirken.
- Klinische Einrichtungen (Krankenhäuser, Hospize) sind seltener und clusterartig verteilt.
- Barrierefreie Toiletten sind ungleichmäßig verteilt; in einigen Bezirken deutlich höherer Anteil.

## Reproduzierbarkeit
```bash
py -3.12 -m pip install pandas numpy matplotlib seaborn
```

Notebook öffnen und ausführen:
- `notebooks/gesund_dortmund_eda.ipynb`

## Limitationen
- Keine Bevölkerungs- oder Bedarfsdaten zur Normierung der Versorgung.
- Datensätze sind klein und nicht zeitlich versioniert.
- Korrelationen basieren nur auf aggregierten Bezirkssummen.
