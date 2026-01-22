# Microclimate in Dortmund (Klimasensoren)

This project explores the city microclimate using open climate sensor data.
It focuses on spatial differences across postal codes, rough land-use proxies
from street names, and sensor quality diagnostics (noise, missing values,
flatlines, and anomalies).

## Data
Source files (copied locally):
- `data/raw/klimasensoren-messdaten.csv`
- `data/raw/klimasensoren-aktuelle-messungen.csv`
- `data/raw/klimasensoren-standorte.csv`

## What is analyzed
- Microclimate heatmap based on current measurements (temperature).
- Temperature differences by postal code.
- Land-use proxy (green/water/urban) inferred from street name keywords.
- Anomaly detection for sudden deviations (robust z-score per device).
- Sensor trust score combining missing data, flatlines, noise, and outliers.

## Outputs
Generated charts are saved to `reports/figures/`:
- `heatmap_temperature.png`
- `plz_temperature.png`
- `landuse_proxy_temperature.png`
- `temperature_anomalies.png`
- `sensor_trust_scores.png`

A summary table is written to `reports/sensor_quality_summary.csv`.

## Run
```bash
py -3.12 -m pip install pandas numpy matplotlib seaborn
py -3.12 projects/06_klimasensoren_mikroklima/src/analysis.py
```

## Notes and limitations
- Land-use categories are heuristic and based on street name keywords.
- Data coverage is short (single snapshot window) so anomalies are indicative.
- Postal code analysis relies on available sensor distribution only.
