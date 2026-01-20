# Accident Victims (2019-2024)

This project analyzes monthly accident victim counts for 2019-2024, produces
summary statistics, visualizations, and a simple forecast for 2025-2026.

## Data
Raw CSV files are stored in `data/raw/` (2019-2024). Fields:
- `Monat` - month
- `Personen_insgesamt` - total victims
- `Leichtverletzte` - minor injuries
- `Schwerverletzte` - serious injuries
- `Getoetete` - fatalities

## Structure
- `data/raw/` - original CSV files
- `data/processed/` - cleaned and aggregated tables
- `reports/figures/` - charts
- `src/analysis.py` - analysis script

## Quick start
```bash
py -3.12 -m pip install pandas numpy matplotlib
py -3.12 src/analysis.py
```

## Key results
- Lowest year total: 2021 - 2709 people
- Highest year total: 2023 - 3158 people
- Seasonality: peak in September and June, lowest in February

## Forecast (2025-2026)
Method: for each month, fit a linear trend over years and forecast 2025-2026.

- 2025: about 3070 people
- 2026: about 3092 people

## Artifacts
- Monthly trends: `reports/figures/monthly_trends.png`
- Yearly totals: `reports/figures/yearly_totals.png`
- Forecast chart: `reports/figures/forecast_2025_2026.png`
- Tables: `data/processed/monthly.csv`, `data/processed/yearly.csv`,
  `data/processed/forecast_2025_2026.csv`, `data/processed/summary_stats.csv`

## Limitations
- No geographic or crash-type breakdowns.
- Forecast is a simple trend without external factors.
