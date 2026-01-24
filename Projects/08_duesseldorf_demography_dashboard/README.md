# Dusseldorf Demography Dashboard

Interactive Streamlit dashboard that blends time-series analytics, cohort dynamics,
and a geo view of migration background for Dusseldorf.

## What you get

- Population trend 2012-2026 (2025-2026 linear projection)
- Hauptwohnsitz vs Nebenwohnsitz split
- Birth-year pyramid by gender (2018-2024)
- Migration background by Stadtbezirk + top Stadtteile
- Forecast to 2035 by Stadtteil + growth since 2012
- Stadtteil spotlight time series + annual change chart

## Key insights (from the included data)

- Total population grew by +49,464 from 2012 to 2024 (+8.1%).
- Nebenwohnsitz share in the latest year sits around 2.45%.
- Highest migration background shares in 2024: Stadtmitte, Hassels, Hafen,
  Oberbilk, Flingern Sued (all above 60%).
- Strongest forecast growth to 2035: Hubbelrath (+191.6%), followed by
  Himmelgeist (+58%).
- Largest population gains since 2012: Flingern Nord, Heerdt, Pempelfort,
  Rath, Bilk.

## Run it

```bash
py -3.12 -m pip install -r requirements.txt
py -3.12 -m streamlit run app.py
```

## Structure

- `app.py` - Streamlit dashboard
- `data/` - source CSV files + Stadtbezirk centroids
- `requirements.txt` - dependencies

## Notes

- The migration map uses simplified Stadtbezirk centroids (not official
  boundaries) to keep the map lightweight.
- Sources: Düsseldorf open data CSVs and the official 2035 forecast.
