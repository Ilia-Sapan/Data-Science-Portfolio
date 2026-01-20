# Wohnungswahl Dusseldorf

Data-driven pet project that compares affordability, migration structure, and crime trends to support housing decisions for Dusseldorf and nearby municipalities.

## What this project answers
- Where housing is most affordable in Dusseldorf and the surrounding municipalities using Bodenrichtwerte (BORIS NRW).
- What the foreign-national population structure looks like by Stadtteil (Dusseldorf only).
- How crime trends evolve in Dusseldorf (city-level) with short-term forecasts.

## Data sources
- Local folder: `Wohnungs/` (foreign population, crime reports, housing supply time series).
- BORIS NRW open data: ArcGIS REST service `boris_nw_bodenrichtwerte_current`.

## How to run
```bash
py -m pip install -r projects/wohnungswahl-duesseldorf/requirements.txt
py projects/wohnungswahl-duesseldorf/src/analysis.py
```

## Outputs
- Figures in `projects/wohnungswahl-duesseldorf/reports/figures/`
  - `duesseldorf_affordable.png`
  - `surroundings_affordable.png`
  - `nationality_top3_share.png`
  - `foreign_share_2020.png`
  - `crime_forecast.png`
  - `housing_backlog_forecast.png`
- Clean datasets in `projects/wohnungswahl-duesseldorf/data/`
- Key takeaways in `projects/wohnungswahl-duesseldorf/reports/insights.json`

## Key insights (from latest run)
- Most affordable Dusseldorf districts by median Bodenrichtwert: Hellerhof, Derendorf, Garath, Lierenfeld, Lohausen.
- Cheapest surrounding municipalities: Dormagen, Erkrath, Mettmann, Kaarst, Monheim am Rhein.
- Crime totals show an upward trend with a short-term forecast of ~81.6k in 2025 and ~84.4k in 2026.
- Housing construction backlog is projected to continue increasing (trend-based).

## Notes and limitations
- Crime statistics in the PKS reports are city-level; district-level crime is not published in structured tables, so the crime index is shared across all districts.
- BORIS data reflects land value zones, not rent prices; it is used as a consistent affordability proxy.
- Forecasts use a simple linear trend to provide directional signals, not precise predictions.

## Next ideas
- Add commute-time scoring using public transport APIs.
- Pull official population counts per Stadtteil to compute crime per 100k if district data becomes available.
- Extend BORIS queries to include Immobilienrichtwerte for a second price lens.
