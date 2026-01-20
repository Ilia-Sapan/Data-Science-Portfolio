# Primary School Students by District (Dusseldorf)

This project explores primary school student counts by district, with breakdowns
by gender and citizenship status. It produces summary statistics and charts that
highlight concentration and composition across districts.

## Data
Raw CSV file stored in `data/raw/` with fields:
- `stadtteil` - district code and name
- `maennlich` - male students
- `weiblich` - female students
- `deutsch` - German students
- `nichtdeutsch` - non-German students

## Structure
- `data/raw/` - original CSV file
- `data/processed/` - cleaned and aggregated tables
- `reports/figures/` - charts
- `src/analysis.py` - analysis script

## Quick start
```bash
py -3.12 -m pip install pandas matplotlib
py -3.12 src/analysis.py
```

## Key results
- Total students: 21314 across 51 districts
- Largest districts by total students: 082, 036, 071
- Highest non-German shares: 013, 098, 043

## Artifacts
- Top districts by total students: `reports/figures/top_districts_total.png`
- Gender split for top districts: `reports/figures/top_districts_gender_split.png`
- Top non-German share: `reports/figures/top_districts_non_german_share.png`
- Tables: `data/processed/cleaned.csv`, `data/processed/top_districts.csv`,
  `data/processed/summary_stats.csv`

## Limitations
- Single snapshot; no time series.
- No population normalization or school capacity data.
