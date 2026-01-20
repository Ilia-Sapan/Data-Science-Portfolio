# COVID-19 in Dusseldorf (Time Series Analysis)

This project analyzes COVID-19 indicators for Dusseldorf with an emphasis on time series EDA, lag relationships, and a simple, honest forecasting baseline. Data ends on 2023-01-09; retro-corrections are possible. Since 2021-02-05 the source is RKI, so population differences and reporting changes can affect comparisons.

## Structure
- `COVID_Duesseldorf.csv`: raw dataset.
- `notebooks/01_eda_timeseries.ipynb`: EDA workflow and figure generation.
- `src/`: reusable data cleaning, feature engineering, modeling, and plotting.
- `figures/`: generated charts.

## Setup
```bash
python -m venv .venv
.venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
```

## Generate figures
```bash
python -m src.generate_figures
```

## Mini-modeling (7-day horizon)
Use the notebook or call `src/modeling.py` from a scratch script. Models include:
- Naive baseline
- Seasonal naive (7-day)
- Linear regression and Ridge on lag/rolling/time features

## Outputs
- `figures/timeseries_key.png`: key indicators over time
- `figures/tests_vs_incidence.png`: tests vs incidence
- `figures/correlations.png`: correlation heatmap
- `figures/lag_correlations.png`: lag correlation curves
