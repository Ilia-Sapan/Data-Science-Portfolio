# Data Science Portfolio

This repository contains my data science pet projects and practical experiments.
The goal is to demonstrate applied analytics, clear storytelling, and pragmatic
modeling using Python.

## Structure

projects/
00_template/ # Base template for new projects
01_gesund_dortmund/ # Health infrastructure in Dortmund
02_unfaelle_2019_2024/ # Accident victims time series
03_covid-19 in Dusseldorf/ # COVID-19 time series
04_grundschule_stadtteile/ # Primary school students by district
05_wohnungswahl-duesseldorf/ # Housing choice project for Dusseldorf and surroundings

Each project is self-contained and includes:
- problem statement and goals
- data sources and cleaning
- analysis and modeling
- insights and next steps

## Projects

| #  | Project | Description | Tools |
|----|--------|-------------|-------|
| 01 | Gesund Dortmund | Health infrastructure, correlations, facility distribution | pandas, numpy, matplotlib, seaborn |
| 02 | Unfaelle 2019-2024 | EDA, summary stats, forecast 2025-2026 | pandas, numpy, matplotlib |
| 03 | COVID-19 in Dusseldorf | Time series EDA, lag analysis, baseline forecasting | pandas, numpy, matplotlib |
| 04 | Grundschule Stadtteile | Primary school students by district, distributions, shares | pandas, matplotlib |
| 05 | Wohnungswahl Dusseldorf | Housing affordability + crime trends + nationality structure, BORIS NRW | pandas, numpy, matplotlib, seaborn, requests |

## Tech Stack

- Python 3.12
- pandas, numpy, scipy
- matplotlib, seaborn
- scikit-learn
- Jupyter Notebook

## How to Run

```bash
py -3.12 -m pip install -r projects/wohnungswahl-duesseldorf/requirements.txt
py projects/wohnungswahl-duesseldorf/src/analysis.py
```

Notes

This repository is intended for educational and demonstration purposes.
Large datasets are not stored in the repository.
