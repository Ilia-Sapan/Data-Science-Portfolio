# Migration and Integration Services in Dusseldorf

This project combines population data by district with the city integration
service registry. The focus is on migration shares over time, spatial service
coverage, and a small predictive model with SHAP explanations.

## Data
Source files (copied locally):
- `data/raw/Bevölkerung_mit_Migrationshintergrund_2022_bis_2024.csv`
- `data/raw/Integrationsberatung in Düsseldorf_1.csv`

## What is analyzed
- Citywide migration share trend (2020-2024).
- District-level differences and the most/least diverse areas.
- Service density per 10k residents and its relation to migration share.
- Spatial proximity of services (nearest-neighbor distances).
- A predictive model for migration share using lagged district features and
  service density, explained with SHAP.

## Feature engineering
- `migration_share` = `pop_migration / pop_total`.
- Lagged migration share and population for each district.
- Growth rate of total population.
- Service count and services per 10k residents.
- Service type counts per district.

## Outputs
Generated charts are saved to `reports/figures/`:
- `migration_share_over_time.html` (interactive)
- `migration_share_heatmap.png`
- `top10_migration_share.png`
- `services_vs_migration_share.png`
- `integration_services_map.html` (interactive)
- `service_nearest_neighbor.png`
- `shap_summary.png`

Tables:
- `reports/migration_summary_by_year.csv`
- `reports/district_service_summary.csv`
- `reports/model_metrics.txt`

## Key insights (from 2024 snapshot)
- Citywide migration share rises from ~42.6% (2020) to ~46.0% (2024).
- Highest migration shares: Stadtmitte (~65.9%), Hassels (~63.0%), Hafen
  (~62.1%), Oberbilk (~60.9%), Flingern Sued (~60.5%).
- Lowest migration shares: Himmelgeist (~21.1%), Hamm (~22.6%), Itter
  (~23.1%), Hubbelrath (~23.5%), Kalkum (~26.2%).
- Service mix is dominated by Welcome Points (17) and Integrationsagenturen
  (13) out of 36 total services.
- Highest services per 10k residents appear in smaller districts such as
  Reisholz, while Oberbilk and Hassels also stand out at larger scale.

## Run
```bash
py -3.12 -m pip install -r projects/07_migration_integration_duesseldorf/requirements.txt
py projects/07_migration_integration_duesseldorf/src/analysis.py
```

## Notes and limitations
- Service locations are treated as static; no time dimension is available.
- District-level service density is a proxy and does not reflect capacity.
- Model outputs are exploratory and intended for interpretation, not policy.
