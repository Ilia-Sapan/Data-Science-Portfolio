# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import folium
from folium.plugins import MarkerCluster
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import shap

PROJECT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT / "data" / "raw"
REPORTS_DIR = PROJECT / "reports"
FIG_DIR = REPORTS_DIR / "figures"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

bev_path = next(DATA_DIR.glob("*Bev*csv"))
integ_path = next(DATA_DIR.glob("*Integrationsberatung*csv"))

bev = pd.read_csv(bev_path, sep=";")
integ = pd.read_csv(integ_path, sep=";")

bev = bev.rename(columns={
    "Jahr": "year",
    "Stadtteil_Nummer": "district_no",
    "Stadtteil_Name": "district",
    "personen_insgesamt": "pop_total",
    "personen_mit_migrationshintergrund": "pop_migration",
    "personen_ohne_migrationshintergrund": "pop_no_migration",
})

bev["district_no"] = bev["district_no"].astype(int)

bev["migration_share"] = bev["pop_migration"] / bev["pop_total"]

integ = integ.rename(columns={
    "LATITUDE": "lat",
    "LONGITUDE": "lon",
    "ART": "service_type",
    "NAME": "name",
    "EINRICHTUNG": "org",
    "STADTTEIL": "district",
    "STADTTEILNR": "district_no",
})
integ["district_no"] = integ["district_no"].astype(int)

# Service counts by district
service_counts = (
    integ.groupby("district_no")
    .size()
    .reset_index(name="service_count")
)

service_types = (
    integ.pivot_table(
        index="district_no",
        columns="service_type",
        values="name",
        aggfunc="count",
        fill_value=0,
    )
    .reset_index()
)

service_summary = service_counts.merge(service_types, on="district_no", how="left")

bev = bev.merge(service_summary, on="district_no", how="left")
bev["service_count"] = bev["service_count"].fillna(0)
bev["service_per_10k"] = (bev["service_count"] / bev["pop_total"]) * 10000

# City-level summary
city_year = (
    bev.groupby("year", as_index=False)
    .agg(
        pop_total=("pop_total", "sum"),
        pop_migration=("pop_migration", "sum"),
    )
)
city_year["migration_share"] = city_year["pop_migration"] / city_year["pop_total"]
city_year.to_csv(REPORTS_DIR / "migration_summary_by_year.csv", index=False)

# District-level summary for the latest year
latest_year = int(bev["year"].max())
district_latest = bev[bev["year"] == latest_year].copy()
district_latest = district_latest.sort_values("migration_share", ascending=False)
district_latest.to_csv(REPORTS_DIR / "district_service_summary.csv", index=False)

# Plot 1: City migration share over time (interactive)
fig_line = px.line(
    city_year,
    x="year",
    y="migration_share",
    markers=True,
    title="Dusseldorf: Migration Share Over Time",
)
fig_line.update_yaxes(tickformat=".1%")
fig_line.write_html(FIG_DIR / "migration_share_over_time.html")

# Plot 2: Heatmap of migration share by district (top 15 by population)
top15 = (
    district_latest.sort_values("pop_total", ascending=False)
    .head(15)["district"]
    .tolist()
)
heatmap_data = (
    bev[bev["district"].isin(top15)]
    .pivot(index="district", columns="year", values="migration_share")
    .loc[top15]
)
plt.figure(figsize=(10, 6))
sns.heatmap(heatmap_data, cmap="YlGnBu", fmt=".0%", annot=True)
plt.title("Migration Share by District (Top 15 by Population)")
plt.tight_layout()
plt.savefig(FIG_DIR / "migration_share_heatmap.png", dpi=200)
plt.close()

# Plot 3: Top 10 districts by migration share (latest year)
plt.figure(figsize=(9, 5))
subset = district_latest.head(10).copy()
sns.barplot(data=subset, x="migration_share", y="district", color="#2a9d8f")
plt.title(f"Top 10 Districts by Migration Share ({latest_year})")
plt.xlabel("Migration share")
plt.ylabel("District")
plt.gca().xaxis.set_major_formatter(lambda x, pos: f"{x:.0%}")
plt.tight_layout()
plt.savefig(FIG_DIR / "top10_migration_share.png", dpi=200)
plt.close()

# Plot 4: Service density vs migration share (latest year)
plt.figure(figsize=(7, 5))
ax = sns.regplot(
    data=district_latest,
    x="service_per_10k",
    y="migration_share",
    scatter_kws={"alpha": 0.7},
    line_kws={"color": "black"},
)
ax.set_title(f"Services per 10k vs Migration Share ({latest_year})")
ax.set_xlabel("Services per 10k residents")
ax.set_ylabel("Migration share")
ax.yaxis.set_major_formatter(lambda x, pos: f"{x:.0%}")
plt.tight_layout()
plt.savefig(FIG_DIR / "services_vs_migration_share.png", dpi=200)
plt.close()

# Plot 5: Service map (interactive)
center_lat = integ["lat"].mean()
center_lon = integ["lon"].mean()
service_map = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles="CartoDB positron")
cluster = MarkerCluster().add_to(service_map)
colors = {
    "Integrationsagentur": "blue",
    "Welcome Point": "green",
    "Welcome Center": "orange",
    "Jugendmigrationsdienst": "purple",
    "Sinti-Beratung": "red",
    "Fokusteam Flüchtlinge": "darkred",
    "Integration Point Düsseldorf": "cadetblue",
}
for _, row in integ.iterrows():
    color = colors.get(row["service_type"], "gray")
    popup = f"{row['service_type']}<br>{row['name']}<br>{row.get('district', '')}"
    folium.CircleMarker(
        location=[row["lat"], row["lon"]],
        radius=5,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.8,
        popup=popup,
    ).add_to(cluster)
service_map.save(FIG_DIR / "integration_services_map.html")

# Nearest neighbor distance for services
coords = integ[["lat", "lon"]].to_numpy()

# Haversine distance in km
R = 6371.0
lat = np.deg2rad(coords[:, 0])
lon = np.deg2rad(coords[:, 1])

lat1 = lat[:, None]
lat2 = lat[None, :]
lon1 = lon[:, None]
lon2 = lon[None, :]

dlat = lat2 - lat1
dlon = lon2 - lon1

a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
c = 2 * np.arcsin(np.sqrt(a))

dist_matrix = R * c
np.fill_diagonal(dist_matrix, np.nan)
integ["nearest_km"] = np.nanmin(dist_matrix, axis=1)

plt.figure(figsize=(8, 5))
sns.boxplot(data=integ, x="service_type", y="nearest_km")
plt.xticks(rotation=35, ha="right")
plt.title("Nearest Neighbor Distance by Service Type")
plt.xlabel("Service type")
plt.ylabel("Nearest neighbor distance (km)")
plt.tight_layout()
plt.savefig(FIG_DIR / "service_nearest_neighbor.png", dpi=200)
plt.close()

# Modeling: predict migration share using lagged district features + service density
bev_sorted = bev.sort_values(["district_no", "year"]).copy()
bev_sorted["lag_total"] = bev_sorted.groupby("district_no")["pop_total"].shift(1)
bev_sorted["lag_migration_share"] = bev_sorted.groupby("district_no")["migration_share"].shift(1)
bev_sorted["lag_migration_count"] = bev_sorted.groupby("district_no")["pop_migration"].shift(1)
bev_sorted["growth_total"] = (
    (bev_sorted["pop_total"] - bev_sorted["lag_total"]) / bev_sorted["lag_total"]
)

model_df = bev_sorted.dropna(subset=["lag_total", "lag_migration_share"]).copy()

features = [
    "year",
    "lag_total",
    "lag_migration_share",
    "lag_migration_count",
    "growth_total",
    "service_count",
    "service_per_10k",
]
X = model_df[features]
y = model_df["migration_share"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(
    n_estimators=300,
    max_depth=8,
    random_state=42,
)
model.fit(X_train, y_train)

preds = model.predict(X_test)
mae = mean_absolute_error(y_test, preds)
r2 = r2_score(y_test, preds)

with open(REPORTS_DIR / "model_metrics.txt", "w", encoding="utf-8") as f:
    f.write("RandomForestRegressor\n")
    f.write(f"MAE: {mae:.4f}\n")
    f.write(f"R2: {r2:.4f}\n")

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_train)

plt.figure(figsize=(8, 5))
shap.summary_plot(shap_values, X_train, show=False)
plt.tight_layout()
plt.savefig(FIG_DIR / "shap_summary.png", dpi=200)
plt.close()

print("Done. Reports saved to:", REPORTS_DIR)
