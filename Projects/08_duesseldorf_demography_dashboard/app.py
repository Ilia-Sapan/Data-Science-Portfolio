from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

DATA_DIR = Path(__file__).resolve().parent / "data"

st.set_page_config(
    page_title="Dusseldorf Demography Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600&family=IBM+Plex+Mono:wght@400;600&display=swap');

html, body, [class*="css"]  {
    font-family: 'Space Grotesk', sans-serif;
}

section.main > div {
    background: radial-gradient(1200px 800px at 10% 0%, #f6f4ef 0%, #f0f2f8 45%, #f8f0f4 100%);
    padding: 2rem 2.5rem 3rem;
    border-radius: 24px;
}

h1, h2, h3 {
    letter-spacing: -0.02em;
}

.metric-card {
    background: #ffffffcc;
    border: 1px solid #e6e7ef;
    border-radius: 18px;
    padding: 1rem 1.2rem;
    box-shadow: 0 12px 30px rgba(15, 20, 35, 0.08);
}

.block-label {
    font-family: 'IBM Plex Mono', monospace;
    text-transform: uppercase;
    font-size: 0.75rem;
    letter-spacing: 0.2em;
    color: #6b6f7a;
}

</style>
""",
    unsafe_allow_html=True,
)


@st.cache_data
def read_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / name, sep=";", encoding="utf-8-sig")


@st.cache_data
def load_total_population() -> pd.DataFrame:
    df = read_csv("Gesamtzahl der Bevölkerung Düsseldorfs seit 2012.csv")
    df["Jahr"] = pd.to_numeric(df["Jahr"], errors="coerce")
    df["Bevoelkerung"] = pd.to_numeric(df["Bevoelkerung"], errors="coerce")
    return df.dropna()


@st.cache_data
def load_residence() -> pd.DataFrame:
    df = read_csv("Bevölkerung mit Hauptwohnsitz und Nebenwohnsitz seit 2013.csv")
    df["Jahr"] = pd.to_numeric(df["Jahr"], errors="coerce")
    for col in ["Hauptwohnsitz", "Nebenwohnsitz"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna()


@st.cache_data
def load_birth_years() -> pd.DataFrame:
    frames = []
    files = sorted(DATA_DIR.glob("Bevölkerung nach Geburtsjahrgängen*.csv"))
    for file_path in files:
        match = re.search(r"311220(\d{2})", file_path.name)
        if not match:
            continue
        year = 2000 + int(match.group(1))
        df = pd.read_csv(file_path, sep=";", encoding="utf-8-sig")
        df = df.replace({".": 0, "-": 0})
        df["maennlich"] = pd.to_numeric(df["maennlich"], errors="coerce")
        df["weiblich"] = pd.to_numeric(df["weiblich"], errors="coerce")
        df["Geburtsjahr"] = pd.to_numeric(df["Geburtsjahr"], errors="coerce")
        df["Jahr"] = year
        frames.append(df)
    return pd.concat(frames, ignore_index=True).dropna(subset=["Geburtsjahr"])


@st.cache_data
def load_migration() -> pd.DataFrame:
    df = read_csv("Bevölkerung_mit_Migrationshintergrund_2022_bis_2024.csv")
    df["Jahr"] = pd.to_numeric(df["Jahr"], errors="coerce")
    df["Stadtteil_Nummer"] = pd.to_numeric(df["Stadtteil_Nummer"], errors="coerce")
    df["personen_mit_migrationshintergrund"] = pd.to_numeric(
        df["personen_mit_migrationshintergrund"], errors="coerce"
    )
    df["personen_insgesamt"] = pd.to_numeric(df["personen_insgesamt"], errors="coerce")

    def to_bezirk(value: float) -> int:
        # Stadtteil numbers encode the Bezirk in the first digit(s).
        num = int(value)
        return 10 if num >= 100 else num // 10

    df["Stadtbezirk"] = df["Stadtteil_Nummer"].apply(to_bezirk)
    return df.dropna()


@st.cache_data
def load_forecast() -> pd.DataFrame:
    df = read_csv(
        "Prognose der Bevölkerungsentwicklung auf Ebene der Düsseldorfer Stadtteile bis 01. Januar 2035_0.csv"
    )
    df["2019"] = pd.to_numeric(df["2019"], errors="coerce")
    df["2035"] = pd.to_numeric(df["2035"], errors="coerce")
    df["Veraenderung in %"] = (
        df["Veraenderung in %"].astype(str).str.replace(",", ".", regex=False)
    )
    df["Veraenderung in %"] = pd.to_numeric(df["Veraenderung in %"], errors="coerce")
    return df


@st.cache_data
def load_stadtteile_timeseries() -> pd.DataFrame:
    df = read_csv("Bevölkerungsstand in den Stadtteilen seit 2012_1.csv")
    id_cols = ["Stadtteilnummer", "Stadtteilname"]
    long = df.melt(id_vars=id_cols, var_name="Jahr", value_name="Bev")
    long["Jahr"] = (
        long["Jahr"].str.replace("_", "", regex=False).str.replace("Jahr", "", regex=False)
    )
    long["Jahr"] = pd.to_numeric(long["Jahr"], errors="coerce")
    long["Bev"] = pd.to_numeric(long["Bev"], errors="coerce")
    return long.dropna(subset=["Jahr", "Bev"])


@st.cache_data
def load_centroids() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "stadtbezirk_centroids.csv")
    return df


def forecast_series(df: pd.DataFrame, horizon: list[int]) -> pd.DataFrame:
    # Simple linear trend to extend the latest series by a short horizon.
    x = df["Jahr"].to_numpy()
    y = df["Bevoelkerung"].to_numpy()
    coeffs = np.polyfit(x, y, deg=1)
    forecast = np.polyval(coeffs, np.array(horizon))
    out = pd.DataFrame({"Jahr": horizon, "Bevoelkerung": forecast})
    out["Serie"] = "Forecast"
    return out


# Load data
pop = load_total_population()
residence = load_residence()
birth = load_birth_years()
migration = load_migration()
forecast = load_forecast()
zeitreihe = load_stadtteile_timeseries()
centroids = load_centroids()

st.title("Dusseldorf Demography Dashboard")
st.markdown(
    """
Ein Projekt, das Zeitreihen, Kohortenanalysen und Geografie zusammenzieht.
Alle Visualisierungen sind interaktiv und lassen sich nach Jahr oder Bezirk filtern.
"""
)

st.sidebar.markdown("<div class='block-label'>Filters</div>", unsafe_allow_html=True)
selected_year = st.sidebar.slider("Cohort year", 2018, 2024, 2024)
mig_years = sorted(migration["Jahr"].unique())
selected_mig_year = st.sidebar.selectbox(
    "Migration year", mig_years, index=len(mig_years) - 1
)
stadtteil_options = sorted(zeitreihe["Stadtteilname"].unique())
selected_stadtteil = st.sidebar.selectbox(
    "Stadtteil time series", stadtteil_options, index=stadtteil_options.index("Stadtmitte")
    if "Stadtmitte" in stadtteil_options
    else 0,
)

# KPI calculations
latest_year = int(pop["Jahr"].max())
base_year = int(pop["Jahr"].min())
latest_pop = float(pop.loc[pop["Jahr"] == latest_year, "Bevoelkerung"].iloc[0])
base_pop = float(pop.loc[pop["Jahr"] == base_year, "Bevoelkerung"].iloc[0])
abs_growth = latest_pop - base_pop
pct_growth = abs_growth / base_pop * 100

res_latest = residence.loc[residence["Jahr"] == residence["Jahr"].max()].iloc[0]
neben_share = res_latest["Nebenwohnsitz"] / (
    res_latest["Hauptwohnsitz"] + res_latest["Nebenwohnsitz"]
)

mig_latest = migration[migration["Jahr"] == selected_mig_year].copy()
mig_latest["share"] = (
    mig_latest["personen_mit_migrationshintergrund"] / mig_latest["personen_insgesamt"]
)

st.markdown("<div class='block-label'>Highlights</div>", unsafe_allow_html=True)
col1, col2, col3, col4 = st.columns(4)
col1.metric("Population growth since 2012", f"{abs_growth:,.0f}", f"{pct_growth:.1f}%")
col2.metric("Latest population", f"{latest_pop:,.0f}")
col3.metric("Nebenwohnsitz share", f"{neben_share*100:.1f}%")
col4.metric("Migration share (avg)", f"{mig_latest['share'].mean()*100:.1f}%")

st.divider()

# Population time series with forecast
forecast_years = [2025, 2026]
pop_actual = pop.copy()
pop_actual["Serie"] = "Actual"
forecast_df = forecast_series(pop_actual, forecast_years)

pop_all = pd.concat([pop_actual, forecast_df], ignore_index=True)

fig_pop = px.line(
    pop_all,
    x="Jahr",
    y="Bevoelkerung",
    color="Serie",
    markers=True,
    color_discrete_map={"Actual": "#111827", "Forecast": "#d12f6a"},
)
fig_pop.update_traces(line_width=3)
fig_pop.update_layout(
    title="Total population (2012-2026)",
    yaxis_title="Population",
    legend_title="Series",
)

col_left, col_right = st.columns([2, 1])
col_left.plotly_chart(fig_pop, use_container_width=True)

# Residence split
fig_res = px.area(
    residence,
    x="Jahr",
    y=["Hauptwohnsitz", "Nebenwohnsitz"],
    color_discrete_sequence=["#2b6cb0", "#f59e0b"],
    title="Haupt- vs Nebenwohnsitz",
)
fig_res.update_layout(legend_title="Residence")
col_right.plotly_chart(fig_res, use_container_width=True)

# Extra context: annual deltas
pop_delta = pop.copy()
pop_delta["Delta"] = pop_delta["Bevoelkerung"].diff()
fig_delta = px.bar(
    pop_delta.dropna(),
    x="Jahr",
    y="Delta",
    title="Annual population change",
    color="Delta",
    color_continuous_scale="RdBu",
)
fig_delta.update_layout(coloraxis_showscale=False)
st.plotly_chart(fig_delta, use_container_width=True)

# Residence share over time
res_share = residence.copy()
res_share["Neben_share"] = res_share["Nebenwohnsitz"] / (
    res_share["Hauptwohnsitz"] + res_share["Nebenwohnsitz"]
)
fig_res_share = px.line(
    res_share,
    x="Jahr",
    y="Neben_share",
    markers=True,
    title="Nebenwohnsitz share over time",
)
fig_res_share.update_layout(yaxis_tickformat=".1%")
st.plotly_chart(fig_res_share, use_container_width=True)

st.divider()

# Cohort pyramid
birth_years = birth[birth["Jahr"] == selected_year].copy()
birth_years = birth_years.dropna(subset=["maennlich", "weiblich"])
birth_years["male"] = -birth_years["maennlich"]

fig_pyramid = go.Figure()
fig_pyramid.add_trace(
    go.Bar(
        x=birth_years["male"],
        y=birth_years["Geburtsjahr"],
        orientation="h",
        name="Male",
        marker_color="#2563eb",
    )
)
fig_pyramid.add_trace(
    go.Bar(
        x=birth_years["weiblich"],
        y=birth_years["Geburtsjahr"],
        orientation="h",
        name="Female",
        marker_color="#f97316",
    )
)
fig_pyramid.update_layout(
    title=f"Birth year pyramid ({selected_year})",
    barmode="overlay",
    bargap=0.1,
    xaxis_title="Population",
    yaxis_title="Birth year",
)

col1, col2 = st.columns([1.3, 1])
col1.plotly_chart(fig_pyramid, use_container_width=True)

# Top cohort dynamics
birth_years["total"] = birth_years["maennlich"] + birth_years["weiblich"]
cohort_top = birth_years.sort_values("total", ascending=False).head(10)
fig_top = px.bar(
    cohort_top,
    x="total",
    y="Geburtsjahr",
    orientation="h",
    color="total",
    color_continuous_scale="Blues",
    title="Largest cohorts",
)
fig_top.update_layout(coloraxis_showscale=False)
col2.plotly_chart(fig_top, use_container_width=True)

# Cohort evolution across years
cohort_total = (
    birth.groupby(["Jahr", "Geburtsjahr"], as_index=False)[["maennlich", "weiblich"]]
    .sum()
)
cohort_total["total"] = cohort_total["maennlich"] + cohort_total["weiblich"]
fig_cohort = px.line(
    cohort_total,
    x="Jahr",
    y="total",
    color="Geburtsjahr",
    title="Cohort size evolution (selected birth years)",
)
fig_cohort.update_layout(showlegend=False)
st.plotly_chart(fig_cohort, use_container_width=True)

st.divider()

# Migration background map
mig_year = migration[migration["Jahr"] == selected_mig_year].copy()
mig_year["share"] = (
    mig_year["personen_mit_migrationshintergrund"] / mig_year["personen_insgesamt"]
)

mig_bezirk = (
    mig_year.groupby("Stadtbezirk", as_index=False)[
        ["personen_mit_migrationshintergrund", "personen_insgesamt"]
    ]
    .sum()
)
mig_bezirk["share"] = (
    mig_bezirk["personen_mit_migrationshintergrund"] / mig_bezirk["personen_insgesamt"]
)

mig_map = mig_bezirk.merge(centroids, left_on="Stadtbezirk", right_on="stadtbezirk")

fig_map = px.scatter_mapbox(
    mig_map,
    lat="lat",
    lon="lon",
    size="share",
    color="share",
    hover_name="Stadtbezirk",
    color_continuous_scale="RdYlGn_r",
    size_max=30,
    zoom=10.5,
    title=f"Migration background share by Stadtbezirk ({selected_mig_year})",
)
fig_map.update_layout(mapbox_style="open-street-map", margin=dict(l=0, r=0, t=40, b=0))

fig_mig_rank = px.bar(
    mig_year.sort_values("share", ascending=False).head(15),
    x="share",
    y="Stadtteil_Name",
    orientation="h",
    color="share",
    color_continuous_scale="RdYlGn_r",
    title="Top Stadtteile by migration share",
)
fig_mig_rank.update_layout(coloraxis_showscale=False)

col1, col2 = st.columns([1.2, 1])
col1.plotly_chart(fig_map, use_container_width=True)
col2.plotly_chart(fig_mig_rank, use_container_width=True)

# Migration trend across years (top 5 stadtteile in latest year)
top_stadtteile = (
    mig_year.sort_values("share", ascending=False)["Stadtteil_Name"].head(5).tolist()
)
mig_trend = migration[migration["Stadtteil_Name"].isin(top_stadtteile)].copy()
mig_trend["share"] = (
    mig_trend["personen_mit_migrationshintergrund"] / mig_trend["personen_insgesamt"]
)
fig_mig_trend = px.line(
    mig_trend,
    x="Jahr",
    y="share",
    color="Stadtteil_Name",
    title="Migration background trend (top 5 Stadtteile)",
)
st.plotly_chart(fig_mig_trend, use_container_width=True)

# Distribution of migration shares by year
fig_mig_dist = px.box(
    migration.assign(
        share=migration["personen_mit_migrationshintergrund"]
        / migration["personen_insgesamt"]
    ),
    x="Jahr",
    y="share",
    title="Migration share distribution by year",
)
fig_mig_dist.update_layout(yaxis_tickformat=".1%")
st.plotly_chart(fig_mig_dist, use_container_width=True)

st.divider()

# Forecast to 2035
forecast_st = forecast[forecast["Indikator"].str.match(r"^\d")].copy()
forecast_st[["Code", "Name"]] = forecast_st["Indikator"].str.split(" ", n=1, expand=True)

forecast_top = forecast_st.sort_values("Veraenderung in %", ascending=False).head(15)
forecast_bottom = forecast_st.sort_values("Veraenderung in %", ascending=True).head(10)

fig_forecast = px.bar(
    forecast_top,
    x="Veraenderung in %",
    y="Name",
    orientation="h",
    color="Veraenderung in %",
    color_continuous_scale="Viridis",
    title="Top growth forecast to 2035",
)
fig_forecast.update_layout(coloraxis_showscale=False)

fig_forecast_bottom = px.bar(
    forecast_bottom,
    x="Veraenderung in %",
    y="Name",
    orientation="h",
    color="Veraenderung in %",
    color_continuous_scale="Viridis",
    title="Lowest growth forecast to 2035",
)
fig_forecast_bottom.update_layout(coloraxis_showscale=False)

col1, col2 = st.columns(2)
col1.plotly_chart(fig_forecast, use_container_width=True)
col2.plotly_chart(fig_forecast_bottom, use_container_width=True)

# 2019 vs 2035 scatter to see absolute scale vs growth
forecast_scatter = forecast_st.dropna(subset=["2019", "2035"]).copy()
fig_forecast_scatter = px.scatter(
    forecast_scatter,
    x="2019",
    y="2035",
    color="Veraenderung in %",
    hover_name="Name",
    title="2019 vs 2035 forecast (Stadtteile)",
    color_continuous_scale="Viridis",
)
st.plotly_chart(fig_forecast_scatter, use_container_width=True)

# Stadtteile growth since 2012
stadt_growth = (
    zeitreihe[zeitreihe["Jahr"].isin([2012, 2024])]
    .pivot(index="Stadtteilname", columns="Jahr", values="Bev")
    .dropna()
)

stadt_growth["delta"] = stadt_growth[2024] - stadt_growth[2012]
stadt_growth = stadt_growth.sort_values("delta", ascending=False).head(10)
fig_growth = px.bar(
    stadt_growth,
    x="delta",
    y=stadt_growth.index,
    orientation="h",
    color="delta",
    color_continuous_scale="Blues",
    title="Largest growth since 2012 (top 10 Stadtteile)",
)
fig_growth.update_layout(coloraxis_showscale=False)

st.plotly_chart(fig_growth, use_container_width=True)

# Stadtteil time series spotlight
stadtteil_series = zeitreihe[zeitreihe["Stadtteilname"] == selected_stadtteil].copy()
fig_stadtteil = px.line(
    stadtteil_series,
    x="Jahr",
    y="Bev",
    markers=True,
    title=f"Population trend: {selected_stadtteil}",
)
st.plotly_chart(fig_stadtteil, use_container_width=True)

# Heatmap of Stadtteile vs year (top 20 by 2024 size for readability)
top_2024 = (
    zeitreihe[zeitreihe["Jahr"] == 2024]
    .sort_values("Bev", ascending=False)
    .head(20)["Stadtteilname"]
)
heatmap = (
    zeitreihe[zeitreihe["Stadtteilname"].isin(top_2024)]
    .pivot_table(index="Stadtteilname", columns="Jahr", values="Bev", aggfunc="sum")
    .sort_values(2024, ascending=False)
)
fig_heat = px.imshow(
    heatmap,
    aspect="auto",
    title="Top Stadtteile population heatmap (2012-2024)",
    color_continuous_scale="Blues",
)
st.plotly_chart(fig_heat, use_container_width=True)

st.markdown(
    """
<div class='block-label'>Notes</div>
The migration map uses simplified centroids for Stadtbezirke to highlight relative
spatial patterns (not official boundaries).
""",
    unsafe_allow_html=True,
)
