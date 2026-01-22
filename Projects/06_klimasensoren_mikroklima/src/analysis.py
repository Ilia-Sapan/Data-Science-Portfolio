from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"


@dataclass
class SensorQuality:
    device_id: str
    missing_rate: float
    flatline_rate: float
    noise_score: float
    outlier_rate: float
    trust_score: float


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # The CSV files use semicolons as separators.
    messdaten = pd.read_csv(DATA_DIR / "klimasensoren-messdaten.csv", sep=";")
    aktuell = pd.read_csv(DATA_DIR / "klimasensoren-aktuelle-messungen.csv", sep=";")
    standorte = pd.read_csv(DATA_DIR / "klimasensoren-standorte.csv", sep=";")
    return messdaten, aktuell, standorte


def parse_coordinates(standorte: pd.DataFrame) -> pd.DataFrame:
    coords = standorte["Standort"].str.split(",", expand=True)
    standorte = standorte.copy()
    standorte["lat"] = pd.to_numeric(coords[0], errors="coerce")
    standorte["lon"] = pd.to_numeric(coords[1], errors="coerce")
    return standorte


def coerce_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def coalesce_columns(df: pd.DataFrame, target: str, candidates: list[str]) -> pd.DataFrame:
    available = [col for col in candidates if col in df.columns]
    if not available:
        return df
    df = df.copy()
    df[target] = df[available].bfill(axis=1).iloc[:, 0]
    return df


def build_landuse_proxy(df: pd.DataFrame) -> pd.DataFrame:
    # Heuristic proxy based on street names to approximate green/water/urban influence.
    df = df.copy()
    street_col = next(
        (col for col in ["Stra\u00dfe", "Stra\u00dfe_stand", "Stra\u00dfe_akt"] if col in df.columns),
        None,
    )
    if street_col is None:
        df["landuse_proxy"] = "urban"
        return df
    street = df[street_col].fillna("").str.lower()
    df["landuse_proxy"] = np.select(
        [
            street.str.contains("park|wald|grun|gruen|wiese|garten"),
            street.str.contains("ufer|see|teich|bach|fluss|kanal"),
        ],
        ["green", "water"],
        default="urban",
    )
    return df


def robust_zscore(values: pd.Series) -> pd.Series:
    median = values.median()
    mad = (values - median).abs().median()
    if mad == 0:
        return pd.Series(np.zeros(len(values)), index=values.index)
    return 0.6745 * (values - median) / mad


def detect_anomalies(df: pd.DataFrame, value_col: str, threshold: float = 3.5) -> pd.DataFrame:
    df = df.copy()
    df["zscore"] = df.groupby("Device ID")[value_col].transform(robust_zscore)
    df["is_anomaly"] = df["zscore"].abs() >= threshold
    return df


def quality_metrics(df: pd.DataFrame) -> list[SensorQuality]:
    metrics: list[SensorQuality] = []
    for device_id, group in df.groupby("Device ID"):
        values = group["Lufttemperatur"]
        missing_rate = values.isna().mean()
        flatline_rate = (values.diff().fillna(0) == 0).mean()
        noise_score = values.diff().abs().median(skipna=True) if values.notna().any() else 0
        outlier_rate = group["is_anomaly"].mean()
        trust_score = 100
        trust_score -= missing_rate * 40
        trust_score -= flatline_rate * 30
        trust_score -= outlier_rate * 20
        trust_score -= min(noise_score, 5) * 2
        trust_score = float(max(trust_score, 0))
        metrics.append(
            SensorQuality(
                device_id=device_id,
                missing_rate=missing_rate,
                flatline_rate=flatline_rate,
                noise_score=noise_score,
                outlier_rate=outlier_rate,
                trust_score=trust_score,
            )
        )
    return metrics


def prepare_visuals() -> None:
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        "figure.facecolor": "#f7f3ef",
        "axes.facecolor": "#f7f3ef",
        "savefig.facecolor": "#f7f3ef",
        "axes.edgecolor": "#3b3b3b",
        "grid.color": "#d6cfc7",
        "font.family": "DejaVu Sans",
    })


def plot_temperature_heatmap(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 7))
    hb = ax.hexbin(
        df["lon"],
        df["lat"],
        C=df["Lufttemperatur"],
        gridsize=25,
        cmap="coolwarm",
        mincnt=1,
        linewidths=0,
        alpha=0.9,
    )
    ax.scatter(df["lon"], df["lat"], s=18, color="#2b2b2b", alpha=0.5)
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label("Air temperature (C)")
    ax.set_title("Microclimate heatmap (current measurements)")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "heatmap_temperature.png", dpi=200)
    plt.close(fig)


def plot_plz_temperatures(df: pd.DataFrame) -> None:
    plz = (
        df.groupby("PLZ", dropna=True)["Lufttemperatur"]
        .mean()
        .sort_values()
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(10, 8))
    palette = sns.color_palette("coolwarm", n_colors=len(plz))
    ax.barh(plz["PLZ"].astype(str), plz["Lufttemperatur"], color=palette)
    ax.set_title("Average temperature by postal code")
    ax.set_xlabel("Temperature (C)")
    ax.set_ylabel("Postal code")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "plz_temperature.png", dpi=200)
    plt.close(fig)


def plot_landuse_proxy(df: pd.DataFrame) -> None:
    if df["landuse_proxy"].nunique() < 2:
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.boxplot(
        data=df,
        x="landuse_proxy",
        y="Lufttemperatur",
        hue="landuse_proxy",
        palette="Set2",
        dodge=False,
        ax=ax,
    )
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    ax.set_title("Temperature by land-use proxy")
    ax.set_xlabel("Proxy category")
    ax.set_ylabel("Temperature (C)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "landuse_proxy_temperature.png", dpi=200)
    plt.close(fig)


def plot_anomalies(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    normal = df[~df["is_anomaly"]]
    anomalies = df[df["is_anomaly"]]
    ax.plot(normal["Messdatum"], normal["Lufttemperatur"], "o", color="#2c7fb8", alpha=0.6)
    ax.plot(anomalies["Messdatum"], anomalies["Lufttemperatur"], "o", color="#d95f02", markersize=6)
    ax.set_title("Temperature anomalies (robust z-score)")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Temperature (C)")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "temperature_anomalies.png", dpi=200)
    plt.close(fig)


def plot_quality(metrics: list[SensorQuality]) -> None:
    df = pd.DataFrame([m.__dict__ for m in metrics]).sort_values("trust_score", ascending=False)
    fig, ax = plt.subplots(figsize=(10, 6))
    palette = sns.color_palette("viridis", n_colors=len(df))
    ax.bar(df["device_id"].astype(str), df["trust_score"], color=palette)
    ax.set_title("Sensor trust score (higher is better)")
    ax.set_xlabel("Device ID")
    ax.set_ylabel("Trust score")
    ax.tick_params(axis="x", labelrotation=90)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "sensor_trust_scores.png", dpi=200)
    plt.close(fig)


def main() -> None:
    warnings.filterwarnings("ignore", message="Mean of empty slice")
    prepare_visuals()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    messdaten, aktuell, standorte = load_data()
    standorte = parse_coordinates(standorte)

    numeric_cols = ["Lufttemperatur", "Relative Luftfeuchtigkeit", "Niederschlag", "Luftdruck"]
    messdaten = coerce_numeric(messdaten, numeric_cols)
    aktuell = coerce_numeric(aktuell, numeric_cols)

    messdaten["Messdatum"] = pd.to_datetime(messdaten["Messdatum"], utc=True, errors="coerce")
    aktuell["Messdatum"] = pd.to_datetime(aktuell["Messdatum"], utc=True, errors="coerce")

    aktuell = aktuell.merge(standorte, on="Device ID", how="left", suffixes=("_akt", "_stand"))
    messdaten = messdaten.merge(standorte[["Device ID", "Stra\u00dfe", "PLZ"]], on="Device ID", how="left")
    aktuell = coalesce_columns(aktuell, "PLZ", ["PLZ", "PLZ_stand", "PLZ_akt"])
    aktuell = coalesce_columns(aktuell, "Stra\u00dfe", ["Stra\u00dfe", "Stra\u00dfe_stand", "Stra\u00dfe_akt"])
    messdaten = coalesce_columns(messdaten, "PLZ", ["PLZ", "PLZ_stand", "PLZ_akt"])
    aktuell = build_landuse_proxy(aktuell)

    plot_temperature_heatmap(aktuell.dropna(subset=["lat", "lon", "Lufttemperatur"]))
    plot_plz_temperatures(aktuell.dropna(subset=["PLZ", "Lufttemperatur"]))
    plot_landuse_proxy(aktuell.dropna(subset=["landuse_proxy", "Lufttemperatur"]))

    messdaten = detect_anomalies(messdaten.dropna(subset=["Lufttemperatur"]), "Lufttemperatur")
    plot_anomalies(messdaten.sort_values("Messdatum"))

    metrics = quality_metrics(messdaten)
    plot_quality(metrics)

    quality_df = pd.DataFrame([m.__dict__ for m in metrics]).sort_values("trust_score")
    quality_df.to_csv(REPORTS_DIR / "sensor_quality_summary.csv", index=False)


if __name__ == "__main__":
    main()
