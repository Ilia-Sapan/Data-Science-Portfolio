from __future__ import annotations

import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pdfplumber
import requests
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
SOURCE_DIR = PROJECT_ROOT.parents[1] / "Wohnungs"

BORIS_BASE = "https://www.gis.nrw.de/arcgis/rest/services/immobilien/boris_nw_bodenrichtwerte_current/MapServer"

TARGET_MUNICIPALITIES = [
    "Düsseldorf",
    "Neuss",
    "Ratingen",
    "Meerbusch",
    "Hilden",
    "Erkrath",
    "Mettmann",
    "Langenfeld (Rheinland)",
    "Monheim am Rhein",
    "Dormagen",
    "Kaarst",
]


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def arcgis_query(layer_id: int, where: str, out_fields: str) -> pd.DataFrame:
    url = f"{BORIS_BASE}/{layer_id}/query"
    all_features = []
    offset = 0
    page_size = 2000
    while True:
        params = {
            "where": where,
            "outFields": out_fields,
            "f": "json",
            "returnGeometry": "false",
            "resultOffset": offset,
            "resultRecordCount": page_size,
        }
        resp = requests.get(url, params=params, timeout=60)
        resp.raise_for_status()
        payload = resp.json()
        features = payload.get("features", [])
        if not features:
            break
        all_features.extend([f["attributes"] for f in features])
        if not payload.get("exceededTransferLimit"):
            break
        offset += page_size
    return pd.DataFrame(all_features)


def fetch_bodenrichtwerte() -> pd.DataFrame:
    frames = []
    fields = "GENA,ORTST,PLZ,BRW,BRWZNR"
    for name in TARGET_MUNICIPALITIES:
        where = f"GENA = '{name}'"
        for layer_id in (2, 5):
            df = arcgis_query(layer_id, where, fields)
            if df.empty:
                continue
            df["layer_id"] = layer_id
            df["gena_filter"] = name
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    data = pd.concat(frames, ignore_index=True)
    data["BRW"] = (
        data["BRW"]
        .astype(str)
        .str.replace(".", "", regex=False)
        .str.replace(",", ".", regex=False)
    )
    data["BRW"] = pd.to_numeric(data["BRW"], errors="coerce")
    data = data.dropna(subset=["BRW"])
    return data


def load_nationality_by_stadtteil() -> pd.DataFrame:
    file_path = next(p for p in SOURCE_DIR.iterdir() if "05-02-08" in p.name)
    raw = pd.read_excel(file_path, sheet_name="2024", header=None)
    header_row = raw.iloc[3].tolist()
    subheader_row = raw.iloc[4].tolist()
    columns = ["gebiet", "ausl_total", "ausl_weiblich"]
    for idx in range(3, 31):
        label = subheader_row[idx]
        if isinstance(label, str) and "darunter" not in label:
            columns.append(label.replace("\n", " "))
        elif label is None and isinstance(header_row[idx], str):
            columns.append(header_row[idx].replace("\n", " "))
        else:
            columns.append(f"col_{idx}")
    table = raw.iloc[5:, :31].copy()
    table.columns = columns
    table = table[table["gebiet"].astype(str).str.match(r"^\d{3}\s")]
    table["stadtteil_code"] = table["gebiet"].str.split().str[0]
    table["stadtteil_name"] = table["gebiet"].str.split().str[1:].str.join(" ")

    nationality_cols = [
        col
        for col in table.columns
        if col
        not in {
            "gebiet",
            "ausl_total",
            "ausl_weiblich",
            "stadtteil_code",
            "stadtteil_name",
        }
        and not col.startswith("col_")
    ]
    table[nationality_cols] = table[nationality_cols].apply(
        pd.to_numeric, errors="coerce"
    )
    table["ausl_total"] = pd.to_numeric(table["ausl_total"], errors="coerce")
    table = table.dropna(subset=["ausl_total"])
    return table[["stadtteil_name", "ausl_total"] + nationality_cols]


def load_foreign_share() -> pd.DataFrame:
    file_path = SOURCE_DIR / "ProzentAusländerStadtteile_seit2013_0.csv"
    df = pd.read_csv(file_path, sep=";", engine="python")
    df = df.rename(columns={"Stadtteilname": "stadtteil_name"})
    df["Jahr2020"] = (
        df["Jahr2020"].astype(str).str.replace(",", ".", regex=False)
    )
    df["Jahr2020"] = pd.to_numeric(df["Jahr2020"], errors="coerce")
    return df[["stadtteil_name", "Jahr2020"]]


def extract_crime_totals() -> pd.DataFrame:
    pdf_files = {
        2021: SOURCE_DIR / "Kriminalstatistik_2021_0.pdf",
        2022: SOURCE_DIR / "pks-2022-jahresbericht_0.pdf",
        2023: SOURCE_DIR / "kriminalstatistik_2023_aktuelle_version.pdf",
        2024: SOURCE_DIR / "d_dokument-pp-dusseldorf-pks-2024.pdf",
    }
    rows = []
    pattern = re.compile(r"insgesamt\s+([\d\.]+)\s+Straftaten", re.IGNORECASE)
    for year, path in pdf_files.items():
        if not path.exists():
            continue
        with pdfplumber.open(path) as pdf:
            text = "\n".join((page.extract_text() or "") for page in pdf.pages[:10])
        match = pattern.search(text)
        if not match:
            continue
        total = float(match.group(1).replace(".", ""))
        rows.append({"year": year, "total_crimes": total})
    df = pd.DataFrame(rows).sort_values("year")
    return df


def linear_forecast(series: pd.Series, years_ahead: int = 2) -> pd.DataFrame:
    x = series.index.values
    y = series.values
    coeffs = np.polyfit(x, y, 1)
    forecast_years = np.arange(x.max() + 1, x.max() + 1 + years_ahead)
    forecast = coeffs[0] * forecast_years + coeffs[1]
    return pd.DataFrame({"year": forecast_years, "value": forecast})


def load_housing_supply() -> pd.DataFrame:
    file_path = SOURCE_DIR / "Wohnungsangebot_3.csv"
    df = pd.read_csv(file_path, sep=";", engine="python")
    cols = [
        "Jahr",
        "Wohnungsbestand  Wohn- und Nichtwohngebaeude ",
        "Bauueberhang Wohnungen",
        "Wohnflaeche je Wohnung in Quadratmeter",
    ]
    df = df[cols].copy()
    df.columns = ["year", "housing_stock", "construction_backlog", "avg_area_sqm"]
    for col in ["housing_stock", "construction_backlog", "avg_area_sqm"]:
        df[col] = (
            df[col].astype(str).str.replace(".", "", regex=False).str.replace(",", ".")
        )
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna()
    return df


def plot_affordability(duesseldorf: pd.DataFrame, surroundings: pd.DataFrame) -> None:
    plt.figure(figsize=(10, 6))
    top_duesseldorf = duesseldorf.nsmallest(12, "median_brw")
    sns.barplot(
        data=top_duesseldorf,
        x="median_brw",
        y="ortst",
        color="#2f6f7e",
    )
    plt.title("Günstigste Stadtteile (Median Bodenrichtwert)")
    plt.xlabel("EUR/m²")
    plt.ylabel("Stadtteil")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "duesseldorf_affordable.png", dpi=180)
    plt.close()

    plt.figure(figsize=(9, 5))
    sns.barplot(
        data=surroundings,
        x="median_brw",
        y="gena",
        color="#c46d3b",
    )
    plt.title("Umland: Median Bodenrichtwert")
    plt.xlabel("EUR/m²")
    plt.ylabel("Kommune")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "surroundings_affordable.png", dpi=180)
    plt.close()


def plot_nationalities(nationality: pd.DataFrame, foreign_share: pd.DataFrame) -> None:
    merged = nationality.merge(foreign_share, on="stadtteil_name", how="left")
    merged = merged.sort_values("ausl_total", ascending=False).head(10)
    cols = [c for c in merged.columns if c not in {"stadtteil_name", "ausl_total", "Jahr2020"}]
    shares = merged[cols].div(merged["ausl_total"], axis=0)
    top3 = shares.apply(lambda row: row.nlargest(3).sum(), axis=1)

    plt.figure(figsize=(10, 6))
    sns.barplot(
        x=top3.values * 100,
        y=merged["stadtteil_name"],
        color="#6b8e23",
    )
    plt.title("Top-3 Nationalitäten Anteil an Ausländerbevölkerung (Top 10 Stadtteile)")
    plt.xlabel("Anteil, %")
    plt.ylabel("Stadtteil")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "nationality_top3_share.png", dpi=180)
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=merged,
        x="Jahr2020",
        y="stadtteil_name",
        color="#5a5f6a",
    )
    plt.title("Ausländeranteil 2020 (Top 10 Stadtteile nach Ausländerzahl)")
    plt.xlabel("Anteil, %")
    plt.ylabel("Stadtteil")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "foreign_share_2020.png", dpi=180)
    plt.close()


def plot_crime_and_forecast(crime: pd.DataFrame) -> pd.DataFrame:
    crime = crime.copy()
    forecast = linear_forecast(crime.set_index("year")["total_crimes"], years_ahead=2)
    plt.figure(figsize=(8, 5))
    plt.plot(crime["year"], crime["total_crimes"], marker="o", label="Historisch")
    plt.plot(forecast["year"], forecast["value"], marker="o", linestyle="--", label="Prognose")
    plt.title("Gesamtkriminalität Düsseldorf (PKS) + Prognose")
    plt.xlabel("Jahr")
    plt.ylabel("Straftaten")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "crime_forecast.png", dpi=180)
    plt.close()
    return forecast


def plot_housing_forecast(housing: pd.DataFrame) -> pd.DataFrame:
    housing = housing.copy()
    forecast = linear_forecast(housing.set_index("year")["construction_backlog"], years_ahead=2)
    plt.figure(figsize=(8, 5))
    plt.plot(housing["year"], housing["construction_backlog"], marker="o", label="Historisch")
    plt.plot(forecast["year"], forecast["value"], marker="o", linestyle="--", label="Prognose")
    plt.title("Bauüberhang Wohnungen + Prognose")
    plt.xlabel("Jahr")
    plt.ylabel("Wohnungen")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "housing_backlog_forecast.png", dpi=180)
    plt.close()
    return forecast


def main() -> None:
    ensure_dirs()

    boris = fetch_bodenrichtwerte()
    if boris.empty:
        raise SystemExit("No BORIS data fetched. Check API access or filters.")
    boris.to_csv(DATA_DIR / "boris_bodenrichtwerte_raw.csv", index=False)

    duesseldorf = boris[boris["GENA"] == "Düsseldorf"].copy()
    duesseldorf_stats = (
        duesseldorf.groupby("ORTST", as_index=False)["BRW"].median().rename(
            columns={"ORTST": "ortst", "BRW": "median_brw"}
        )
    )
    duesseldorf_stats.to_csv(DATA_DIR / "duesseldorf_affordability.csv", index=False)

    surroundings = (
        boris.groupby("GENA", as_index=False)["BRW"].median().rename(
            columns={"GENA": "gena", "BRW": "median_brw"}
        )
    )
    surroundings = surroundings[surroundings["gena"].isin(TARGET_MUNICIPALITIES)]
    surroundings = surroundings.sort_values("median_brw")
    surroundings.to_csv(DATA_DIR / "surroundings_affordability.csv", index=False)

    nationality = load_nationality_by_stadtteil()
    foreign_share = load_foreign_share()
    nationality.to_csv(DATA_DIR / "nationality_by_stadtteil.csv", index=False)
    foreign_share.to_csv(DATA_DIR / "foreign_share_2020.csv", index=False)

    crime = extract_crime_totals()
    crime.to_csv(DATA_DIR / "crime_totals.csv", index=False)

    housing = load_housing_supply()
    housing.to_csv(DATA_DIR / "housing_supply.csv", index=False)

    plot_affordability(duesseldorf_stats, surroundings)
    plot_nationalities(nationality, foreign_share)
    crime_forecast = plot_crime_and_forecast(crime)
    housing_forecast = plot_housing_forecast(housing)

    insights = {
        "affordable_duesseldorf": duesseldorf_stats.nsmallest(5, "median_brw").to_dict(
            orient="records"
        ),
        "affordable_surroundings": surroundings.nsmallest(5, "median_brw").to_dict(
            orient="records"
        ),
        "crime_forecast": crime_forecast.to_dict(orient="records"),
        "housing_backlog_forecast": housing_forecast.to_dict(orient="records"),
    }
    (REPORTS_DIR / "insights.json").write_text(
        json.dumps(insights, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
