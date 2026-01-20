from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
FIG_DIR = BASE_DIR / "reports" / "figures"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

MONTH_MAP = {
    "januar": 1,
    "februar": 2,
    "maerz": 3,
    "marz": 3,
    "april": 4,
    "mai": 5,
    "juni": 6,
    "juli": 7,
    "august": 8,
    "september": 9,
    "oktober": 10,
    "november": 11,
    "dezember": 12,
}

MONTH_LABELS = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
]


def normalize_month(value: str) -> int:
    # Normalize German month names to a numeric month.
    raw = str(value).strip().lower()
    raw = raw.replace("ä", "ae").replace("ö", "oe").replace("ü", "ue").replace("ß", "ss")
    raw = raw.replace(" ", "")
    ascii_only = re.sub(r"[^a-z]", "", raw)
    if ascii_only in MONTH_MAP:
        return MONTH_MAP[ascii_only]
    if ascii_only == "mrz":
        return 3
    raise ValueError(f"Unknown month value: {value}")


def clean_int(value: str) -> int:
    # Convert numeric strings with spaces/dashes into integers.
    if pd.isna(value):
        return 0
    text = str(value).strip()
    if text in {"-", "–", ""}:
        return 0
    text = text.replace(" ", "")
    return int(float(text))


def load_monthly_data() -> pd.DataFrame:
    # Read all raw CSVs and return a normalized monthly table.
    frames = []
    for path in sorted(RAW_DIR.glob("Verungl*csv")):
        match = re.search(r"(\d{4})", path.name)
        if not match:
            continue
        year = int(match.group(1))
        df = pd.read_csv(path, sep=";", encoding="latin-1")
        df.columns = [col.strip() for col in df.columns]
        df["year"] = year
        df["month_name"] = df["Monat"].astype(str).str.strip()
        df["month"] = df["Monat"].apply(normalize_month)
        for col in [
            "Personen_insgesamt",
            "Leichtverletzte",
            "Schwerverletzte",
            "Getoetete",
        ]:
            df[col] = df[col].apply(clean_int)
        frames.append(df)

    if not frames:
        raise RuntimeError("No input files found in data/raw")

    result = pd.concat(frames, ignore_index=True)
    result = result.sort_values(["year", "month"]).reset_index(drop=True)
    return result


def summarize(monthly: pd.DataFrame) -> pd.DataFrame:
    stats = monthly[[
        "Personen_insgesamt",
        "Leichtverletzte",
        "Schwerverletzte",
        "Getoetete",
    ]].describe().round(2)
    return stats


def forecast_monthly(monthly: pd.DataFrame, years: list[int]) -> pd.DataFrame:
    # Fit a linear trend for each month separately and forecast future years.
    metrics = [
        "Personen_insgesamt",
        "Leichtverletzte",
        "Schwerverletzte",
        "Getoetete",
    ]
    rows = []
    for month in range(1, 13):
        subset = monthly[monthly["month"] == month]
        x = subset["year"].values
        for target_year in years:
            row = {"year": target_year, "month": month}
            for metric in metrics:
                y = subset[metric].values
                coef = np.polyfit(x, y, 1)
                pred = coef[0] * target_year + coef[1]
                row[metric] = max(0, int(round(pred)))
            rows.append(row)
    forecast = pd.DataFrame(rows)
    return forecast


def plot_monthly_trends(monthly: pd.DataFrame) -> None:
    pivot = monthly.pivot(index="month", columns="year", values="Personen_insgesamt")
    plt.figure(figsize=(10, 5))
    for year in pivot.columns:
        plt.plot(pivot.index, pivot[year], marker="o", label=str(year))
    plt.xticks(range(1, 13), MONTH_LABELS)
    plt.title("Monthly accident victims by year")
    plt.xlabel("Month")
    plt.ylabel("Total persons")
    plt.legend(ncol=3, fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "monthly_trends.png", dpi=160)
    plt.close()


def plot_yearly_totals(yearly: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 4.5))
    plt.bar(yearly["year"].astype(str), yearly["Personen_insgesamt"], color="#4C78A8")
    plt.title("Yearly total accident victims")
    plt.xlabel("Year")
    plt.ylabel("Total persons")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "yearly_totals.png", dpi=160)
    plt.close()


def plot_forecast(yearly: pd.DataFrame, forecast_yearly: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 4.5))
    plt.plot(yearly["year"], yearly["Personen_insgesamt"], marker="o", label="History")
    plt.plot(
        forecast_yearly["year"],
        forecast_yearly["Personen_insgesamt"],
        marker="o",
        linestyle="--",
        label="Forecast",
    )
    plt.title("Forecast of total accident victims")
    plt.xlabel("Year")
    plt.ylabel("Total persons")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "forecast_2025_2026.png", dpi=160)
    plt.close()


def main() -> None:
    # Load and normalize monthly data across all years.
    monthly = load_monthly_data()
    monthly.to_csv(PROCESSED_DIR / "monthly.csv", index=False)

    # Aggregate to yearly totals for reporting.
    yearly = (
        monthly.groupby("year", as_index=False)[
            ["Personen_insgesamt", "Leichtverletzte", "Schwerverletzte", "Getoetete"]
        ]
        .sum()
        .sort_values("year")
    )
    yearly.to_csv(PROCESSED_DIR / "yearly.csv", index=False)

    # Summary statistics for quick inspection.
    summary = summarize(monthly)
    summary.to_csv(PROCESSED_DIR / "summary_stats.csv")

    # Simple per-month linear trend forecast.
    forecast = forecast_monthly(monthly, years=[2025, 2026])
    forecast = forecast.sort_values(["year", "month"]).reset_index(drop=True)
    forecast.to_csv(PROCESSED_DIR / "forecast_2025_2026.csv", index=False)

    # Aggregate forecast to yearly totals for plotting.
    forecast_yearly = (
        forecast.groupby("year", as_index=False)[
            ["Personen_insgesamt", "Leichtverletzte", "Schwerverletzte", "Getoetete"]
        ]
        .sum()
        .sort_values("year")
    )

    plot_monthly_trends(monthly)
    plot_yearly_totals(yearly)
    plot_forecast(yearly, forecast_yearly)


if __name__ == "__main__":
    main()
