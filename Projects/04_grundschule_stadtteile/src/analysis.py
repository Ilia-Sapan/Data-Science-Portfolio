from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
FIG_DIR = BASE_DIR / "reports" / "figures"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_data() -> pd.DataFrame:
    # Load the single raw CSV and normalize column names.
    path = next(RAW_DIR.glob("Grundschule_*csv"))
    df = pd.read_csv(path, sep=";", encoding="utf-8")
    df.columns = ["stadtteil", "maennlich", "weiblich", "deutsch", "nichtdeutsch"]
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    # Split district code and name, and ensure numeric columns are integers.
    parts = df["stadtteil"].str.strip().str.extract(r"^(\\d+)\\s+(.*)$")
    df["district_code"] = pd.to_numeric(parts[0], errors="coerce")
    df["district_name"] = parts[1].fillna(df["stadtteil"].str.strip())

    for col in ["maennlich", "weiblich", "deutsch", "nichtdeutsch"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    df["total_students"] = df["maennlich"] + df["weiblich"]
    df["male_share"] = (df["maennlich"] / df["total_students"]).fillna(0)
    df["non_german_share"] = (df["nichtdeutsch"] / df["total_students"]).fillna(0)
    return df


def save_summary(df: pd.DataFrame) -> None:
    summary = df[["total_students", "maennlich", "weiblich", "deutsch", "nichtdeutsch"]]
    summary.describe().round(2).to_csv(PROCESSED_DIR / "summary_stats.csv")


def plot_top_districts(df: pd.DataFrame) -> None:
    top = df.sort_values("total_students", ascending=False).head(10)
    plt.figure(figsize=(9, 5))
    plt.barh(top["district_name"], top["total_students"], color="#3B6EA5")
    plt.title("Top 10 districts by total students")
    plt.xlabel("Students")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "top_districts_total.png", dpi=160)
    plt.close()


def plot_gender_split(df: pd.DataFrame) -> None:
    top = df.sort_values("total_students", ascending=False).head(10)
    plt.figure(figsize=(9, 5))
    plt.barh(top["district_name"], top["maennlich"], color="#4C78A8", label="Male")
    plt.barh(
        top["district_name"],
        top["weiblich"],
        left=top["maennlich"],
        color="#F58518",
        label="Female",
    )
    plt.title("Gender split in top 10 districts")
    plt.xlabel("Students")
    plt.gca().invert_yaxis()
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "top_districts_gender_split.png", dpi=160)
    plt.close()


def plot_non_german_share(df: pd.DataFrame) -> None:
    top = df.sort_values("non_german_share", ascending=False).head(10)
    plt.figure(figsize=(9, 5))
    plt.barh(top["district_name"], top["non_german_share"] * 100, color="#54A24B")
    plt.title("Top 10 districts by non-German share")
    plt.xlabel("Share (%)")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "top_districts_non_german_share.png", dpi=160)
    plt.close()


def main() -> None:
    df = load_data()
    df = clean_data(df)

    df.to_csv(PROCESSED_DIR / "cleaned.csv", index=False)
    save_summary(df)

    top = df.sort_values("total_students", ascending=False).head(10)
    top.to_csv(PROCESSED_DIR / "top_districts.csv", index=False)

    plot_top_districts(df)
    plot_gender_split(df)
    plot_non_german_share(df)


if __name__ == "__main__":
    main()
