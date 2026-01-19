from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import pandas as pd

from .config import CUMULATIVE_COLS, DATE_COL


@dataclass
class CleaningReport:
    duplicate_dates: int
    missing_dates: int
    non_monotonic: Dict[str, List[str]]


def _coerce_numeric(series: pd.Series) -> pd.Series:
    if series.dtype == object:
        series = (
            series.astype(str)
            .str.replace(".", "", regex=False)
            .str.replace(",", ".", regex=False)
            .replace("nan", pd.NA)
        )
    return pd.to_numeric(series, errors="coerce")


def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
    df = df.copy()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], dayfirst=True, errors="coerce")
    df = df.sort_values(DATE_COL)

    duplicate_dates = df.duplicated(subset=[DATE_COL]).sum()
    df = df.drop_duplicates(subset=[DATE_COL], keep="last")

    for col in df.columns:
        if col == DATE_COL:
            continue
        df[col] = _coerce_numeric(df[col])

    missing_dates = 0
    if df[DATE_COL].notna().any():
        full_range = pd.date_range(df[DATE_COL].min(), df[DATE_COL].max(), freq="D")
        missing_dates = len(full_range.difference(df[DATE_COL]))

    non_monotonic: Dict[str, List[str]] = {}
    for col in CUMULATIVE_COLS:
        if col not in df.columns:
            continue
        diffs = df[col].diff()
        bad = df.loc[diffs < 0, DATE_COL].dt.strftime("%Y-%m-%d").tolist()
        if bad:
            non_monotonic[col] = bad
            df[col] = df[col].cummax()

    df["post_rki_period"] = (df[DATE_COL] >= pd.Timestamp("2021-02-05")).astype(int)

    report = CleaningReport(
        duplicate_dates=int(duplicate_dates),
        missing_dates=int(missing_dates),
        non_monotonic=non_monotonic,
    )
    return df, report
