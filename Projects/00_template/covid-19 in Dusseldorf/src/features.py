from __future__ import annotations

import pandas as pd

from .config import DATE_COL


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["dow"] = df[DATE_COL].dt.dayofweek
    dow_dummies = pd.get_dummies(df["dow"], prefix="dow", drop_first=False)
    df = pd.concat([df, dow_dummies], axis=1)
    return df


def add_lag_features(df: pd.DataFrame, cols: list[str], lags: list[int]) -> pd.DataFrame:
    df = df.copy()
    for col in cols:
        for lag in lags:
            df[f"{col}__lag_{lag}"] = df[col].shift(lag)
    return df


def add_rolling_features(df: pd.DataFrame, cols: list[str], windows: list[int]) -> pd.DataFrame:
    df = df.copy()
    for col in cols:
        for window in windows:
            df[f"{col}__roll_mean_{window}"] = df[col].rolling(window=window).mean()
    return df


def add_growth_features(df: pd.DataFrame, cols: list[str], periods: list[int]) -> pd.DataFrame:
    df = df.copy()
    for col in cols:
        for period in periods:
            df[f"{col}__pct_change_{period}"] = df[col].pct_change(periods=period)
    return df


def make_supervised(
    df: pd.DataFrame, target_col: str, horizon: int
) -> pd.DataFrame:
    df = df.copy()
    df[f"{target_col}__t_plus_{horizon}"] = df[target_col].shift(-horizon)
    df[f"{target_col}__naive"] = df[target_col]
    df[f"{target_col}__seasonal_naive"] = df[target_col].shift(7)
    return df
