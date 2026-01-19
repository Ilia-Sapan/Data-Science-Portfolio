from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit


@dataclass
class ModelResult:
    model: str
    mae: float
    mape: float


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true != 0
    if mask.sum() == 0:
        return float("nan")
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def _evaluate_split(
    y_true: pd.Series, y_pred: pd.Series, label: str
) -> ModelResult:
    mae = mean_absolute_error(y_true, y_pred)
    mape = _mape(y_true.to_numpy(), y_pred.to_numpy())
    return ModelResult(model=label, mae=mae, mape=mape)


def evaluate_models(
    df: pd.DataFrame,
    target_col: str,
    horizon: int,
    feature_cols: List[str],
    n_splits: int = 5,
) -> List[ModelResult]:
    target = f"{target_col}__t_plus_{horizon}"
    df = df.dropna(subset=[target] + feature_cols).reset_index(drop=True)

    X = df[feature_cols]
    y = df[target]

    tscv = TimeSeriesSplit(n_splits=n_splits)
    results: List[ModelResult] = []

    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        naive_pred = df.iloc[test_idx][f"{target_col}__naive"]
        seasonal_pred = df.iloc[test_idx][f"{target_col}__seasonal_naive"]
        results.append(_evaluate_split(y_test, naive_pred, "Naive"))
        results.append(_evaluate_split(y_test, seasonal_pred, "Seasonal Naive"))

        lr = LinearRegression()
        lr.fit(X_train, y_train)
        lr_pred = pd.Series(lr.predict(X_test), index=y_test.index)
        results.append(_evaluate_split(y_test, lr_pred, "Linear Regression"))

        ridge = Ridge(alpha=1.0)
        ridge.fit(X_train, y_train)
        ridge_pred = pd.Series(ridge.predict(X_test), index=y_test.index)
        results.append(_evaluate_split(y_test, ridge_pred, "Ridge"))

    return results


def summarize_results(results: List[ModelResult]) -> pd.DataFrame:
    df = pd.DataFrame([r.__dict__ for r in results])
    summary = df.groupby("model", as_index=False).agg(
        mae=("mae", "mean"),
        mape=("mape", "mean"),
    )
    return summary.sort_values("mae")
