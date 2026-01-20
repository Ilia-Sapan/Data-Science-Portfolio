from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .config import DATE_COL, FIGURES_DIR, KEY_COLS


def _setup_style() -> None:
    sns.set_theme(style="whitegrid")


def plot_key_timeseries(df: pd.DataFrame, out_path: Path) -> None:
    _setup_style()
    cols = [c for c in KEY_COLS if c in df.columns]
    n = len(cols)
    rows = (n + 1) // 2
    fig, axes = plt.subplots(rows, 2, figsize=(14, 4 * rows), sharex=True)
    axes = axes.flatten()
    for ax, col in zip(axes, cols):
        ax.plot(df[DATE_COL], df[col], linewidth=1.2)
        ax.set_title(col)
    for ax in axes[len(cols) :]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_corr_heatmap(df: pd.DataFrame, out_path: Path) -> None:
    _setup_style()
    numeric_df = df.select_dtypes(include=["number"])
    corr = numeric_df.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, cmap="coolwarm", center=0, linewidths=0.5)
    plt.title("Correlation Heatmap (Numeric Features)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_tests_vs_incidence(df: pd.DataFrame, out_path: Path) -> None:
    _setup_style()
    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(df[DATE_COL], df["Abstriche gesamt"], color="#1f77b4", label="Abstriche gesamt")
    ax1.set_ylabel("Abstriche gesamt")
    ax2 = ax1.twinx()
    ax2.plot(df[DATE_COL], df["7-Tages-Inzidenz"], color="#d62728", label="7-Tages-Inzidenz")
    ax2.set_ylabel("7-Tages-Inzidenz")
    ax1.set_title("Tests vs 7-Tages-Inzidenz")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_lag_correlations(
    df: pd.DataFrame,
    pairs: Dict[str, str],
    max_lag: int,
    out_path: Path,
) -> None:
    _setup_style()
    lags = range(-max_lag, max_lag + 1)
    fig, ax = plt.subplots(figsize=(12, 5))
    for name, (x_col, y_col) in pairs.items():
        corrs = []
        for lag in lags:
            if lag < 0:
                corr = df[x_col].shift(-lag).corr(df[y_col])
            else:
                corr = df[x_col].corr(df[y_col].shift(lag))
            corrs.append(corr)
        ax.plot(lags, corrs, label=name)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Lag (days)")
    ax.set_ylabel("Correlation")
    ax.set_title("Lag Correlations")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def ensure_figures_dir() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
