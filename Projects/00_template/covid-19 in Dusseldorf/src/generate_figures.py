from __future__ import annotations

from pathlib import Path

from .cleaning import clean_data
from .config import DATA_PATH, FIGURES_DIR, KEY_COLS
from .data_io import load_data
from .plots import (
    ensure_figures_dir,
    plot_corr_heatmap,
    plot_key_timeseries,
    plot_lag_correlations,
    plot_tests_vs_incidence,
)


def main() -> None:
    df = load_data(DATA_PATH)
    df, _ = clean_data(df)

    ensure_figures_dir()

    plot_key_timeseries(df, FIGURES_DIR / "timeseries_key.png")
    plot_tests_vs_incidence(df, FIGURES_DIR / "tests_vs_incidence.png")
    plot_corr_heatmap(df, FIGURES_DIR / "correlations.png")

    pairs = {
        "Tests vs Incidence": ("Abstriche gesamt", "7-Tages-Inzidenz"),
        "Incidence vs Hospital": ("7-Tages-Inzidenz", "Anzahl in Krankenhäusern"),
        "Hospital vs ICU": ("Anzahl in Krankenhäusern", "davon auf Intensivstationen"),
        "Incidence vs Deaths": ("7-Tages-Inzidenz", "Todesfaelle"),
    }
    plot_lag_correlations(df, pairs=pairs, max_lag=28, out_path=FIGURES_DIR / "lag_correlations.png")


if __name__ == "__main__":
    main()
