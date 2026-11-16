"""Readers for the two raw TMDB files.

Every notebook loads data through these functions so that dtypes, file
paths and the zero-as-missing rule are applied in exactly one place.
"""

from pathlib import Path

import numpy as np
import pandas as pd

# Repo root = parent of src/, regardless of where the caller runs from.
ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"
FIGURES = ROOT / "figures"
STATS = ROOT / "report" / "stats"

PRIMARY_CSV = RAW / "movies_eda_checkpoint_2.csv"
SECONDARY_CSV = RAW / "top10K-TMDB-movies.csv"

# Columns in which TMDB stores "unknown" as 0.
ZERO_AS_MISSING = ["Budget", "Worldwide_Revenue"]


def load_primary(zero_to_nan: bool = True) -> pd.DataFrame:
    """Primary scraped dataset (10,000 x 17).

    zero_to_nan=False returns the file as stored, for before/after
    comparisons in the cleaning chapter.
    """
    df = pd.read_csv(PRIMARY_CSV)
    # Dates are M/D/YYYY, except two 1890s films stored as ISO YYYY-MM-DD.
    parsed = pd.to_datetime(df["Release_Date"], format="%m/%d/%Y", errors="coerce")
    iso_fallback = pd.to_datetime(df["Release_Date"], format="%Y-%m-%d", errors="coerce")
    df["Release_Date"] = parsed.fillna(iso_fallback)
    df["Release_Year"] = df["Release_Year"].astype("Int64")
    if zero_to_nan:
        for col in ZERO_AS_MISSING:
            df[col] = df[col].astype("float64").replace(0.0, np.nan)
    return df


def load_secondary() -> pd.DataFrame:
    """Secondary Kaggle dataset (10,000 x 9). No financial columns."""
    df = pd.read_csv(SECONDARY_CSV)
    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    return df


def load_interim(name: str) -> pd.DataFrame:
    return pd.read_parquet(INTERIM / f"{name}.parquet")


def load_processed(name: str) -> pd.DataFrame:
    return pd.read_parquet(PROCESSED / f"{name}.parquet")
