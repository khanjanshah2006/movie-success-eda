"""Every cleaning rule as a named function.

Each function does one documented thing and returns the modified frame plus
the count of affected rows/values, so the notebook can show a before/after
for every rule and build the reconciliation table without silent drops.

Decisions these rules encode (justified in Chapter 2 of the report):

- Zeros in Budget / Worldwide_Revenue are TMDB's encoding of "unknown" and
  become NaN (handled in src.load, counted here).
- Financial values from $1 to $999 are entry errors or thousands-scale
  figures; they become NaN rather than dropping the whole row.
- Exact duplicate rows are dropped. Films sharing title+year but differing
  in director/runtime are DIFFERENT films and are kept — verified manually
  (e.g. two 2011 "War of the Buttons", two 2026 "The Odyssey").
- Films dated after 2025 are flagged, not dropped: their revenue is
  incomplete, so they are excluded from revenue/ROI analysis only.
"""

import numpy as np
import pandas as pd

FINANCIAL_FLOOR = 1_000
FUTURE_YEAR = 2025  # strictly greater than this = incomplete revenue
CPI_BASE_YEAR = 2025

COUNTRY_RENAMES = {
    "United States of America": "United States",
}


# ---------------------------------------------------------------- duplicates

def drop_exact_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Drop rows identical in every column, keeping the first."""
    n = int(df.duplicated().sum())
    return df.drop_duplicates().reset_index(drop=True), n


# ---------------------------------------------------------------- bad values

def floor_financials(df: pd.DataFrame, floor: int = FINANCIAL_FLOOR) -> tuple[pd.DataFrame, dict]:
    """Set budgets/revenues in (0, floor) to NaN — entry errors, not money."""
    df = df.copy()
    counts = {}
    for col in ["Budget", "Worldwide_Revenue"]:
        mask = df[col].between(1, floor - 1)
        counts[col] = int(mask.sum())
        df.loc[mask, col] = np.nan
    return df, counts


def clean_runtime(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """A zero runtime is impossible for a released film -> NaN."""
    df = df.copy()
    mask = df["Runtime_Mins"] == 0
    df["Runtime_Mins"] = df["Runtime_Mins"].astype("float64")
    df.loc[mask, "Runtime_Mins"] = np.nan
    return df, int(mask.sum())


def fix_director_placeholder(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """The literal string 'Name' is a scraping placeholder, not a person."""
    df = df.copy()
    mask = df["Director"] == "Name"
    df.loc[mask, "Director"] = np.nan
    return df, int(mask.sum())


# ------------------------------------------------------------- categoricals

def clean_content_rating(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Strip whitespace (fixes 'PG-13 ') and merge NR into Unrated."""
    df = df.copy()
    before = df["Content_Rating"].copy()
    df["Content_Rating"] = df["Content_Rating"].str.strip().replace({"NR": "Unrated"})
    counts = {
        "whitespace_fixed": int((before != before.str.strip()).sum()),
        "nr_merged": int((before.str.strip() == "NR").sum()),
    }
    return df, counts


def normalise_countries(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Consistent country names inside the multi-label string."""
    df = df.copy()
    affected = int(
        df["Production_Country"].str.contains("United States of America", na=False).sum()
    )
    for old, new in COUNTRY_RENAMES.items():
        df["Production_Country"] = df["Production_Country"].str.replace(old, new, regex=False)
    return df, affected


# ------------------------------------------------------------------- flags

def flag_future(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """is_future=True -> excluded from all revenue and ROI analysis."""
    df = df.copy()
    df["is_future"] = (df["Release_Year"] > FUTURE_YEAR).fillna(False)
    return df, int(df["is_future"].sum())


def flag_has_financials(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """has_financials=True -> budget and revenue both known and plausible."""
    df = df.copy()
    df["has_financials"] = df["Budget"].notna() & df["Worldwide_Revenue"].notna()
    return df, int(df["has_financials"].sum())


# -------------------------------------------------------- secondary dataset

def norm_title(s: pd.Series) -> pd.Series:
    return s.str.strip().str.casefold()


def attach_secondary(df: pd.DataFrame, df2: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Join the Kaggle file on (normalised title, release year).

    Recovers the TMDB id as a stable key, brings in vote_average /
    popularity for cross-source validation, and fills the handful of
    missing Genres / Overview values. Nothing else crosses over.
    """
    df = df.copy()
    sec = df2.copy()
    sec["_key_title"] = norm_title(sec["title"])
    sec["_key_year"] = sec["release_date"].dt.year.astype("Int64")
    sec = sec.drop_duplicates(subset=["_key_title", "_key_year"])
    sec_cols = sec[
        ["_key_title", "_key_year", "id", "vote_average", "popularity", "genre", "overview"]
    ].rename(
        columns={
            "id": "tmdb_id",
            "vote_average": "sec_vote_average",
            "popularity": "sec_popularity",
            "genre": "sec_genre",
            "overview": "sec_overview",
        }
    )

    df["_key_title"] = norm_title(df["Title"])
    df["_key_year"] = df["Release_Year"]
    df = df.merge(sec_cols, on=["_key_title", "_key_year"], how="left")

    counts = {"matched": int(df["tmdb_id"].notna().sum())}

    # Fill missing genres/overviews from the secondary file; its genre
    # strings have no space after the comma, so normalise the separator.
    genre_fill = df["Genres"].isna() & df["sec_genre"].notna()
    df.loc[genre_fill, "Genres"] = df.loc[genre_fill, "sec_genre"].str.replace(",", ", ", regex=False)
    overview_fill = df["Overview"].isna() & df["sec_overview"].notna()
    df.loc[overview_fill, "Overview"] = df.loc[overview_fill, "sec_overview"]
    counts["genres_filled"] = int(genre_fill.sum())
    counts["overviews_filled"] = int(overview_fill.sum())

    df = df.drop(columns=["_key_title", "_key_year", "sec_genre", "sec_overview"])
    return df, counts


# ---------------------------------------------------------------- inflation

def cpi_annual(path) -> pd.Series:
    """Annual mean CPI-U (NSA) indexed by year, from the committed FRED file."""
    cpi = pd.read_csv(path)
    cpi.columns = ["date", "cpi"]
    cpi["date"] = pd.to_datetime(cpi["date"])
    annual = cpi.set_index("date")["cpi"].groupby(lambda d: d.year).mean()
    annual.index.name = "year"
    return annual


def deflate(values: pd.Series, years: pd.Series, cpi: pd.Series, base_year: int = CPI_BASE_YEAR) -> pd.Series:
    """Convert nominal dollars to constant base-year dollars.

    CPI-U starts in 1913; years before that (a handful of silent films,
    none with usable financials) get NaN rather than a made-up factor.
    Future years (2026) also get NaN — there is no CPI for them yet.
    """
    base = cpi.loc[base_year]
    factors = years.map(cpi).astype("float64")
    # Years past the base have only partial CPI data and are excluded
    # from financial analysis anyway (is_future) — no deflated value.
    factors[years.astype("float64") > base_year] = np.nan
    return values * (base / factors)
