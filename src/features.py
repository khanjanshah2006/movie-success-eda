"""Engineered columns, one named function per feature family.

Rules enforced here:

- Director history features use ONLY films released strictly earlier than
  the film in question (and only films present in this dataset) — no
  future information leaks into them.
- Post-release engagement (Total_Votes, Popularity_Score) is never used as
  a predictor downstream: both are outcomes, not attributes a studio knows
  before release. They stay in the frame but are excluded from the model
  feature lists defined at the bottom.
- is_holiday_release uses the conventional release windows (summer
  May-July, winter holidays November-December), stated rather than tuned.
"""

import numpy as np
import pandas as pd

SUMMER_MONTHS = {5, 6, 7}
HOLIDAY_MONTHS = {11, 12}

# Company film-count thresholds for studio tiers (counts over the full
# cleaned dataset; the notebook reports the resulting tier sizes).
MAJOR_MIN_FILMS = 100
MID_MIN_FILMS = 20


# ------------------------------------------------------------- financial

def add_financial_features(df: pd.DataFrame) -> pd.DataFrame:
    """ROI, profit, is_profitable — NaN wherever either financial is unknown.

    ROI is a same-year dollar ratio, so nominal and deflated ROI coincide.
    is_profitable means worldwide gross exceeded production budget; the
    true bar (with marketing) is higher, which the report states.
    """
    df = df.copy()
    budget = df["Budget"].astype("float64")
    revenue = df["Worldwide_Revenue"].astype("float64")
    df["roi"] = revenue / budget
    df["profit"] = revenue - budget
    df["is_profitable"] = pd.array(
        np.where(df["roi"].isna(), pd.NA, df["roi"] > 1.0), dtype="boolean"
    )
    return df


# -------------------------------------------------------------- calendar

def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["release_month"] = df["Release_Date"].dt.month.astype("Int64")
    df["release_quarter"] = df["Release_Date"].dt.quarter.astype("Int64")
    df["release_decade"] = (df["Release_Year"] // 10 * 10).astype("Int64")
    month = df["release_month"]
    df["is_summer_release"] = month.isin(SUMMER_MONTHS).fillna(False)
    df["is_holiday_release"] = month.isin(HOLIDAY_MONTHS).fillna(False)
    return df


# ---------------------------------------------------------------- genres

def genre_list(df: pd.DataFrame) -> pd.Series:
    return df["Genres"].str.split(",").map(
        lambda xs: [x.strip() for x in xs] if isinstance(xs, list) else []
    )


def add_genre_features(df: pd.DataFrame) -> pd.DataFrame:
    """genre_count plus one-hot genre_<name> flags (order-insensitive)."""
    df = df.copy()
    lists = genre_list(df)
    df["genre_count"] = lists.str.len().astype("Int64")
    all_genres = sorted({g for xs in lists for g in xs})
    for gname in all_genres:
        col = "genre_" + gname.lower().replace(" ", "_")
        df[col] = lists.map(lambda xs: gname in xs)
    return df


# -------------------------------------------------------------- language

def add_language_feature(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["is_english"] = df["Original_Language"] == "en"
    return df


# ------------------------------------------------------ director history

def add_director_history(df: pd.DataFrame) -> pd.DataFrame:
    """Prior filmography of the director, within this dataset only.

    For each film: how many earlier films by the same director the dataset
    holds, and their mean audience rating. "Earlier" is strictly by
    release date, so nothing from the film's own release or later leaks
    in. First films (and unknown directors) get count 0 / rating NaN.
    """
    df = df.copy()
    order = df.sort_values("Release_Date", kind="stable")
    grp = order.groupby("Director", dropna=True)
    prior_count = grp.cumcount()
    prior_mean = (
        grp["Audience_Rating"].transform(lambda s: s.expanding().mean().shift(1))
    )
    df["director_prior_film_count"] = prior_count.reindex(df.index).fillna(0).astype("Int64")
    df["director_mean_prior_rating"] = prior_mean.reindex(df.index)
    df.loc[df["Director"].isna(), "director_prior_film_count"] = 0
    return df


# ----------------------------------------------------------- studio tier

def company_counts(df: pd.DataFrame) -> pd.Series:
    return (
        df["Production_Company"].dropna().str.split(",").explode().str.strip().value_counts()
    )


def add_studio_tier(df: pd.DataFrame) -> pd.DataFrame:
    """Tier of the film's best-connected production company.

    Companies are ranked by how many films of the cleaned dataset they
    appear on; a film takes the tier of its most frequent company:
    major (>= MAJOR_MIN_FILMS films), mid (>= MID_MIN_FILMS), else indie.
    Films with no company listed get 'unknown'.
    """
    df = df.copy()
    counts = company_counts(df)

    def tier(companies):
        if not isinstance(companies, str):
            return "unknown"
        best = max(counts.get(c.strip(), 0) for c in companies.split(","))
        if best >= MAJOR_MIN_FILMS:
            return "major"
        if best >= MID_MIN_FILMS:
            return "mid"
        return "indie"

    df["studio_tier"] = df["Production_Company"].map(tier)
    return df


# ------------------------------------------------------------------ text

def add_text_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["overview_length"] = df["Overview"].str.len().astype("Int64")
    return df


# ----------------------------------------------------------- orchestrator

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = add_financial_features(df)
    df = add_calendar_features(df)
    df = add_genre_features(df)
    df = add_language_feature(df)
    df = add_director_history(df)
    df = add_studio_tier(df)
    df = add_text_features(df)
    return df


# Feature lists for Chapter 5. Post-release engagement (Total_Votes,
# Popularity_Score) is deliberately absent: it is an outcome, not a
# pre-release attribute, and would leak the thing being predicted.
# genre_count is deliberately absent: it equals the sum of the one-hot
# genre flags exactly, so the two cannot enter a linear model together
# (verified by an assert in notebook 05). The flags carry more information.
NUMERIC_PREDICTORS = [
    "Budget_2025usd",
    "Runtime_Mins",
    "Release_Year",
    "director_prior_film_count",
    "director_mean_prior_rating",
    "overview_length",
]
BINARY_PREDICTORS = [
    "is_english",
    "is_summer_release",
    "is_holiday_release",
]
CATEGORICAL_PREDICTORS = ["studio_tier", "Content_Rating"]
