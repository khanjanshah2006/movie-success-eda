"""Export the data behind the project dashboard, from the project's own outputs.

    python dashboard/export_data.py

Writes three JSON files to dashboard/data/:
- films.json        one row per film in the financial sample (budget and revenue
                    known, released by 2025), constant-2025 dollars
- models.json       held-out test R^2 for every model and target (notebook 06)
- missingness.json  documented vs undocumented films (notebook 02)

Run the notebooks first; this only reshapes what they produced.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from src.load import load_processed
from src.reporting import read_stats

OUT = ROOT / "dashboard" / "data"
OUT.mkdir(parents=True, exist_ok=True)

df = load_processed("movies_features")
fin = df[df["has_financials"] & ~df["is_future"]].dropna(subset=["Budget_2025usd", "Revenue_2025usd"])

films = []
for _, r in fin.iterrows():
    genres = [g.strip() for g in str(r["Genres"]).split(",") if g.strip() and r["Genres"] == r["Genres"]]
    films.append({
        "title": str(r["Title"]),
        "year": int(r["Release_Year"]),
        "decade": int(r["Release_Year"]) // 10 * 10,
        "month": None if r["release_month"] is None or r["release_month"] != r["release_month"] else int(r["release_month"]),
        "genres": "|".join(genres),
        "budget_2025": round(float(r["Budget_2025usd"])),
        "revenue_2025": round(float(r["Revenue_2025usd"])),
        "roi": round(float(r["roi"]), 4),
        "rating": round(float(r["Audience_Rating"]), 2),
        "english": bool(r["is_english"]),
        "studio_tier": str(r["studio_tier"]),
    })
(OUT / "films.json").write_text(json.dumps(films, ensure_ascii=False), encoding="utf-8")

m = read_stats("ch05_modelling")
models = []
for model, key in [("OLS", "ols"), ("Ridge", "ridge"), ("Gradient-boosted tree", "gb")]:
    models.append({"model": model, "target": "log revenue", "test_r2": m[f"{key}_rev_r2_test"]})
    models.append({"model": model, "target": "log ROI", "test_r2": m[f"{key}_roi_r2_test"]})
for row in models:
    row["n_test"] = m["n_test"]
    row["test_years"] = f"{m['split_year'] + 1}-2025"
(OUT / "models.json").write_text(json.dumps(models), encoding="utf-8")

c = read_stats("ch02_cleaning")
miss = [
    {"measure": "Median release year", "known": c["mar_median_year_doc"], "missing": c["mar_median_year_undoc"], "p": c["mar_p_year"]},
    {"measure": "Median total votes", "known": c["mar_median_votes_doc"], "missing": c["mar_median_votes_undoc"], "p": c["mar_p_votes"]},
    {"measure": "Median popularity", "known": round(c["mar_median_pop_doc"], 2), "missing": round(c["mar_median_pop_undoc"], 2), "p": None},
    {"measure": "Median audience rating", "known": round(c["mar_median_rating_doc"], 2), "missing": round(c["mar_median_rating_undoc"], 2), "p": c["mar_p_rating"]},
    {"measure": "% English-language", "known": c["mar_pct_english_doc"], "missing": c["mar_pct_english_undoc"], "p": c["mar_p_language"]},
]
for row in miss:
    row["n_known"] = c["mar_n_doc"]
    row["n_missing"] = c["mar_n_undoc"]
(OUT / "missingness.json").write_text(json.dumps(miss), encoding="utf-8")

print(f"films.json: {len(films)} rows; models.json: {len(models)}; missingness.json: {len(miss)}")
