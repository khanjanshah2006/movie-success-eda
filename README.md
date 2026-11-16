# Understanding What Makes a Movie Successful

IT 462 Exploratory Data Analysis, Dhirubhai Ambani University, Autumn 2026.
Supervisor: Dr. Gopinath Panda.

## Reproduce everything

```bash
pip install -r requirements.txt
```

Run the six notebooks in order. Each runs top to bottom on a fresh kernel, loads only from `data/` and imports from `src/`:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/0*.ipynb
```

Then turn the statistics the notebooks wrote into LaTeX macros, check the report, and bundle it:

```bash
python report/build_stats.py
python report/check_report.py report
python report/make_overleaf_zip.py
```

Compile `report/template2026.tex` with pdflatex (run it twice so the cross-references resolve), or upload `report/overleaf_upload.zip` to Overleaf.

## Layout

| Path | Contents |
|---|---|
| `data/raw/` | the two TMDB CSVs and the FRED CPI series, never modified |
| `data/interim/` | `movies_clean.parquet` (notebook 02) |
| `data/processed/` | `movies_features.parquet` (notebook 05) |
| `notebooks/` | `00_complete_project.ipynb` (the whole analysis in one notebook), then 01 audit, 02 cleaning, 03 univariate, 04 multivariate, 05 features, 06 modelling |
| `dashboard/` | `export_data.py` writes the dashboard's data to `dashboard/data/`; `index.html` is the dashboard page |
| `src/` | `load.py`, `clean.py`, `features.py`, `plotting.py`, `reporting.py` |
| `figures/` | every figure in the report, as vector PDF named by figure number (`fig3_6_...` is Figure 3.6) |
| `report/stats/` | JSON statistics written by the notebooks, and the generated `stats.tex` |
| `report/tables/` | LaTeX tables written by the notebooks |

## Rule for the report

No number in `template2026.tex` is typed by hand. Each one is a macro from `report/stats/stats.tex`, for example `\CleanUsableFinancialSample`, and each macro is listed there with the JSON key it came from. To change a number, change the analysis and rerun; never edit `stats.tex` or `tables/` by hand.

## Still to do before submission

- Group number: set `\groupno` near the top of `template2026.tex`.
- Signatures: replace `dau-logo.jpg` in each `\member{...}` call with that member's signature scan.
- Each member should revise the prose of their own chapter, and their short bio, in their own words.
