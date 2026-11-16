"""Bundle the report into one zip that compiles as-is on Overleaf (or any TeX install).

    python report/make_overleaf_zip.py   ->   report/overleaf_upload.zip

The zip holds template2026.tex, the logo, stats/stats.tex, tables/*.tex and
figures/*.pdf (placed in figures/, which the report's \\graphicspath also
searches). Rebuild it after rerunning the notebooks and build_stats.py.
"""

import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIGURES = HERE.parent / "figures"
OUT = HERE / "overleaf_upload.zip"

with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
    z.write(HERE / "template2026.tex", "template2026.tex")
    z.write(HERE / "dau-logo.jpg", "dau-logo.jpg")
    z.write(HERE / "stats" / "stats.tex", "stats/stats.tex")
    for p in sorted((HERE / "tables").glob("*.tex")):
        z.write(p, f"tables/{p.name}")
    for p in sorted(FIGURES.glob("*.pdf")):
        z.write(p, f"figures/{p.name}")
    n = len(z.namelist())

print(f"{n} files -> {OUT}")
