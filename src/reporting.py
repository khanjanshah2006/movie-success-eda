"""Key statistics and tables quoted in the report are written here, never typed by hand.

Each notebook calls write_stats("chNN_name", {...}) once; report/build_stats.py
turns every JSON file into LaTeX macros, so a rerun that changes a number shows
up as a diff. Tables go through write_table() into report/tables/*.tex and are
\\input{} by the report.
"""

import json
import math
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
STATS = ROOT / "report" / "stats"
TABLES = ROOT / "report" / "tables"


def _jsonable(v):
    # numpy scalars -> plain Python so json can serialise them
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {k: _jsonable(x) for k, x in v.items()}
    return v.item() if hasattr(v, "item") else v


def write_stats(name: str, stats: dict) -> Path:
    STATS.mkdir(parents=True, exist_ok=True)
    out = STATS / f"{name}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({k: _jsonable(v) for k, v in stats.items()}, f, indent=2)
    return out


def read_stats(name: str) -> dict:
    with open(STATS / f"{name}.json", encoding="utf-8") as f:
        return json.load(f)


# ------------------------------------------------------------------ tables

def fmt_p(p: float) -> str:
    """p-value as LaTeX: scientific below 0.001, an explicit bound at underflow."""
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return "--"
    if p == 0:
        return r"$<10^{-300}$"
    if p < 1e-3:
        mant, exp = f"{p:.1e}".split("e")
        return rf"${mant}\times10^{{{int(exp)}}}$"
    return f"{p:.3f}"


def fmt_int(v) -> str:
    return f"{int(round(v)):,}".replace(",", "{,}")


def fmt_money_m(v) -> str:
    """Dollars -> millions with one decimal, e.g. 20000000 -> 20.0."""
    return f"{v / 1e6:,.1f}".replace(",", "{,}")


def _escape(s: str) -> str:
    # values already containing LaTeX math ($...$) or {,} groups are left alone
    if "$" in s or "{,}" in s:
        return s
    for a, b in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
                 ("#", r"\#"), ("_", r"\_"), ("{", r"\{"), ("}", r"\}")]:
        s = s.replace(a, b)
    # a hyphen before a digit is a minus sign: typeset it as one
    return re.sub(r"(?<!\w)-(?=\d)", "$-$", s)


def write_table(df: pd.DataFrame, name: str, align: str | None = None) -> Path:
    """Write df as a booktabs tabular (no float, no caption) to report/tables/.

    Cells should already be formatted strings where formatting matters
    (use fmt_p / fmt_int / fmt_money_m). Everything is escaped except
    cells that already carry LaTeX math. The report supplies the
    table environment and caption.
    """
    TABLES.mkdir(parents=True, exist_ok=True)
    cols = [_escape(str(c)) for c in df.columns]
    if align is None:
        align = "l" + "r" * (len(cols) - 1)
    lines = [rf"\begin{{tabular}}{{{align}}}", r"\toprule", " & ".join(cols) + r" \\", r"\midrule"]
    for _, row in df.iterrows():
        lines.append(" & ".join(_escape(str(v)) for v in row.values) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    out = TABLES / f"{name}.tex"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out
