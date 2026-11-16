"""One shared plotting style for every figure in the report.

Call apply_style() once at the top of each notebook, and save_fig() for
every figure that the report includes. Palette is colourblind-safe
(seaborn's "colorblind", i.e. the Okabe-Ito-derived set).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

FIGURES = Path(__file__).resolve().parent.parent / "figures"

# Single-figure size used unless a plot genuinely needs more room.
FIGSIZE = (7.0, 4.5)


def apply_style() -> None:
    sns.set_theme(
        style="whitegrid",
        palette="colorblind",
        rc={
            "figure.figsize": FIGSIZE,
            "figure.dpi": 110,
            "axes.titlesize": 0,  # no in-figure titles; captions live in the report
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.bbox": "tight",
        },
    )


def save_fig(fig: plt.Figure, name: str) -> Path:
    """Save as vector PDF under figures/, named after the report figure."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    out = FIGURES / f"{name}.pdf"
    fig.savefig(out)
    return out
