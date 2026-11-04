"""Static UpSet plots of dataset overlap for the report.

Kept separate from the notebook so the sample and drug plots share one
implementation, including the workaround for upsetplot's count labels.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402 - the backend must be chosen first
import pandas as pd  # noqa: E402
import upsetplot  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402


def upset_figure(indicators: pd.DataFrame, title: str, min_subset_size: int = 10) -> Figure:
    """Draw an UpSet plot from a boolean entity x dataset table.

    Args:
        indicators: One row per entity (sample, drug), one boolean column per
            dataset, as returned by ``summary.dataset_indicators``.
        title: Figure title.
        min_subset_size: Intersections smaller than this are hidden.

    Returns:
        The rendered figure; a note instead of a plot when there is only one
        dataset, which has no overlap to show (upsetplot needs two or more).
    """
    if indicators.shape[1] < 2:
        fig = plt.figure(figsize=(6, 1))
        fig.text(0.5, 0.5, f"{title}: only one dataset, no overlap to show", ha="center", va="center")
        return fig
    fig = plt.figure(figsize=(14, 7))
    axes = upsetplot.UpSet(
        upsetplot.from_indicators(indicators),
        subset_size="count",
        sort_by="cardinality",
        sort_categories_by="-cardinality",
        min_subset_size=min_subset_size,
        facecolor="#1f77b4",
    ).plot(fig=fig)
    # upsetplot 0.9's own show_counts labels crash on current matplotlib, so the
    # bar counts are added with matplotlib's bar_label instead.
    for key in ("intersections", "totals"):
        for bars in axes[key].containers:
            axes[key].bar_label(bars, fmt="%d", fontsize=8, padding=2)
    fig.suptitle(title)
    return fig
