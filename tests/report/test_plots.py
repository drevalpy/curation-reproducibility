"""Tests for the report's UpSet plots."""

import pandas as pd

from data_curation.report.plots import upset_figure


def test_two_datasets_are_plotted() -> None:
    indicators = pd.DataFrame({"A": [True, True, False], "B": [True, False, True]})
    assert len(upset_figure(indicators, "t", min_subset_size=0).axes) > 1


def test_a_single_dataset_gets_a_note_instead_of_failing() -> None:
    figure = upset_figure(pd.DataFrame({"A": [True, True]}), "Samples")
    assert not figure.axes
    assert "only one dataset" in figure.texts[0].get_text()
