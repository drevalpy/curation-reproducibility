"""Tests for the CCLE viability preparer.

The interesting logic is the activity-to-viability conversion and the unpacking
of the two comma-separated series, both of which are silent corruption risks:
the published values are a percent change in growth on roughly [-100, 0], and a
dose paired with the wrong measurement produces a plausible but wrong curve.
"""

import pandas as pd
import pytest
from series_helpers import number
from upath import UPath

from data_curation.core.errors import CurationError
from data_curation.response.prepare.ccle import SHEET_INDEX, explode_dose_series, prepare_ccle

SHEET_NAMES = [f"Table S{i}" for i in range(SHEET_INDEX + 1)]

ROW = {
    "CCLE Cell Line Name": "1321N1_CENTRAL_NERVOUS_SYSTEM",
    "Primary Cell Line Name": "1321N1",
    "Compound": "17-AAG",
    "Target": "HSP90",
    "Doses (uM)": ".0025,.0080",
    "Activity Data (median)": "0,-100",
    "Activity SD": "43.3,11.5",
    "Num Data": 8,
    "FitType": "Sigmoid",
    "EC50 (µM)": 0.19,
    "IC50 (µM)": 0.23,
    "Amax": -72.1,
    "ActArea": 3.03,
}


@pytest.fixture
def ccle_dir(tmp_dir: UPath):
    """Return a factory writing a stand-in Barretina supplement workbook.

    The real file is a 5.6 MiB legacy ``.xls`` with 13 sheets; the preparer only
    reads sheet 11, so the fixture writes the same sheet index into a small
    ``.xlsx`` and pads the preceding sheets.
    """

    def _write(rows: list[dict]) -> UPath:
        path = tmp_dir / "barretina2012_supplementary_tables.xlsx"
        with pd.ExcelWriter(str(path)) as writer:
            for name in SHEET_NAMES[:-1]:
                pd.DataFrame({"a": [1]}).to_excel(writer, sheet_name=name, index=False)
            # skiprows=2 in the reader, so two filler rows precede the header.
            frame = pd.DataFrame(rows)
            filler = pd.DataFrame([[""] * len(frame.columns)] * 2, columns=frame.columns)
            pd.concat([filler, pd.DataFrame([frame.columns], columns=frame.columns), frame]).to_excel(
                writer, sheet_name=SHEET_NAMES[-1], index=False, header=False
            )
        return tmp_dir

    return _write


def _prepare(raw_dir: UPath) -> pd.DataFrame:
    return prepare_ccle(raw_dir, {}).frame


def _series(raw_dir: UPath) -> pd.DataFrame:
    wells = prepare_ccle(raw_dir, {})
    return number(wells).series


class TestActivityToViability:
    def test_zero_activity_is_full_viability(self, ccle_dir) -> None:
        """0 means the treated well matched the untreated control."""
        result = _prepare(ccle_dir([ROW]))
        assert result["intensity"].iloc[0] == pytest.approx(1.0)

    def test_minus_one_hundred_is_no_viability(self, ccle_dir) -> None:
        result = _prepare(ccle_dir([ROW]))
        assert result["intensity"].iloc[1] == pytest.approx(0.0)

    def test_positive_activity_exceeds_one(self, ccle_dir) -> None:
        """25% of the published values are positive: treated grew faster."""
        row = ROW | {"Activity Data (median)": "50,-50"}
        result = _prepare(ccle_dir([row]))
        assert result["intensity"].iloc[0] == pytest.approx(1.5)

    def test_activity_below_minus_one_hundred_is_clipped(self, ccle_dir) -> None:
        """0.16% of values fall below -100; clipping matches the other preparers."""
        row = ROW | {"Activity Data (median)": "-239,-50"}
        result = _prepare(ccle_dir([row]))
        assert result["intensity"].iloc[0] == 0.0


class TestNativeIdentifiers:
    def test_the_primary_cell_line_name_is_the_sample(self, ccle_dir) -> None:
        """Named rather than accessioned, so it resolves like CTRP after the fit."""
        assert set(_series(ccle_dir([ROW]))["native_sample"]) == {"1321N1"}

    def test_the_compound_name_is_the_drug(self, ccle_dir) -> None:
        assert set(_series(ccle_dir([ROW]))["native_drug"]) == {"17-AAG"}


class TestDoseSeries:
    def test_each_dose_becomes_its_own_row(self, ccle_dir) -> None:
        assert len(_prepare(ccle_dir([ROW]))) == 2

    def test_leading_dot_doses_parse_as_floats(self, ccle_dir) -> None:
        """The workbook stores ".0025", not "0.0025"."""
        assert _prepare(ccle_dir([ROW]))["concentration"].tolist() == pytest.approx([0.0025, 0.008])

    def test_mismatched_series_lengths_are_rejected(self) -> None:
        """A dose silently paired with the wrong measurement is the failure mode."""
        frame = pd.DataFrame([{"Doses (uM)": ".1,.2,.3", "Activity Data (median)": "0,-50"}])
        with pytest.raises(CurationError, match="differently sized activity series"):
            explode_dose_series(frame)

    def test_matching_series_lengths_are_accepted(self) -> None:
        frame = pd.DataFrame([{"Doses (uM)": ".1,.2", "Activity Data (median)": "0,-50"}])
        assert len(explode_dose_series(frame)) == 2

    def test_two_rows_of_the_same_line_stay_separate_drugs(self, ccle_dir) -> None:
        rows = [ROW, ROW | {"Compound": "Erlotinib"}]
        assert set(_series(ccle_dir(rows))["native_drug"]) == {"17-AAG", "Erlotinib"}


class TestSeries:
    def test_every_pair_is_one_series(self, ccle_dir) -> None:
        series = _series(ccle_dir([ROW, ROW | {"Compound": "Erlotinib"}]))
        assert series[["native_drug", "replicate", "n_doses"]].values.tolist() == [
            ["17-AAG", 0, 2],
            ["Erlotinib", 0, 2],
        ]
