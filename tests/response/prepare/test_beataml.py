"""Tests for the BeatAML2 viability preparer.

The samples are primary patient specimens, whose native identifier is also the
canonical one, so what matters here is the percent rescaling and how the run,
panel, plate and native replicate columns group wells into series.
"""

import pandas as pd
import pytest
from series_helpers import number
from upath import UPath

from data_curation.response.prepare.beataml import prepare_beataml

RAW_ROW = {
    "dbgap_subject_id": 2100,
    "dbgap_dnaseq_sample": "BA1D",
    "dbgap_rnaseq_sample": "BA1R",
    "inhibitor": "Venetoclax",
    "run_index": 1,
    "inhibitor_panel": "Full05",
    "plate_num": 1,
    "replicate": 1,
    "well_concentration": 0.1,
    "normalized_viability": 85.0,
    "time_of_read": 24.0,
    "passed_qc": True,
}


@pytest.fixture
def beataml_dir(tmp_dir: UPath):
    """Return a factory writing a BeatAML2 raw inhibitor table."""

    def _write(rows: list[dict]) -> UPath:
        pd.DataFrame(rows).to_csv(str(tmp_dir / "beataml_raw_inhibitor.txt"), sep="\t", index=False)
        return tmp_dir

    return _write


def _prepare(raw_dir: UPath) -> pd.DataFrame:
    return prepare_beataml(raw_dir, {}).frame


def _series(raw_dir: UPath) -> pd.DataFrame:
    wells = prepare_beataml(raw_dir, {})
    return number(wells).series


def _doses(**overrides) -> list[dict]:
    return [RAW_ROW | {"well_concentration": dose} | overrides for dose in (0.1, 1.0, 10.0)]


class TestViability:
    def test_percent_viability_is_rescaled_to_a_fraction(self, beataml_dir) -> None:
        """Unlike GDSC/CTRP, BeatAML publishes a percentage with tails above 100."""
        result = _prepare(beataml_dir([RAW_ROW]))
        assert result["intensity"].tolist() == pytest.approx([0.85])

    def test_values_above_one_hundred_percent_are_kept(self, beataml_dir) -> None:
        row = RAW_ROW | {"normalized_viability": 130.0}
        assert _prepare(beataml_dir([row]))["intensity"].tolist() == pytest.approx([1.3])

    def test_negative_viability_is_clipped_at_zero(self, beataml_dir) -> None:
        row = RAW_ROW | {"normalized_viability": -5.0}
        assert _prepare(beataml_dir([row]))["intensity"].tolist() == [0.0]


class TestNativeIdentifiers:
    def test_the_subject_id_is_the_sample(self, beataml_dir) -> None:
        """Primary patient specimens, so the dbGaP subject is the sample."""
        assert set(_series(beataml_dir([RAW_ROW]))["native_sample"]) == {"2100"}

    def test_the_inhibitor_name_is_the_drug(self, beataml_dir) -> None:
        assert set(_series(beataml_dir([RAW_ROW]))["native_drug"]) == {"Venetoclax"}


class TestSeries:
    """One series per (run, panel, plate, native replicate)."""

    def test_an_exactly_duplicated_row_is_one_measurement(self, beataml_dir) -> None:
        assert len(_prepare(beataml_dir([RAW_ROW, RAW_ROW]))) == 1

    def test_two_plates_are_two_technical_replicates(self, beataml_dir) -> None:
        series = _series(beataml_dir([*_doses(), *_doses(plate_num=2)]))
        columns = ["series_key", "n_doses", "biological_replicate", "technical_replicate"]
        assert series[columns].values.tolist() == [["1:Full05:1:1", 3, 0, 0], ["1:Full05:2:1", 3, 0, 1]]

    def test_a_missing_well_does_not_merge_plates(self, beataml_dir) -> None:
        series = _series(beataml_dir([*_doses()[:2], *_doses(plate_num=2)]))
        assert series["n_doses"].tolist() == [2, 3]

    def test_the_native_replicate_separates_series(self, beataml_dir) -> None:
        series = _series(beataml_dir([*_doses(), *_doses(replicate=2)]))
        assert series["replicate_native"].tolist() == ["1", "2"]

    def test_another_run_is_a_biological_replicate(self, beataml_dir) -> None:
        series = _series(beataml_dir([*_doses(), *_doses(run_index=2, dbgap_dnaseq_sample="BA2D")]))
        assert series[["biological_replicate", "technical_replicate"]].values.tolist() == [[0, 0], [1, 0]]

    def test_run_metadata_is_kept_per_series(self, beataml_dir) -> None:
        row = _series(beataml_dir(_doses())).iloc[0]
        assert (row["run_index"], row["inhibitor_panel"], row["plate_num"], row["time_of_read"]) == (
            "1",
            "Full05",
            "1",
            "24",
        )
