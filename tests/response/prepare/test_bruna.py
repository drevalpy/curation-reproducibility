"""Tests for the PDX_Bruna viability preparer.

The samples are PDTCs, whose native identifier is also the canonical one, so what
matters here is the blank-corrected viability arithmetic and that the native
identifiers reach the output untouched.
"""

import pandas as pd
import pytest
from series_helpers import number
from upath import UPath

from data_curation.response.prepare.bruna import prepare_bruna

RAW_COLUMNS = [
    "Model",
    "ID",
    "BARCODE",
    "DRUG_ID",
    *[f"D{i}_CONC" for i in range(1, 6)],
    *[f"D{i}_INTENSITY" for i in range(1, 6)],
    "Control",
    "Blank",
]

RAW_ROW = dict(
    zip(
        RAW_COLUMNS,
        ["STG139", "STG139-T1", 115237, "AZD6244", 0.001, 0.01, 0.1, 1.0, 10.0, 900, 700, 500, 300, 100, 1100, 100],
        strict=True,
    )
)


@pytest.fixture
def bruna_dir(tmp_dir: UPath):
    """Return a factory writing a wide five-dose PDX_Bruna screen."""

    def _write(rows: list[dict]) -> UPath:
        pd.DataFrame(rows)[RAW_COLUMNS].to_csv(str(tmp_dir / "RawDataDrugsSingleAgents.txt"), sep="\t", index=False)
        return tmp_dir

    return _write


def _prepare(raw_dir: UPath) -> pd.DataFrame:
    return prepare_bruna(raw_dir, {}).frame


def _series(raw_dir: UPath) -> pd.DataFrame:
    wells = prepare_bruna(raw_dir, {})
    return number(wells).series


class TestViability:
    def test_the_wide_dose_columns_become_five_rows(self, bruna_dir) -> None:
        assert len(_prepare(bruna_dir([RAW_ROW]))) == 5

    def test_viability_is_blank_corrected_against_the_control(self, bruna_dir) -> None:
        """(intensity - Blank) / (Control - Blank), with the paper's 1 - omitted."""
        result = _prepare(bruna_dir([RAW_ROW])).sort_values("concentration")
        assert result["intensity"].tolist() == pytest.approx([0.8, 0.6, 0.4, 0.2, 0.0])

    def test_viability_is_clipped_at_zero(self, bruna_dir) -> None:
        row = RAW_ROW | {"INTENSITY5": 50}
        assert _prepare(bruna_dir([row]))["intensity"].min() == 0.0

    def test_each_dose_keeps_its_own_concentration(self, bruna_dir) -> None:
        result = _prepare(bruna_dir([RAW_ROW]))
        assert sorted(result["concentration"]) == pytest.approx([0.001, 0.01, 0.1, 1.0, 10.0])


class TestNativeIdentifiers:
    def test_the_pdtc_id_is_the_sample_not_the_model(self, bruna_dir) -> None:
        """ID identifies the PDTC; Model is the originating tumour, kept as metadata."""
        assert set(_series(bruna_dir([RAW_ROW]))["native_sample"]) == {"STG139-T1"}

    def test_the_drug_name_is_the_native_drug_id(self, bruna_dir) -> None:
        assert set(_series(bruna_dir([RAW_ROW]))["native_drug"]) == {"AZD6244"}

    def test_two_pdtcs_of_one_model_stay_distinct_samples(self, bruna_dir) -> None:
        rows = [RAW_ROW, RAW_ROW | {"ID": "STG139-T2", "BARCODE": 115238}]
        assert set(_series(bruna_dir(rows))["native_sample"]) == {"STG139-T1", "STG139-T2"}


class TestSeries:
    def test_two_plates_are_two_technical_replicates(self, bruna_dir) -> None:
        series = _series(bruna_dir([RAW_ROW, RAW_ROW | {"BARCODE": 115238}]))
        columns = ["barcode", "replicate", "n_doses", "biological_replicate", "technical_replicate"]
        assert series[columns].values.tolist() == [["115237", 0, 5, 0, 0], ["115238", 1, 5, 0, 1]]

    def test_a_missing_dose_does_not_merge_plates(self, bruna_dir) -> None:
        rows = [RAW_ROW | {"D5_INTENSITY": None}, RAW_ROW | {"BARCODE": 115238}]
        assert _series(bruna_dir(rows))["n_doses"].tolist() == [4, 5]
