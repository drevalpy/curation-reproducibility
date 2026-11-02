"""Tests for the GDSC plate-level viability preparer."""

import pandas as pd
import pytest
from series_helpers import number
from upath import UPath

from data_curation.response.prepare.gdsc import prepare_gdsc

DS_PARAMS = {"negative_control_tag": "NC-1", "positive_control_tag": "B"}

RAW_COLUMNS = [
    "RESEARCH_PROJECT",
    "BARCODE",
    "SCAN_ID",
    "DATE_CREATED",
    "SCAN_DATE",
    "CELL_ID",
    "MASTER_CELL_ID",
    "COSMIC_ID",
    "CELL_LINE_NAME",
    "SANGER_MODEL_ID",
    "SEEDING_DENSITY",
    "DRUGSET_ID",
    "ASSAY",
    "DURATION",
    "POSITION",
    "TAG",
    "DRUG_ID",
    "CONC",
    "INTENSITY",
]


def _well(tag: str, intensity: float, **overrides) -> dict:
    base = {
        "RESEARCH_PROJECT": "GDSC_SA",
        "BARCODE": 100,
        "SCAN_ID": 1,
        "DATE_CREATED": "2015-02-13T00:00:00Z",
        "SCAN_DATE": "2015-02-17T00:00:00Z",
        "MASTER_CELL_ID": 1,
        "COSMIC_ID": 1,
        "CELL_LINE_NAME": "A",
        "SEEDING_DENSITY": 250,
        "DRUGSET_ID": 159,
        "ASSAY": "Glo",
        "DURATION": 4,
        "POSITION": 1,
        "TAG": tag,
        "DRUG_ID": None,
        "CONC": None,
        "INTENSITY": intensity,
        "SANGER_MODEL_ID": "SIDM00001",
        "CELL_ID": 1,
    }
    return base | overrides


@pytest.fixture
def write_raw(tmp_dir: UPath):
    """Return a factory writing a GDSC raw well table into the raw directory."""

    def _write(wells: list[dict]) -> UPath:
        frame = pd.DataFrame(wells)[RAW_COLUMNS]
        frame.to_csv(str(tmp_dir / "GDSC2_public_raw_data.csv"), index=False)
        return tmp_dir

    return _write


#: One plate: blank at 0, untreated at 100, one library well at 40.
PLATE = [
    _well("B", 0.0),
    _well("NC-1", 100.0),
    _well("L1-D1-S", 40.0, POSITION=2, DRUG_ID=1001, CONC=0.5),
]


def _prepare(raw_dir: UPath, params: dict = DS_PARAMS) -> pd.DataFrame:
    return prepare_gdsc(raw_dir, params).frame


def _series(raw_dir: UPath) -> pd.DataFrame:
    wells = prepare_gdsc(raw_dir, DS_PARAMS)
    return number(wells).series


def _replicates(raw_dir: UPath) -> list[list]:
    """``[series_key, biological_replicate, technical_replicate]`` per series, in replicate order."""
    return _series(raw_dir)[["series_key", "biological_replicate", "technical_replicate"]].values.tolist()


class TestViability:
    def test_wells_are_normalized_between_the_controls(self, write_raw) -> None:
        result = _prepare(write_raw(PLATE))
        assert result["intensity"].tolist() == pytest.approx([0.4])

    def test_viability_is_clipped_at_zero(self, write_raw) -> None:
        wells = [*PLATE[:2], _well("L1-D1-S", -20.0, POSITION=2, DRUG_ID=1001, CONC=0.5)]
        assert _prepare(write_raw(wells))["intensity"].tolist() == [0.0]

    def test_the_dose_comes_from_conc(self, write_raw) -> None:
        assert _prepare(write_raw(PLATE))["concentration"].tolist() == [0.5]


class TestNativeIdentifiers:
    def test_the_sanger_model_id_is_the_sample(self, write_raw) -> None:
        """Native, not the CVCL accession: identity is resolved after the fit."""
        assert _series(write_raw(PLATE))["native_sample"].tolist() == ["SIDM00001"]

    def test_the_drug_id_is_the_drug(self, write_raw) -> None:
        assert _series(write_raw(PLATE))["native_drug"].tolist() == ["1001"]


class TestWellSelection:
    def test_control_wells_are_not_measurements(self, write_raw) -> None:
        assert len(_prepare(write_raw(PLATE))) == 1

    def test_anchor_and_test_arms_are_excluded(self, write_raw) -> None:
        """Only the L* library arm becomes a dose-response measurement."""
        wells = [*PLATE, _well("R1-D1-S", 50.0, POSITION=3, DRUG_ID=1002, CONC=0.5)]
        assert _prepare(write_raw(wells))["DRUG_ID"].tolist() == [1001]

    def test_a_failed_position_is_dropped_entirely(self, write_raw) -> None:
        """UN-USED marks a failed drug; its whole (SCAN_ID, POSITION) is unusable."""
        wells = [*PLATE, _well("UN-USED", 0.0, POSITION=2)]
        assert _prepare(write_raw(wells)).empty

    def test_a_treatment_well_without_a_drug_is_dropped(self, write_raw) -> None:
        wells = [*PLATE[:2], _well("L1-D1-S", 40.0, POSITION=2, CONC=0.5)]
        assert _prepare(write_raw(wells)).empty

    def test_the_negative_control_tag_is_read_from_params(self, write_raw) -> None:
        """GDSC1 tags its untreated wells NC-0, GDSC2 NC-1."""
        wells = [_well("B", 0.0), _well("NC-0", 100.0), PLATE[2]]
        params = {"negative_control_tag": "NC-0", "positive_control_tag": "B"}
        assert _prepare(write_raw(wells), params)["intensity"].tolist() == pytest.approx([0.4])


def _plate(
    barcode: int, scan_id: int, doses: list[float], *, slot: str = "L1", cell_id: int = 1, **overrides
) -> list[dict]:
    """Controls plus one library series of drug 1001 on one plate."""
    plate = {"BARCODE": barcode, "SCAN_ID": scan_id, "CELL_ID": cell_id, **overrides}
    wells = [_well("B", 0.0, **plate), _well("NC-1", 100.0, **plate)]
    for position, dose in enumerate(doses, start=10):
        tag = f"{slot}-D{position}-S"
        wells.append(_well(tag, 50.0, POSITION=position, DRUG_ID=1001, CONC=dose, **plate))
    return wells


DOSES = [0.1, 0.5, 2.5]


class TestSeries:
    """One series per (plate, library slot)."""

    def test_two_plates_are_two_replicates(self, write_raw) -> None:
        series = _series(write_raw([*_plate(100, 1, DOSES), *_plate(200, 2, DOSES)]))
        assert series[["series_key", "replicate", "n_doses"]].values.tolist() == [["100:L1", 0, 3], ["200:L1", 1, 3]]

    def test_a_missing_well_does_not_merge_plates(self, write_raw) -> None:
        series = _series(write_raw([*_plate(100, 1, DOSES[:2]), *_plate(200, 2, DOSES)]))
        assert series["n_doses"].tolist() == [2, 3]

    def test_another_slot_on_the_same_plate_is_another_series(self, write_raw) -> None:
        wells = [*_plate(100, 1, DOSES), *_plate(100, 1, DOSES, slot="L2")[2:]]
        assert _series(write_raw(wells))["series_key"].tolist() == ["100:L1", "100:L2"]

    def test_a_slot_screened_twice_on_a_plate_is_split_by_position(self, write_raw) -> None:
        """GDSC2 repeats some TAG blocks on one plate; each block is its own series."""
        repeat = [_well("L1-D1-S", 40.0, POSITION=20 + i, DRUG_ID=1001, CONC=dose) for i, dose in enumerate(DOSES)]
        series = _series(write_raw([*_plate(100, 1, DOSES), *repeat]))
        assert series["series_key"].tolist() == ["100:L1#0", "100:L1#1"]

    def test_plates_of_one_cell_stock_and_day_are_technical(self, write_raw) -> None:
        wells = [*_plate(100, 1, DOSES), *_plate(200, 2, DOSES), *_plate(300, 3, DOSES, cell_id=2)]
        assert _replicates(write_raw(wells)) == [["100:L1", 0, 0], ["200:L1", 0, 1], ["300:L1", 1, 0]]

    def test_another_seeding_day_is_biological(self, write_raw) -> None:
        later = _plate(100, 2, DOSES, DATE_CREATED="2015-03-01T00:00:00Z")
        assert _replicates(write_raw([*_plate(200, 1, DOSES), *later])) == [["200:L1", 0, 0], ["100:L1", 1, 0]]

    def test_another_slot_or_library_of_one_seeding_is_technical(self, write_raw) -> None:
        wells = [*_plate(100, 1, DOSES), *_plate(200, 2, DOSES, slot="L7", DRUGSET_ID=160)]
        assert _replicates(write_raw(wells)) == [["100:L1", 0, 0], ["200:L7", 0, 1]]

    def test_the_seeding_day_is_the_uk_date(self, write_raw) -> None:
        """GDSC1 stores midnight BST as 23:00Z of the previous day."""
        wells = [
            *_plate(100, 1, DOSES, DATE_CREATED="2011-04-03T23:00:00Z"),
            *_plate(200, 2, DOSES, DATE_CREATED="2011-04-04T09:00:00Z"),
        ]
        series = _series(write_raw(wells))
        assert series["experiment"].tolist() == ["1:2011-04-04", "1:2011-04-04"]

    def test_plate_metadata_is_kept_per_series(self, write_raw) -> None:
        series = _series(write_raw(_plate(100, 1, DOSES)))
        row = series.iloc[0]
        assert (row["barcode"], row["scan_id"], row["assay"], row["drugset_id"]) == ("100", "1", "Glo", "159")
        assert row["tag"] == "L1-D10-S;L1-D11-S;L1-D12-S"
