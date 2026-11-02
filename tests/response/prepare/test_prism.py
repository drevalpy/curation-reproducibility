"""Tests for the PRISM secondary screen preparer.

The raw matrix is wide (cell lines x conditions), so the risks are a condition
joined to the wrong dose or plate, the MTS010 redo not superseding the original
screen, and the two cell-line pools of 20Q2 collapsing into one series.
"""

import pandas as pd
import pytest
from series_helpers import number
from upath import UPath

from data_curation.response.prepare.prism import prepare_prism

DOSES = [0.1, 1.0, 10.0]


def _conditions(plate: str, drug: str = "BRD-1", screen: str = "HTS002", doses=DOSES) -> list[dict]:
    """One condition column per dose of *drug* on detection plate *plate*."""
    compound_plate = plate.rsplit("_", 1)[0]
    return [
        {
            "column_name": f"{plate}:A{i:02d}::{screen}",
            "broad_id": drug,
            "dose": dose,
            "perturbation_type": "experimental_treatment",
            "screen_id": screen,
            "detection_plate": plate,
            "compound_plate": compound_plate,
            "well": f"A{i:02d}",
            "name": "drug",
        }
        for i, dose in enumerate(doses, start=1)
    ]


@pytest.fixture
def prism_dir(tmp_dir: UPath):
    """Return a factory writing the three PRISM tables; LFC values default to 0 (full viability)."""

    def _write(conditions: list[dict], rows: dict[str, str], lfc: dict[str, dict[str, float]] | None = None) -> UPath:
        info = pd.DataFrame(conditions)
        info.to_csv(str(tmp_dir / "secondary-screen-replicate-treatment-info.csv"), index=False)
        pd.DataFrame({"row_name": list(rows), "depmap_id": list(rows.values())}).to_csv(
            str(tmp_dir / "secondary-screen-cell-line-info.csv"), index=False
        )
        matrix = pd.DataFrame(0.0, index=pd.Index(list(rows)), columns=pd.Index(info["column_name"]))
        for row, values in (lfc or {}).items():
            for column, value in values.items():
                matrix.loc[row, column] = value
        matrix.to_csv(str(tmp_dir / "secondary-screen-logfold-change.csv"))
        return tmp_dir

    return _write


def _number(raw_dir: UPath):
    wells = prepare_prism(raw_dir, {})
    return number(wells)


class TestViability:
    def test_log2_fold_change_is_linearized(self, prism_dir) -> None:
        conditions = _conditions("PREP014_CS5_X1")
        raw = prism_dir(conditions, {"ACH-1": "ACH-1"}, {"ACH-1": {conditions[0]["column_name"]: -1.0}})
        viability, _ = _number(raw)
        assert viability["intensity"].tolist() == pytest.approx([0.5, 1.0, 1.0])

    def test_each_condition_keeps_its_dose(self, prism_dir) -> None:
        viability, _ = _number(prism_dir(_conditions("PREP014_CS5_X1"), {"ACH-1": "ACH-1"}))
        assert viability["concentration"].tolist() == DOSES

    def test_a_missing_value_is_dropped(self, prism_dir) -> None:
        conditions = _conditions("PREP014_CS5_X1")
        raw = prism_dir(conditions, {"ACH-1": "ACH-1"}, {"ACH-1": {conditions[0]["column_name"]: float("nan")}})
        assert len(prepare_prism(raw, {}).frame) == 2

    def test_controls_are_not_measurements(self, prism_dir) -> None:
        conditions = _conditions("PREP014_CS5_X1")
        conditions[0]["perturbation_type"] = "vehicle_control"
        assert len(prepare_prism(prism_dir(conditions, {"ACH-1": "ACH-1"}), {}).frame) == 2

    def test_the_mts010_redo_supersedes_the_original_screen(self, prism_dir) -> None:
        conditions = [*_conditions("PREP014_CS5_X1"), *_conditions("PROS001_PR500_120H_X1_P4", screen="MTS010")]
        _, series = _number(prism_dir(conditions, {"ACH-1": "ACH-1"}))
        assert series["screen_id"].tolist() == ["MTS010"]


class TestNativeIdentifiers:
    def test_row_names_map_to_depmap_ids(self, prism_dir) -> None:
        _, series = _number(prism_dir(_conditions("PREP014_CS5_X1"), {"PR500_ACH-1": "ACH-1"}))
        assert series[["native_sample", "native_drug"]].values.tolist() == [["ACH-1", "BRD-1"]]


class TestSeries:
    """One series per (detection plate, pool)."""

    def test_replicate_plates_are_technical_replicates(self, prism_dir) -> None:
        conditions = [*_conditions("PREP014_CS5_X1"), *_conditions("PREP014_CS5_X2")]
        _, series = _number(prism_dir(conditions, {"ACH-1": "ACH-1"}))
        columns = ["detection_plate", "n_doses", "biological_replicate", "technical_replicate"]
        assert series[columns].values.tolist() == [["PREP014_CS5_X1", 3, 0, 0], ["PREP014_CS5_X2", 3, 0, 1]]

    def test_a_missing_well_does_not_merge_plates(self, prism_dir) -> None:
        conditions = [*_conditions("PREP014_CS5_X1"), *_conditions("PREP014_CS5_X2")]
        missing = {"ACH-1": {conditions[0]["column_name"]: float("nan")}}
        _, series = _number(prism_dir(conditions, {"ACH-1": "ACH-1"}, missing))
        assert series["n_doses"].tolist() == [2, 3]

    def test_another_compound_plate_is_a_biological_replicate(self, prism_dir) -> None:
        conditions = [*_conditions("PREP014_CS5_X1"), *_conditions("PREP015_CS5_X1")]
        _, series = _number(prism_dir(conditions, {"ACH-1": "ACH-1"}))
        assert series[["biological_replicate", "technical_replicate"]].values.tolist() == [[0, 0], [1, 0]]

    def test_a_cell_line_in_both_pools_gets_a_series_per_pool(self, prism_dir) -> None:
        """20Q2 screens 96 cell lines in both PR300 and PR500; each pool is its own measurement."""
        conditions = _conditions("PROS001_PR500_120H_X1_P4", screen="MTS010")
        _, series = _number(prism_dir(conditions, {"PR300_ACH-1": "ACH-1", "PR500_ACH-1": "ACH-1"}))
        assert series["pool"].tolist() == ["PR300", "PR500"]
        assert series["n_doses"].tolist() == [3, 3]

    def test_without_a_pool_prefix_the_pool_comes_from_the_plate(self, prism_dir) -> None:
        conditions = _conditions("PROS001_PR500_120H_X1_P4", screen="MTS010")
        _, series = _number(prism_dir(conditions, {"ACH-1": "ACH-1"}))
        assert series["pool"].tolist() == ["PR500"]

    def test_a_single_pool_fills_the_plates_that_do_not_name_it(self, prism_dir) -> None:
        """19Q4 screened only PR500, but only its MTS plate names say so."""
        conditions = [
            *_conditions("PREP014_CS5_X1"),
            *_conditions("PROS001_PR500_120H_X1_P4", drug="BRD-2", screen="MTS010"),
        ]
        _, series = _number(prism_dir(conditions, {"ACH-1": "ACH-1"}))
        assert series["pool"].tolist() == ["PR500", "PR500"]

    def test_an_unnamed_pool_stays_missing(self, prism_dir) -> None:
        _, series = _number(prism_dir(_conditions("PREP014_CS5_X1"), {"ACH-1": "ACH-1"}))
        assert series["pool"].isna().all()
