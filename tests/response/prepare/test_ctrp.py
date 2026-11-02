"""Tests for the CTRPv1 and CTRPv2 viability preparers."""

import numpy as np
import pandas as pd
import pytest
from series_helpers import number
from upath import UPath

from data_curation.response.prepare.ctrp import prepare_ctrpv1, prepare_ctrpv2


def _write_wells(raw_dir: UPath, wells: list[tuple]) -> None:
    """Write ``(experiment_id, plate, dose, log2 value)`` wells of compound 606135."""
    frame = pd.DataFrame(
        wells, columns=pd.Index(["experiment_id", "assay_plate_barcode", "cpd_conc_umol", "bsub_value_log2"])
    )
    frame["raw_value_log2"] = 14.5
    frame["dmso_zscore_log2"] = 0.0
    frame["master_cpd_id"] = 606135
    frame.to_csv(str(raw_dir / "v20.data.per_cpd_well.txt"), sep="\t", index=False)


def _series(wells) -> pd.DataFrame:
    return number(wells).series


@pytest.fixture
def ctrpv1_dir(tmp_dir: UPath) -> UPath:
    """A raw directory holding a minimal CTRPv1 average-viability table."""
    frame = pd.DataFrame(
        {
            "ccl_name": ["MEC-1", "MEC-1"],
            "cpd_name": ["vorinostat", "vorinostat"],
            "cpd_conc_umol": [0.1, 1.0],
            "cpd_avg_pv": [0.92, -0.05],
        }
    )
    frame.to_csv(str(tmp_dir / "v10.D3.avg_pct_viability_data.txt"), sep="\t", index=False)
    return tmp_dir


@pytest.fixture
def ctrpv2_dir(tmp_dir: UPath) -> UPath:
    """A raw directory holding the four CTRPv2 tables the preparer joins."""
    _write_wells(tmp_dir, [(1, "AU1", 0.1, 0.0), (1, "AU1", 1.0, -1.0)])
    pd.DataFrame(
        {
            "experiment_id": [1, 2, 2],
            "run_id": ["R1", "R2", "R2"],
            "experiment_date": [20120501, 20120611, 20120612],
            "culture_media": ["DMEM001"] * 3,
            "baseline_signal": [0.2225] * 3,
            "cells_per_well": [500] * 3,
            "growth_mode": ["adherent"] * 3,
            "snp_fp_status": ["SNP-matched-reference"] * 3,
            "master_ccl_id": [7, 7, 7],
        }
    ).to_csv(str(tmp_dir / "v20.meta.per_experiment.txt"), sep="\t", index=False)
    pd.DataFrame({"master_ccl_id": [7], "ccl_name": ["MEC1"]}).to_csv(
        str(tmp_dir / "v20.meta.per_cell_line.txt"), sep="\t", index=False
    )
    pd.DataFrame({"master_cpd_id": [606135], "cpd_name": ["vorinostat"]}).to_csv(
        str(tmp_dir / "v20.meta.per_compound.txt"), sep="\t", index=False
    )
    return tmp_dir


class TestCtrpv1:
    def test_cpd_avg_pv_is_used_as_is(self, ctrpv1_dir: UPath) -> None:
        """CTRPv1 already publishes a DMSO-normalized fractional viability."""
        result = prepare_ctrpv1(ctrpv1_dir, {}).frame
        assert result["intensity"].tolist() == pytest.approx([0.92, 0.0])

    def test_the_native_name_is_the_sample(self, ctrpv1_dir: UPath) -> None:
        assert set(_series(prepare_ctrpv1(ctrpv1_dir, {}))["native_sample"]) == {"MEC-1"}

    def test_the_compound_name_is_the_drug(self, ctrpv1_dir: UPath) -> None:
        """CTRPv1's drug_mapping keys on cpd_name, not a numeric ID."""
        assert set(_series(prepare_ctrpv1(ctrpv1_dir, {}))["native_drug"]) == {"vorinostat"}

    def test_the_dose_comes_from_cpd_conc_umol(self, ctrpv1_dir: UPath) -> None:
        assert prepare_ctrpv1(ctrpv1_dir, {}).frame["concentration"].tolist() == [0.1, 1.0]


class TestCtrpv2:
    def test_log2_fold_change_is_linearized(self, ctrpv2_dir: UPath) -> None:
        """bsub_value_log2 is a log2 ratio to DMSO, so 0 is full viability."""
        result = prepare_ctrpv2(ctrpv2_dir, {}).frame
        assert result["intensity"].tolist() == pytest.approx([1.0, 0.5])

    def test_the_cell_line_name_is_joined_through_two_hops(self, ctrpv2_dir: UPath) -> None:
        """experiment_id -> master_ccl_id -> ccl_name."""
        assert set(_series(prepare_ctrpv2(ctrpv2_dir, {}))["native_sample"]) == {"MEC1"}

    def test_the_master_cpd_id_is_the_drug(self, ctrpv2_dir: UPath) -> None:
        """CTRPv2's drug_mapping keys on master_cpd_id, unlike v1's cpd_name."""
        assert set(_series(prepare_ctrpv2(ctrpv2_dir, {}))["native_drug"]) == {"606135"}

    def test_an_unjoinable_experiment_is_dropped(self, ctrpv2_dir: UPath) -> None:
        _write_wells(ctrpv2_dir, [(99, "AU1", 0.1, 0.0)])
        assert _series(prepare_ctrpv2(ctrpv2_dir, {})).empty

    def test_viability_is_clipped_at_zero(self, ctrpv2_dir: UPath) -> None:
        frame = pd.read_csv(str(ctrpv2_dir / "v20.data.per_cpd_well.txt"), sep="\t")
        frame["bsub_value_log2"] = [-np.inf, -1.0]
        frame.to_csv(str(ctrpv2_dir / "v20.data.per_cpd_well.txt"), sep="\t", index=False)
        assert prepare_ctrpv2(ctrpv2_dir, {}).frame["intensity"].min() == 0.0


class TestCtrpv1Series:
    def test_every_pair_is_one_series(self, ctrpv1_dir: UPath) -> None:
        series = _series(prepare_ctrpv1(ctrpv1_dir, {}))
        assert series[["replicate", "n_doses"]].values.tolist() == [[0, 2]]


class TestCtrpv2Series:
    """One series per (experiment, assay plate)."""

    def test_two_plates_are_two_technical_replicates(self, ctrpv2_dir: UPath) -> None:
        _write_wells(
            ctrpv2_dir, [(1, "AU1", 0.1, 0.0), (1, "AU2", 0.1, 0.0), (1, "AU1", 1.0, 0.0), (1, "AU2", 1.0, 0.0)]
        )
        series = _series(prepare_ctrpv2(ctrpv2_dir, {}))
        columns = ["series_key", "n_doses", "biological_replicate", "technical_replicate"]
        assert series[columns].values.tolist() == [["1:AU1", 2, 0, 0], ["1:AU2", 2, 0, 1]]

    def test_a_missing_well_does_not_merge_plates(self, ctrpv2_dir: UPath) -> None:
        _write_wells(ctrpv2_dir, [(1, "AU1", 0.1, 0.0), (1, "AU2", 0.1, 0.0), (1, "AU2", 1.0, 0.0)])
        assert _series(prepare_ctrpv2(ctrpv2_dir, {}))["n_doses"].tolist() == [1, 2]

    def test_another_experiment_is_a_biological_replicate(self, ctrpv2_dir: UPath) -> None:
        _write_wells(ctrpv2_dir, [(1, "AU1", 0.1, 0.0), (2, "AU9", 0.1, 0.0)])
        series = _series(prepare_ctrpv2(ctrpv2_dir, {}))
        assert series[["biological_replicate", "technical_replicate"]].values.tolist() == [[0, 0], [1, 0]]

    def test_a_series_repeated_on_one_plate_is_split_in_row_order(self, ctrpv2_dir: UPath) -> None:
        _write_wells(
            ctrpv2_dir, [(1, "AU1", 0.1, 0.0), (1, "AU1", 1.0, 0.0), (1, "AU1", 0.1, -1.0), (1, "AU1", 1.0, -1.0)]
        )
        viability, series = number(prepare_ctrpv2(ctrpv2_dir, {}))
        assert series["series_key"].tolist() == ["1:AU1#0", "1:AU1#1"]
        assert viability.groupby("replicate")["intensity"].mean().tolist() == pytest.approx([1.0, 0.5])

    def test_an_experiment_listed_twice_does_not_duplicate_wells(self, ctrpv2_dir: UPath) -> None:
        """153 experiment_ids appear once per plating date in meta.per_experiment."""
        _write_wells(ctrpv2_dir, [(2, "AU9", 0.1, 0.0), (2, "AU9", 1.0, 0.0)])
        wells = prepare_ctrpv2(ctrpv2_dir, {})
        assert len(wells.frame) == 2
        assert _series(wells).loc[0, "experiment_date"] == "20120611;20120612"

    def test_experiment_metadata_is_kept_per_series(self, ctrpv2_dir: UPath) -> None:
        row = _series(prepare_ctrpv2(ctrpv2_dir, {})).iloc[0]
        assert (row["run_id"], row["culture_media"], row["cells_per_well"], row["assay_plate_barcode"]) == (
            "R1",
            "DMEM001",
            "500",
            "AU1",
        )
