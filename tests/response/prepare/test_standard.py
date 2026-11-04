"""Tests for the standard-format viability preparer.

The well table arrives already in curate's column names, so what matters is that
the optional columns become the series spec, everything else stays metadata, and
the values reach the well table untouched (bar the clip at zero).
"""

import pandas as pd
import pytest
from upath import UPath

from data_curation.core.errors import CurationError
from data_curation.response.prepare.standard import prepare_standard
from data_curation.response.prepare.wells import to_wells
from data_curation.response.prepare_viability import prepare

MINIMAL = {
    "sample": ["S1", "S1"],
    "drug": ["D1", "D1"],
    "concentration": [0.1, 1.0],
    "intensity": [0.9, 0.4],
}

FULL = MINIMAL | {
    "experiment": ["E1", "E2"],
    "replicate": ["R1", "R2"],
    "position": ["A01", "A02"],
    "plate_type": ["384", "384"],
}


def _write(raw_dir: UPath, columns: dict[str, list]) -> UPath:
    pd.DataFrame(columns).to_parquet(str(raw_dir / "wells.parquet"), index=False)
    return raw_dir


def _wells(raw_dir: UPath) -> pd.DataFrame:
    return to_wells(prepare("Screen", {"format": "standard"}, raw_dir))


class TestColumns:
    def test_a_minimal_table_has_only_the_required_columns(self, tmp_dir: UPath) -> None:
        wells = _wells(_write(tmp_dir, MINIMAL))
        assert list(wells.columns) == ["sample", "drug", "concentration", "intensity"]

    def test_a_full_table_keeps_the_optional_columns_then_the_metadata(self, tmp_dir: UPath) -> None:
        wells = _wells(_write(tmp_dir, FULL))
        assert list(wells.columns) == [
            "sample",
            "drug",
            "concentration",
            "intensity",
            "experiment",
            "replicate",
            "position",
            "plate_type",
        ]

    def test_the_spec_names_the_optional_columns(self, tmp_dir: UPath) -> None:
        spec = prepare_standard(_write(tmp_dir, FULL), {}).spec
        assert (spec.key, spec.experiment, spec.position, spec.metadata) == (
            ("replicate",),
            ("experiment",),
            "position",
            ("plate_type",),
        )


class TestValues:
    def test_the_values_reach_the_well_table_untouched(self, tmp_dir: UPath) -> None:
        wells = _wells(_write(tmp_dir, FULL))
        assert wells.astype(str).to_dict("list") == {
            column: [str(value) for value in values] for column, values in FULL.items()
        }

    def test_negative_intensity_is_clipped_at_zero(self, tmp_dir: UPath) -> None:
        wells = _wells(_write(tmp_dir, MINIMAL | {"intensity": [-0.2, 0.4]}))
        assert wells["intensity"].tolist() == pytest.approx([0.0, 0.4])


class TestContract:
    def test_a_missing_file_is_a_curation_error(self, tmp_dir: UPath) -> None:
        with pytest.raises(CurationError, match="wells.parquet"):
            prepare_standard(tmp_dir, {})

    def test_a_missing_required_column_is_a_curation_error(self, tmp_dir: UPath) -> None:
        columns = {key: value for key, value in MINIMAL.items() if key != "intensity"}
        with pytest.raises(CurationError, match="intensity"):
            prepare_standard(_write(tmp_dir, columns), {})
