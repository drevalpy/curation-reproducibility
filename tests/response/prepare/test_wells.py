"""Tests for renaming a preparer's wells into the columns ``drevalpy.curation.curate()`` reads."""

import numpy as np
import pandas as pd

from data_curation.response.prepare.wells import WELL_COLUMNS, SeriesSpec, Wells, metadata_columns, to_wells

SPEC = SeriesSpec(
    sample="cell",
    drug="compound",
    key=("plate", "slot"),
    experiment=("stock",),
    metadata=("plate", "DATE"),
    position="pos",
)


def _frame(**overrides) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "cell": ["c1", "c1"],
            "compound": [1001.0, 1001.0],
            "concentration": [0.1, 1.0],
            "intensity": [0.9, 0.4],
            "plate": ["P1", "P1"],
            "slot": [4.0, np.nan],
            "stock": ["s", "s"],
            "pos": [3, 7],
            "DATE": ["2020-01-01", "2020-01-01"],
        }
    )
    return frame.assign(**overrides)


class TestToWells:
    def test_columns_are_curates_then_the_raw_metadata(self) -> None:
        wells = to_wells(Wells(_frame(), SPEC))
        assert list(wells.columns) == [*WELL_COLUMNS, "plate", "DATE"]
        assert metadata_columns(wells) == ["plate", "DATE"]

    def test_key_columns_are_joined_into_the_replicate_label(self) -> None:
        wells = to_wells(Wells(_frame(), SPEC))
        assert wells["replicate"].astype(str).tolist() == ["P1:4", "P1:"]
        assert wells["experiment"].astype(str).tolist() == ["s", "s"]

    def test_identifiers_and_positions_pass_through_unchanged(self) -> None:
        wells = to_wells(Wells(_frame(), SPEC))
        assert wells["sample"].tolist() == ["c1", "c1"]
        assert wells["drug"].tolist() == [1001.0, 1001.0]
        assert wells["position"].tolist() == [3, 7]

    def test_columns_mixing_types_are_written_as_strings(self, tmp_path) -> None:
        frame = _frame(cell=[906, "ACH-1"])
        wells = to_wells(Wells(frame, SPEC))
        assert wells["sample"].tolist() == ["906", "ACH-1"]
        assert wells["DATE"].tolist() == ["2020-01-01", "2020-01-01"]

        wells = to_wells(Wells(_frame(DATE=[102720, "102721"], plate=[None, "P1"]), SPEC))
        assert wells["DATE"].tolist() == ["102720", "102721"]
        assert wells["plate"].tolist() == [None, "P1"]
        wells.to_parquet(tmp_path / "wells.parquet", index=False)

    def test_a_spec_without_key_experiment_or_position_leaves_those_columns_out(self) -> None:
        wells = to_wells(Wells(_frame(), SeriesSpec(sample="cell", drug="compound")))
        assert list(wells.columns) == ["sample", "drug", "concentration", "intensity"]

    def test_curate_accepts_the_result(self) -> None:
        from drevalpy.curation import curate

        frame = pd.DataFrame(
            {
                "cell": ["c1"] * 5,
                "compound": ["d"] * 5,
                "concentration": [0.001, 0.01, 0.1, 1.0, 10.0],
                "intensity": [1.0, 0.95, 0.7, 0.3, 0.1],
                "plate": ["P1"] * 5,
                "slot": [1] * 5,
                "stock": ["s"] * 5,
                "pos": range(5),
                "DATE": ["x"] * 5,
            }
        )
        wells = to_wells(Wells(frame, SPEC))

        adata = curate(wells, metadata=metadata_columns(wells), max_workers=1, fit_speed="fast")

        assert adata.obs_names.tolist() == ["c1"]
