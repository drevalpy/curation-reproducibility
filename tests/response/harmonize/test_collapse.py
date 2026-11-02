"""Tests for the native -> canonical duplicate collapse rule."""

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from data_curation.response.harmonize.collapse import (
    NATIVE_IDS_COLUMN,
    Collapse,
    collapse_axis,
    log_collapses,
)


def _adata(
    obs_names: list[str],
    x: list[list[float]],
    *,
    signal_quality: list[list[float]] | None = None,
    rmse: list[list[float]] | None = None,
    var_names: list[str] | None = None,
) -> ad.AnnData:
    layers = {}
    if signal_quality is not None:
        layers["signal_quality"] = np.array(signal_quality, dtype=np.float32)
    if rmse is not None:
        layers["RMSE"] = np.array(rmse, dtype=np.float32)
    return ad.AnnData(
        X=np.array(x, dtype=np.float32),
        obs=pd.DataFrame(index=pd.Index(obs_names)),
        var=pd.DataFrame(index=pd.Index(var_names or [f"d{i}" for i in range(len(x[0]))])),
        layers=layers,
    )


def _canonical(mapping: dict[str, str]) -> pd.Series:
    return pd.Series(mapping)


def _values(matrix) -> list[float]:
    """Flatten an AnnData matrix, whose static type is a wide union."""
    return np.asarray(matrix).ravel().tolist()


class TestInjectiveAxis:
    def test_labels_are_renamed_and_sorted(self) -> None:
        adata = _adata(["SIDM2", "SIDM1"], [[2.0], [1.0]])
        result, collapses = collapse_axis(adata, _canonical({"SIDM2": "CVCL_B", "SIDM1": "CVCL_A"}), axis=0)
        assert list(result.obs_names) == ["CVCL_A", "CVCL_B"]
        assert _values(result.X) == [1.0, 2.0]
        assert collapses == []

    def test_layers_follow_the_rename(self) -> None:
        adata = _adata(["a"], [[1.0]], signal_quality=[[0.5]], rmse=[[0.1]])
        result, _ = collapse_axis(adata, _canonical({"a": "CVCL_A"}), axis=0)
        assert _values(result.layers["signal_quality"]) == [0.5]
        assert _values(result.layers["RMSE"]) == pytest.approx([0.1])

    def test_the_x_alias_does_not_become_a_real_layer(self) -> None:
        """Anndata 0.13 lists a None key in .layers as an alias for .X.

        Stringifying the keys while copying turned that alias into a real layer
        literally named "None" - a second copy of X in the written file.
        """
        adata = _adata(["a"], [[1.0]], signal_quality=[[0.5]])
        result, _ = collapse_axis(adata, _canonical({"a": "CVCL_A"}), axis=0)
        assert "None" not in [str(key) for key in result.layers.keys() if key is not None]  # noqa: SIM118

    def test_var_annotations_survive(self) -> None:
        adata = _adata(["a"], [[1.0]])
        adata.var["canonical_smiles"] = ["CCO"]
        result, _ = collapse_axis(adata, _canonical({"a": "CVCL_A"}), axis=0)
        assert list(result.var["canonical_smiles"]) == ["CCO"]

    def test_uns_survives(self) -> None:
        adata = _adata(["a"], [[1.0]])
        adata.uns["x_column"] = "pEC50"
        result, _ = collapse_axis(adata, _canonical({"a": "CVCL_A"}), axis=0)
        assert result.uns["x_column"] == "pEC50"

    def test_the_native_sources_are_recorded(self) -> None:
        adata = _adata(["SIDM1"], [[1.0]])
        result, _ = collapse_axis(adata, _canonical({"SIDM1": "CVCL_A"}), axis=0)
        assert list(result.obs[NATIVE_IDS_COLUMN]) == ["SIDM1"]


class TestCollapseRule:
    """The documented rule: fitted over missing, then higher quality, then lower RMSE."""

    def test_a_fitted_value_beats_a_missing_one(self) -> None:
        adata = _adata(["d1", "d2"], [[np.nan], [7.0]])
        result, _ = collapse_axis(adata, _canonical({"d1": "CID", "d2": "CID"}), axis=0)
        assert _values(result.X) == [7.0]

    def test_higher_signal_quality_wins(self) -> None:
        adata = _adata(["d1", "d2"], [[1.0], [2.0]], signal_quality=[[0.1], [0.9]])
        result, _ = collapse_axis(adata, _canonical({"d1": "CID", "d2": "CID"}), axis=0)
        assert _values(result.X) == [2.0]

    def test_quality_outranks_rmse(self) -> None:
        """A worse RMSE cannot rescue a curve with lower signal quality."""
        adata = _adata(["d1", "d2"], [[1.0], [2.0]], signal_quality=[[0.1], [0.9]], rmse=[[0.01], [0.5]])
        result, _ = collapse_axis(adata, _canonical({"d1": "CID", "d2": "CID"}), axis=0)
        assert _values(result.X) == [2.0]

    def test_rmse_breaks_a_quality_tie(self) -> None:
        adata = _adata(["d1", "d2"], [[1.0], [2.0]], signal_quality=[[0.5], [0.5]], rmse=[[0.5], [0.1]])
        result, _ = collapse_axis(adata, _canonical({"d1": "CID", "d2": "CID"}), axis=0)
        assert _values(result.X) == [2.0]

    def test_a_full_tie_takes_the_smaller_native_id(self) -> None:
        adata = _adata(["d2", "d1"], [[2.0], [1.0]], signal_quality=[[0.5], [0.5]], rmse=[[0.1], [0.1]])
        result, _ = collapse_axis(adata, _canonical({"d2": "CID", "d1": "CID"}), axis=0)
        assert _values(result.X) == [1.0]

    def test_the_result_does_not_depend_on_row_order(self) -> None:
        """Row order used to decide it, via drevalpy's aggfunc="first"."""
        forward = _adata(["d1", "d2"], [[1.0], [2.0]], signal_quality=[[0.1], [0.9]])
        reverse = _adata(["d2", "d1"], [[2.0], [1.0]], signal_quality=[[0.9], [0.1]])
        mapping = _canonical({"d1": "CID", "d2": "CID"})
        assert _values(collapse_axis(forward, mapping, axis=0)[0].X) == _values(
            collapse_axis(reverse, mapping, axis=0)[0].X
        )

    def test_missing_quality_layers_still_prefer_a_fitted_value(self) -> None:
        adata = _adata(["d1", "d2"], [[np.nan], [3.0]])
        result, _ = collapse_axis(adata, _canonical({"d1": "CID", "d2": "CID"}), axis=0)
        assert _values(result.X) == [3.0]

    def test_all_missing_stays_missing(self) -> None:
        adata = _adata(["d1", "d2"], [[np.nan], [np.nan]])
        result, _ = collapse_axis(adata, _canonical({"d1": "CID", "d2": "CID"}), axis=0)
        assert np.isnan(np.asarray(result.X)).all()

    def test_the_winner_is_chosen_per_cell_not_per_row(self) -> None:
        """A repeat screen can win one drug's cell and lose another's."""
        adata = _adata(
            ["d1", "d2"],
            [[1.0, 10.0], [2.0, 20.0]],
            signal_quality=[[0.9, 0.1], [0.1, 0.9]],
        )
        result, _ = collapse_axis(adata, _canonical({"d1": "CID", "d2": "CID"}), axis=0)
        assert _values(result.X) == [1.0, 20.0]

    def test_every_layer_takes_the_same_winner(self) -> None:
        """A cell must not mix pEC50 from one screen with the RMSE of another."""
        adata = _adata(["d1", "d2"], [[1.0], [2.0]], signal_quality=[[0.1], [0.9]], rmse=[[0.5], [0.2]])
        result, _ = collapse_axis(adata, _canonical({"d1": "CID", "d2": "CID"}), axis=0)
        assert _values(result.X) == [2.0]
        assert _values(result.layers["RMSE"]) == pytest.approx([0.2])


class TestReporting:
    def test_a_collapse_is_reported_with_its_sources(self) -> None:
        adata = _adata(["d1", "d2"], [[1.0], [2.0]], signal_quality=[[0.9], [0.1]])
        _, collapses = collapse_axis(adata, _canonical({"d1": "CID", "d2": "CID"}), axis=0)
        assert collapses == [Collapse(canonical="CID", natives=("d1", "d2"), cells_won=(1, 0))]

    def test_only_genuine_collapses_are_reported(self) -> None:
        adata = _adata(["d1", "d2"], [[1.0], [2.0]])
        _, collapses = collapse_axis(adata, _canonical({"d1": "CID_A", "d2": "CID_B"}), axis=0)
        assert collapses == []

    def test_the_sources_are_recorded_on_the_surviving_entry(self) -> None:
        adata = _adata(["d2", "d1"], [[2.0], [1.0]])
        result, _ = collapse_axis(adata, _canonical({"d1": "CID", "d2": "CID"}), axis=0)
        assert list(result.obs[NATIVE_IDS_COLUMN]) == ["d1;d2"]

    def test_every_collapse_is_logged(self, caplog) -> None:
        collapses = [Collapse(canonical="CID", natives=("1001", "1002"), cells_won=(3, 5))]
        with caplog.at_level("INFO"):
            log_collapses(collapses, axis_name="drug")
        assert "CID" in caplog.text
        assert "1002 won 5" in caplog.text

    def test_an_injective_axis_says_so(self, caplog) -> None:
        with caplog.at_level("INFO"):
            log_collapses([], axis_name="sample")
        assert "injective" in caplog.text


class TestVarAxis:
    def test_collapsing_var_leaves_obs_untouched(self) -> None:
        adata = _adata(["CVCL_A", "CVCL_B"], [[1.0, 2.0], [3.0, 4.0]], var_names=["1001", "1002"])
        result, collapses = collapse_axis(adata, _canonical({"1001": "CID", "1002": "CID"}), axis=1)
        assert list(result.obs_names) == ["CVCL_A", "CVCL_B"]
        assert list(result.var_names) == ["CID"]
        assert collapses == [Collapse(canonical="CID", natives=("1001", "1002"), cells_won=(2, 0))]

    def test_the_var_winner_follows_the_same_rule(self) -> None:
        adata = _adata(
            ["CVCL_A"],
            [[1.0, 2.0]],
            signal_quality=[[0.1, 0.9]],
            var_names=["1001", "1002"],
        )
        result, _ = collapse_axis(adata, _canonical({"1001": "CID", "1002": "CID"}), axis=1)
        assert _values(result.X) == [2.0]

    def test_obs_annotations_survive_a_var_collapse(self) -> None:
        adata = _adata(["CVCL_A"], [[1.0, 2.0]], var_names=["1001", "1002"])
        adata.obs["cell_line_name"] = ["MEC-1"]
        result, _ = collapse_axis(adata, _canonical({"1001": "CID", "1002": "CID"}), axis=1)
        assert list(result.obs["cell_line_name"]) == ["MEC-1"]
