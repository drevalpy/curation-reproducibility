"""Tests for building a Cellosaurus-keyed AnnData from a native-ID-keyed matrix."""

import logging

import numpy as np
import pandas as pd
import pytest

from data_curation.omics.samples import to_sample_anndata

IDS = {
    "a": ("CVCL_0001", "Alpha"),
    "b": ("CVCL_0002", "Beta"),
    "c": ("CVCL_0001", "Alpha again"),
}


def _frame(index: list[str], columns: list[str]) -> pd.DataFrame:
    data = np.arange(len(index) * len(columns), dtype=float).reshape(len(index), len(columns))
    return pd.DataFrame(data, index=pd.Index(index), columns=pd.Index(columns))


class TestToSampleAnndata:
    def test_obs_are_accessions_with_the_cell_line_name(self) -> None:
        adata = to_sample_anndata(_frame(["a", "b"], ["G1", "G2"]), IDS, var_name="gene_symbol")

        assert list(adata.obs_names) == ["CVCL_0001", "CVCL_0002"]
        assert adata.obs_names.name == "sample_id"
        assert list(adata.obs["cell_line_name"]) == ["Alpha", "Beta"]
        assert adata.var_names.name == "gene_symbol"
        assert np.asarray(adata.X).dtype == np.float32

    def test_unmapped_samples_are_dropped_with_a_warning(self, caplog) -> None:
        values = _frame(["a", "x", "b"], ["G1"])

        with caplog.at_level(logging.WARNING):
            adata = to_sample_anndata(values, IDS, var_name="gene_symbol")

        assert list(adata.obs_names) == ["CVCL_0001", "CVCL_0002"]
        assert np.asarray(adata.X)[:, 0].tolist() == [0.0, 2.0]
        assert "Dropping 1 of 3 samples" in caplog.text

    @pytest.mark.filterwarnings("ignore:Observation names are not unique")
    def test_the_first_sample_of_a_shared_accession_is_kept(self, caplog) -> None:
        values = _frame(["a", "b", "c"], ["G1", "G2"])

        with caplog.at_level(logging.WARNING):
            adata = to_sample_anndata(values, IDS, var_name="gene_symbol")

        assert list(adata.obs_names) == ["CVCL_0001", "CVCL_0002"]
        assert list(adata.obs["cell_line_name"]) == ["Alpha", "Beta"]
        assert np.asarray(adata.X).tolist() == [[0.0, 1.0], [2.0, 3.0]]
        assert "Dropping 1 samples whose accession" in caplog.text

    def test_layers_follow_the_same_sample_selection(self) -> None:
        values = _frame(["a", "x", "b"], ["G1"])
        layer = values * 10

        adata = to_sample_anndata(values, IDS, var_name="gene_symbol", layers={"counts": layer})

        assert adata.layers["counts"][:, 0].tolist() == [0.0, 20.0]
        assert adata.layers["counts"].dtype == np.float32

    def test_repeated_feature_names_are_made_unique(self) -> None:
        values = _frame(["a"], ["SNORD", "SNORD", "TP53"])

        with pytest.warns(UserWarning, match="not unique"):
            adata = to_sample_anndata(values, IDS, var_name="gene_symbol")

        assert list(adata.var_names) == ["SNORD", "SNORD-1", "TP53"]

    def test_native_ids_may_be_integers(self) -> None:
        values = _frame(["1", "2"], ["G1"])
        values.index = pd.Index([1, 2])

        adata = to_sample_anndata(values, {"1": ("CVCL_0001", "Alpha")}, var_name="gene_symbol")

        assert list(adata.obs_names) == ["CVCL_0001"]
