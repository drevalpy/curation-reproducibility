"""Tests for the standard-format transcriptomics processor and its route in the process stage."""

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import yaml
from upath import UPath

from data_curation.core import layout
from data_curation.omics import COUNT_LAYER, process_transcriptomics
from data_curation.omics.transcriptomics.standard import process_standard_transcriptomics


def _write_expression(raw_dir: UPath, frame: pd.DataFrame) -> UPath:
    raw_dir.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(str(raw_dir / "expression.parquet"), index=False)
    return raw_dir


def _expression(**columns: list) -> pd.DataFrame:
    base = {
        "sample_id": ["S1", "S1", "S2", "S2"],
        "gene_symbol": ["TP53", "KRAS", "TP53", "KRAS"],
        "tpm": [0.0, 1.0, 3.0, 7.0],
    }
    return pd.DataFrame(base | columns)


class TestProcessStandardTranscriptomics:
    def test_x_is_log2_tpm_plus_one(self, tmp_path) -> None:
        raw = _write_expression(UPath(tmp_path), _expression())

        adata = process_standard_transcriptomics(raw, "counts")

        assert np.asarray(adata.X).dtype == np.float32
        np.testing.assert_allclose(adata.to_df().loc[["S1", "S2"], ["TP53", "KRAS"]], [[0.0, 1.0], [2.0, 3.0]])
        assert adata.uns["expression_unit"] == "log2_tpm_plus_1"

    def test_counts_become_a_layer_with_their_unit(self, tmp_path) -> None:
        raw = _write_expression(UPath(tmp_path), _expression(counts=[1.0, 2.0, 3.0, 4.0]))

        adata = process_standard_transcriptomics(raw, "salmon_numreads")

        layer = adata.to_df(layer=COUNT_LAYER)
        assert adata.layers[COUNT_LAYER].dtype == np.float32
        assert layer.loc["S2", "KRAS"] == 4.0
        assert adata.uns["layer_units"] == {COUNT_LAYER: "salmon_numreads"}

    def test_without_counts_there_is_no_layer(self, tmp_path) -> None:
        raw = _write_expression(UPath(tmp_path), _expression())

        adata = process_standard_transcriptomics(raw, "counts")

        assert COUNT_LAYER not in adata.layers
        assert "layer_units" not in adata.uns

    def test_a_gene_missing_in_one_sample_is_dropped(self, tmp_path) -> None:
        frame = pd.DataFrame(
            {"sample_id": ["S1", "S1", "S2"], "gene_symbol": ["TP53", "KRAS", "TP53"], "tpm": [1.0, 1.0, 1.0]}
        )
        raw = _write_expression(UPath(tmp_path), frame)

        adata = process_standard_transcriptomics(raw, "counts")

        assert list(adata.var_names) == ["TP53"]

    def test_a_gene_with_missing_counts_is_dropped(self, tmp_path) -> None:
        raw = _write_expression(UPath(tmp_path), _expression(counts=[1.0, np.nan, 3.0, 4.0]))

        adata = process_standard_transcriptomics(raw, "counts")

        assert list(adata.var_names) == ["TP53"]

    def test_a_repeated_symbol_keeps_its_first_row(self, tmp_path) -> None:
        frame = pd.DataFrame(
            {
                "sample_id": ["S1", "S1", "S2", "S2"],
                "gene_symbol": ["TP53", "TP53", "TP53", "TP53"],
                "tpm": [1.0, 15.0, 3.0, 15.0],
            }
        )
        raw = _write_expression(UPath(tmp_path), frame)

        adata = process_standard_transcriptomics(raw, "counts")

        assert list(adata.var_names) == ["TP53"]
        np.testing.assert_allclose(adata.to_df()["TP53"], [1.0, 2.0])

    def test_samples_and_genes_keep_their_order_and_index_names(self, tmp_path) -> None:
        raw = _write_expression(UPath(tmp_path), _expression())

        adata = process_standard_transcriptomics(raw, "counts")

        assert list(adata.obs_names) == ["S1", "S2"]
        assert list(adata.var_names) == ["TP53", "KRAS"]
        assert adata.obs_names.name == layout.SAMPLE_INDEX_NAME
        assert adata.var_names.name == "gene_symbol"
        assert "cell_line_name" not in adata.obs


@pytest.fixture
def standard_workdir(monkeypatch, tmp_path) -> UPath:
    """A scratch repository root whose only dataset uses the standard transcriptomics format."""
    (tmp_path / "config").mkdir()
    config = {"datasets": {"Inhouse": {"transcriptomics": {"format": "standard", "counts_unit": "salmon_numreads"}}}}
    (tmp_path / "config" / "sources.yaml").write_text(yaml.safe_dump(config))
    monkeypatch.chdir(tmp_path)
    return UPath(tmp_path)


class TestMainRoutesStandardDatasets:
    def test_main_writes_the_standard_dataset(self, standard_workdir) -> None:
        _write_expression(layout.raw_dir("transcriptomics", "Inhouse"), _expression(counts=[1.0, 2.0, 3.0, 4.0]))

        process_transcriptomics.main("Inhouse")

        adata = ad.read_h5ad(str(layout.omics("transcriptomics", "Inhouse")))
        assert adata.shape == (2, 2)
        assert adata.uns["expression_unit"] == "log2_tpm_plus_1"
        assert dict(adata.uns["layer_units"]) == {COUNT_LAYER: "salmon_numreads"}
        assert adata.obs_names.name == layout.SAMPLE_INDEX_NAME
