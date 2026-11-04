"""Tests for attaching a standard dataset's ``published.parquet`` as AnnData layers."""

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from upath import UPath

from data_curation.core.errors import CurationError
from data_curation.response.harmonize.published import add_published_layers

DRUG_MAPPING = pd.DataFrame({"DRUG_ID": ["D1", "D2"], "pubchem_id": ["31703", "2244"]})


@pytest.fixture
def adata() -> ad.AnnData:
    return ad.AnnData(
        X=np.zeros((2, 2), dtype=np.float32),
        obs=pd.DataFrame(index=pd.Index(["PDO-7", "PDO-8"])),
        var=pd.DataFrame(index=pd.Index(["2244", "31703"])),
    )


def _write(raw_dir: UPath, columns: dict[str, list]) -> UPath:
    pd.DataFrame(columns).to_parquet(str(raw_dir / "published.parquet"), index=False)
    return raw_dir


def _add(adata: ad.AnnData, raw_dir: UPath) -> None:
    add_published_layers(adata, "standard", raw_dir, raw_dir / "model_list.csv", DRUG_MAPPING)


class TestStandard:
    def test_every_metric_becomes_a_published_layer(self, tmp_dir: UPath, adata) -> None:
        _add(adata, _write(tmp_dir, {"sample": ["PDO-7"], "drug": ["D1"], "AUC": [0.4], "IC50": [3.0]}))
        assert {"AUC_published", "IC50_published"} <= set(adata.layers)

    def test_the_layers_align_to_the_canonical_axes(self, tmp_dir: UPath, adata) -> None:
        raw_dir = _write(tmp_dir, {"sample": ["PDO-8", "PDO-7"], "drug": ["D1", "D2"], "AUC": [0.4, 0.6]})
        _add(adata, raw_dir)
        layer = adata.layers["AUC_published"]
        assert layer[1, 1] == pytest.approx(0.4)
        assert layer[0, 0] == pytest.approx(0.6)
        assert np.isnan(layer[0, 1])
        assert np.isnan(layer[1, 0])

    def test_duplicates_are_averaged(self, tmp_dir: UPath, adata) -> None:
        raw_dir = _write(
            tmp_dir, {"sample": ["PDO-7", "PDO-7"], "drug": ["D1", "D1"], "AUC": [0.2, 0.4], "IC50": [1.0, 3.0]}
        )
        _add(adata, raw_dir)
        assert adata.layers["AUC_published"][0, 1] == pytest.approx(0.3)
        assert adata.layers["IC50_published"][0, 1] == pytest.approx(2.0)

    def test_an_unmapped_drug_is_skipped(self, tmp_dir: UPath, adata) -> None:
        _add(adata, _write(tmp_dir, {"sample": ["PDO-7", "PDO-7"], "drug": ["D1", "D9"], "AUC": [0.4, 0.9]}))
        layer = adata.layers["AUC_published"]
        assert layer[0, 1] == pytest.approx(0.4)
        assert int(np.isfinite(layer).sum()) == 1

    def test_an_absent_file_adds_nothing(self, tmp_dir: UPath, adata) -> None:
        _add(adata, tmp_dir)
        assert not any(str(name).endswith("_published") for name in adata.layers)

    def test_a_table_without_metrics_is_a_curation_error(self, tmp_dir: UPath, adata) -> None:
        with pytest.raises(CurationError, match="no metric column"):
            _add(adata, _write(tmp_dir, {"sample": ["PDO-7"], "drug": ["D1"]}))
