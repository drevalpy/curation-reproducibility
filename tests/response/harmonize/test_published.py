"""Tests for attaching published (author-reported) metrics as AnnData layers."""

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from upath import UPath

from data_curation.core.errors import CurationError
from data_curation.response.harmonize import published
from data_curation.response.harmonize.published import add_published_layers

DRUG_MAPPING = pd.DataFrame({"DRUG_ID": [1001, 1002], "pubchem_id": ["31703", "GDSC_1002"]})


@pytest.fixture
def adata() -> ad.AnnData:
    return ad.AnnData(
        X=np.zeros((1, 2), dtype=np.float32),
        obs=pd.DataFrame(index=pd.Index(["CVCL_0001"])),
        var=pd.DataFrame(index=pd.Index(["31703", "GDSC_1002"])),
    )


def _write_fitted(raw_dir: UPath, columns: dict[str, list]) -> None:
    pd.DataFrame(columns).to_excel(str(raw_dir / "fitted_dose_response.xlsx"), index=False)


@pytest.fixture(autouse=True)
def _sidm_map(monkeypatch) -> None:
    monkeypatch.setattr(published, "build_sidm_to_cellosaurus_map", lambda _path: {"SIDM1": ("CVCL_0001", "Line")})


class TestGdsc:
    def test_the_layer_is_keyed_by_cellosaurus_id_and_pubchem_id(self, tmp_path, adata) -> None:
        raw_dir = UPath(tmp_path)
        _write_fitted(
            raw_dir,
            {"SANGER_MODEL_ID": ["SIDM1", "SIDM1"], "DRUG_ID": [1001, 1002], "LN_IC50": [1.5, 2.5], "AUC": [0.1, 0.2]},
        )

        add_published_layers(adata, "gdsc", raw_dir, raw_dir / "model_list.csv", DRUG_MAPPING)

        assert adata.layers["LN_IC50_published"].tolist() == [[1.5, 2.5]]

    def test_a_table_without_model_ids_is_an_error_naming_the_column(self, tmp_path, adata) -> None:
        """A cell-line *name* can never join the Cellosaurus-keyed axis, so fail instead of matching nothing."""
        raw_dir = UPath(tmp_path)
        _write_fitted(raw_dir, {"CELL_LINE_NAME": ["A"], "DRUG_ID": [1001], "LN_IC50": [1.5], "AUC": [0.1]})

        with pytest.raises(CurationError, match="SANGER_MODEL_ID"):
            add_published_layers(adata, "gdsc", raw_dir, raw_dir / "model_list.csv", DRUG_MAPPING)

    def test_a_missing_workbook_adds_nothing(self, tmp_path, adata) -> None:
        raw_dir = UPath(tmp_path)

        add_published_layers(adata, "gdsc", raw_dir, raw_dir / "model_list.csv", DRUG_MAPPING)

        assert "LN_IC50_published" not in adata.layers


class TestBeatAml:
    def test_the_layers_are_filled_by_drug_id(self, tmp_path) -> None:
        raw_dir = UPath(tmp_path)
        pd.DataFrame({"dbgap_subject_id": ["P1"], "inhibitor": ["1001"], "auc": [0.4], "ic50": [3.0]}).to_csv(
            str(raw_dir / "beataml_probit_curve_fits.txt"), sep="\t", index=False
        )
        adata = ad.AnnData(
            X=np.zeros((1, 2), dtype=np.float32),
            obs=pd.DataFrame(index=pd.Index(["P1"])),
            var=pd.DataFrame(index=pd.Index(["31703", "GDSC_1002"])),
        )

        add_published_layers(adata, "beataml", raw_dir, raw_dir / "model_list.csv", DRUG_MAPPING)

        assert adata.layers["AUC_published"][0, 0] == pytest.approx(0.4)
        assert np.isnan(adata.layers["AUC_published"][0, 1])
