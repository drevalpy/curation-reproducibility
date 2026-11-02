"""Tests for MuData assembly, focused on the global obs frame."""

import shutil
from pathlib import Path

import anndata as ad
import mudata as md
import numpy as np
import pandas as pd
import pytest
from drevalpy.curation._replicate_fits import ID_COLUMNS, METRICS, to_schema
from drevalpy.data._replicate_fits import KEY

from data_curation.assembly.assemble_mudata import (
    _build_global_obs,
    _subset_to_screened,
    main,
)
from data_curation.cell_lines.meta import CMP_META_COLUMNS
from data_curation.core import layout
from data_curation.core.errors import CurationError

SOURCES = Path(__file__).parents[2] / "config" / "sources.yaml"

md.set_options(pull_on_update=False)


def _anndata(obs_names: list[str], n_vars: int = 3, obs: pd.DataFrame | None = None) -> ad.AnnData:
    index = pd.Index(obs_names, name=layout.SAMPLE_INDEX_NAME)
    return ad.AnnData(
        X=np.zeros((len(obs_names), n_vars), dtype=np.float32),
        obs=pd.DataFrame(index=index) if obs is None else obs.set_axis(index),
        var=pd.DataFrame(index=pd.Index([f"f{i}" for i in range(n_vars)])),
    )


def _cell_line_row(accession: str, name: str, tissue: str, **extra) -> dict:
    return {"sample_id": accession, "cellosaurus_id": accession, "cell_line_name": name, "tissue": tissue} | extra


@pytest.fixture
def meta_parquet(tmp_path: Path):
    """Factory writing a metadata Parquet file; cell-line tables get the real CMP column set."""

    def _write(rows: list[dict], *, cell_lines: bool = True) -> Path:
        frame = pd.DataFrame(rows)
        columns = CMP_META_COLUMNS if cell_lines else list(frame.columns)
        for column in columns:
            if column not in frame.columns:
                frame[column] = pd.NA
        path = tmp_path / "meta.parquet"
        frame[columns].to_parquet(path, index=False)
        return path

    return _write


class TestBuildGlobalObs:
    def test_metadata_is_joined_onto_the_union_index(self, meta_parquet) -> None:
        mdata = md.MuData({"response": _anndata(["CVCL_1", "CVCL_2"])})
        path = meta_parquet([_cell_line_row("CVCL_1", "A", "Lung"), _cell_line_row("CVCL_2", "B", "Skin")])
        _build_global_obs(mdata, path)
        assert list(mdata.obs["tissue"]) == ["Lung", "Skin"]
        assert list(mdata.obs.columns) == [c for c in CMP_META_COLUMNS if c != "sample_id"]

    def test_the_global_index_is_the_sample_id(self, meta_parquet) -> None:
        mdata = md.MuData({"response": _anndata(["CVCL_1"])})
        _build_global_obs(mdata, meta_parquet([_cell_line_row("CVCL_1", "A", "Lung")]))
        assert mdata.obs.index.name == layout.SAMPLE_INDEX_NAME

    def test_a_cell_line_keeps_its_accession_and_name_as_columns(self, meta_parquet) -> None:
        mdata = md.MuData({"response": _anndata(["CVCL_1"])})
        _build_global_obs(mdata, meta_parquet([_cell_line_row("CVCL_1", "A", "Lung")]))
        assert mdata.obs.loc["CVCL_1", "cellosaurus_id"] == "CVCL_1"
        assert mdata.obs.loc["CVCL_1", "cell_line_name"] == "A"

    def test_samples_that_are_not_cell_lines_get_tissue_only(self, meta_parquet, tmp_path: Path) -> None:
        """The PDX_Bruna shape: no accession and no cell line name exist, so none is written."""
        mdata = md.MuData({"response": _anndata(["HCI001-X4C"])})
        path = meta_parquet([{"sample_id": "HCI001-X4C", "tissue": "Breast"}], cell_lines=False)
        _build_global_obs(mdata, path)

        assert list(mdata.obs.columns) == ["tissue"]
        out = tmp_path / "out.h5mu"
        mdata.write(out)
        reread = md.read_h5mu(out)
        assert reread.obs.loc["HCI001-X4C", "tissue"] == "Breast"
        assert reread.obs.index.name == layout.SAMPLE_INDEX_NAME

    def test_a_cell_line_dataset_survives_the_write_with_unset_columns(self, meta_parquet, tmp_path: Path) -> None:
        """A cell line missing from the table leaves every CMP column unset.

        h5py cannot write an all-NaN object column as strings, so a plain reindex
        produced `TypeError: Can't implicitly convert non-string objects to
        strings` when writing `cancer_type`.
        """
        mdata = md.MuData({"response": _anndata(["CVCL_UNKNOWN"])})
        _build_global_obs(mdata, meta_parquet([_cell_line_row("CVCL_1", "A", "Lung")]))

        assert mdata.obs.loc["CVCL_UNKNOWN", "cancer_type"] == ""
        out = tmp_path / "out.h5mu"
        mdata.write(out)
        assert md.read_h5mu(out).obs.loc["CVCL_UNKNOWN", "tissue"] == ""

    def test_problematic_round_trips_as_a_boolean(self, meta_parquet) -> None:
        mdata = md.MuData({"response": _anndata(["CVCL_1", "CVCL_2"])})
        path = meta_parquet(
            [
                _cell_line_row("CVCL_1", "A", "Lung", problematic=True),
                _cell_line_row("CVCL_2", "B", "Skin", problematic=False),
            ]
        )
        _build_global_obs(mdata, path)
        assert mdata.obs["problematic"].dtype == bool
        assert list(mdata.obs["problematic"]) == [True, False]

    def test_an_accession_absent_from_the_table_falls_back_to_its_own_id(self, meta_parquet) -> None:
        mdata = md.MuData({"response": _anndata(["CVCL_1", "CVCL_UNKNOWN"])})
        _build_global_obs(mdata, meta_parquet([_cell_line_row("CVCL_1", "A", "Lung")]))
        assert mdata.obs.loc["CVCL_UNKNOWN", "cell_line_name"] == "CVCL_UNKNOWN"
        assert mdata.obs.loc["CVCL_UNKNOWN", "cellosaurus_id"] == "CVCL_UNKNOWN"
        assert mdata.obs.loc["CVCL_UNKNOWN", "tissue"] == ""

    def test_an_absent_accession_takes_its_name_from_the_response_modality(self, meta_parquet) -> None:
        """Cellosaurus-rescued lines have no CMP entry, but the screen named them."""
        response = _anndata(["CVCL_1", "CVCL_WW42"], obs=pd.DataFrame({"cell_line_name": ["A", "BT-159"]}))
        mdata = md.MuData({"response": response})
        _build_global_obs(mdata, meta_parquet([_cell_line_row("CVCL_1", "A", "Lung")]))
        assert mdata.obs.loc["CVCL_WW42", "cell_line_name"] == "BT-159"
        assert "cell_line_name" not in mdata.mod["response"].obs.columns

    def test_a_stale_table_without_a_sample_id_is_rejected(self, meta_parquet) -> None:
        mdata = md.MuData({"response": _anndata(["CVCL_1"])})
        path = meta_parquet([{"cellosaurus_id": "CVCL_1", "tissue": "Lung"}], cell_lines=False)
        with pytest.raises(CurationError, match="sample_id"):
            _build_global_obs(mdata, path)

    def test_a_table_with_only_one_cell_line_column_is_rejected(self, meta_parquet) -> None:
        mdata = md.MuData({"response": _anndata(["CVCL_1"])})
        path = meta_parquet([{"sample_id": "CVCL_1", "cell_line_name": "A", "tissue": "Lung"}], cell_lines=False)
        with pytest.raises(CurationError, match="both of"):
            _build_global_obs(mdata, path)


class TestSubsetToScreened:
    def test_omics_are_restricted_to_screened_samples(self) -> None:
        modalities = {
            "response": _anndata(["CVCL_1", "CVCL_2"]),
            "gene_expression": _anndata(["CVCL_1", "CVCL_2", "CVCL_3"]),
        }
        subset = _subset_to_screened(modalities)
        assert list(subset["gene_expression"].obs_names) == ["CVCL_1", "CVCL_2"]

    def test_response_is_never_subset(self) -> None:
        modalities = {
            "response": _anndata(["CVCL_1", "CVCL_9"]),
            "gene_expression": _anndata(["CVCL_1"]),
        }
        subset = _subset_to_screened(modalities)
        assert list(subset["response"].obs_names) == ["CVCL_1", "CVCL_9"]

    def test_without_a_response_nothing_changes(self) -> None:
        modalities = {"gene_expression": _anndata(["CVCL_1", "CVCL_2"])}
        assert list(_subset_to_screened(modalities)["gene_expression"].obs_names) == ["CVCL_1", "CVCL_2"]


class TestMainReadsTheRightMetadata:
    @pytest.fixture
    def workdir(self, monkeypatch, tmp_path: Path) -> Path:
        (tmp_path / "config").mkdir()
        shutil.copy(SOURCES, tmp_path / "config" / "sources.yaml")
        monkeypatch.chdir(tmp_path)
        return tmp_path

    @staticmethod
    def _write_meta(path, sample_id: str, tissue: str, *, cell_line: bool) -> None:
        columns = CMP_META_COLUMNS if cell_line else ["sample_id", "tissue"]
        frame = pd.DataFrame([_cell_line_row(sample_id, "X", tissue)])
        for column in columns:
            if column not in frame.columns:
                frame[column] = pd.NA
        path.parent.mkdir(parents=True, exist_ok=True)
        frame[columns].to_parquet(str(path), index=False)

    @staticmethod
    def _write_response(dataset: str, accession: str) -> None:
        path = layout.harmonized(dataset)
        path.parent.mkdir(parents=True, exist_ok=True)
        _anndata([accession]).write_h5ad(str(path))

    def test_a_cmp_dataset_reads_the_shared_table(self, workdir: Path) -> None:
        self._write_meta(layout.cmp_sample_meta(), "CVCL_1", "Lung", cell_line=True)
        self._write_response("GDSC2", "CVCL_1")

        main("GDSC2")

        obs = md.read_h5mu(str(layout.assembled("GDSC2"))).obs
        assert obs.loc["CVCL_1", "tissue"] == "Lung"
        assert obs.loc["CVCL_1", "cellosaurus_id"] == "CVCL_1"

    def test_a_constant_tissue_dataset_reads_its_own_table(self, workdir: Path) -> None:
        self._write_meta(layout.cmp_sample_meta(), "PDX1", "Lung", cell_line=True)
        self._write_meta(layout.sample_meta("PDX_Bruna"), "PDX1", "Breast", cell_line=False)
        self._write_response("PDX_Bruna", "PDX1")

        main("PDX_Bruna")

        obs = md.read_h5mu(str(layout.assembled("PDX_Bruna"))).obs
        assert obs.loc["PDX1", "tissue"] == "Breast"
        assert list(obs.columns) == ["tissue"]


def _replicate_fits(pairs: list[tuple[str, str]]) -> pd.DataFrame:
    """A schema-typed replicate-fits table with two series per (sample, drug) pair."""
    rows = [pair for pair in pairs for _ in range(2)]
    table = pd.DataFrame({column: pd.Series(["x"] * len(rows), dtype=object) for column in ID_COLUMNS})
    table["sample_id"] = [sample for sample, _ in rows]
    table["pubchem_id"] = [drug for _, drug in rows]
    table["replicate"] = [0, 1] * len(pairs)
    table["biological_replicate"] = 0
    table["technical_replicate"] = [0, 1] * len(pairs)
    table["n_doses"] = 8
    table["status"] = "fitted"
    table["experiment"] = None
    for metric in METRICS:
        table[metric] = np.linspace(0, 1, len(rows))
    table["screen_medium"] = [None, "RPMI"] * len(pairs)
    return to_schema(table)


class TestMainAttachesReplicateFits:
    @pytest.fixture
    def workdir(self, monkeypatch, tmp_path: Path) -> Path:
        (tmp_path / "config").mkdir()
        shutil.copy(SOURCES, tmp_path / "config" / "sources.yaml")
        monkeypatch.chdir(tmp_path)
        TestMainReadsTheRightMetadata._write_meta(layout.cmp_sample_meta(), "CVCL_1", "Lung", cell_line=True)
        path = layout.harmonized("GDSC2")
        path.parent.mkdir(parents=True, exist_ok=True)
        _anndata(["CVCL_1", "CVCL_2"], n_vars=2).write_h5ad(str(path))
        return tmp_path

    @staticmethod
    def _write_fits(table: pd.DataFrame) -> None:
        path = layout.harmonized("GDSC2")
        response = ad.read_h5ad(str(path))
        response.uns[KEY] = table
        response.write_h5ad(str(path))

    def test_the_table_is_restricted_to_the_response_axes_and_round_trips(self, workdir: Path) -> None:
        table = _replicate_fits([("CVCL_1", "f0"), ("CVCL_2", "f1"), ("CVCL_9", "f0"), ("CVCL_1", "f7")])
        self._write_fits(table)

        main("GDSC2")

        mdata = md.read_h5mu(str(layout.assembled("GDSC2")))
        fits = mdata.mod["response"].uns[KEY]
        expected = table.iloc[:4].reset_index(drop=True)
        expected["sample_id"] = expected["sample_id"].cat.remove_unused_categories()
        expected["pubchem_id"] = expected["pubchem_id"].cat.remove_unused_categories()
        pd.testing.assert_frame_equal(fits, expected)
        assert fits["replicate"].dtype == np.int32
        assert fits["pEC50"].dtype == np.float32
        assert fits["experiment"].isna().all()

    def test_no_random_replicate_modalities_are_assembled(self, workdir: Path) -> None:
        self._write_fits(_replicate_fits([("CVCL_1", "f0")]))

        main("GDSC2")

        assert list(md.read_h5mu(str(layout.assembled("GDSC2"))).mod) == ["response"]

    def test_a_dataset_without_replicate_fits_has_no_table(self, workdir: Path) -> None:
        main("GDSC2")

        assert KEY not in md.read_h5mu(str(layout.assembled("GDSC2"))).mod["response"].uns
