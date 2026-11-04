"""Tests for the shared CMP sample metadata table and the per-dataset constant-tissue ones."""

import shutil
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from upath import UPath

from data_curation.cell_lines import process_sample_meta
from data_curation.cell_lines.download_model_list import cmp_model_list_path
from data_curation.cell_lines.meta import CMP_META_COLUMNS, sample_meta_path
from data_curation.core import layout
from data_curation.core.errors import CurationError

SOURCES = Path(__file__).parents[2] / "config" / "sources.yaml"


@pytest.fixture
def workdir(monkeypatch, tmp_path, write_model_list) -> UPath:
    """A scratch repository root with the sources config, a model list and a Cellosaurus table."""
    (tmp_path / "config").mkdir()
    shutil.copy(SOURCES, tmp_path / "config" / "sources.yaml")
    monkeypatch.chdir(tmp_path)

    model_list = write_model_list(
        [
            {"model_id": "SIDM00001", "model_name": "A549", "RRID": "CVCL_0023", "tissue": "Lung"},
            {
                "model_id": "SIDM00002",
                "model_name": "MEC-1",
                "RRID": "CVCL_1870",
                "tissue": "Haematopoietic and Lymphoid",
            },
        ],
        name="staged.csv",
    )
    target = UPath(cmp_model_list_path())
    target.parent.mkdir(parents=True)
    target.write_bytes(model_list.read_bytes())

    layout.CELLOSAURUS.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "cellosaurus_id": ["CVCL_0023"],
            "name": ["A549"],
            "synonyms": [""],
            "problematic": [False],
            "problem_description": [""],
            "discontinued": [""],
            "misspelling": [""],
        }
    ).to_parquet(str(layout.CELLOSAURUS), index=False)
    return UPath(tmp_path)


class TestMetaPath:
    @pytest.mark.parametrize("dataset", ["CCLE", "CTRPv1", "CTRPv2", "GDSC1", "GDSC2", "PRISM_19Q4", "PRISM_20Q2"])
    def test_cmp_metadata_datasets_share_one_table(self, dataset: str) -> None:
        assert sample_meta_path(dataset) == layout.cmp_sample_meta()

    @pytest.mark.parametrize("dataset", ["PDX_Bruna", "BeatAML2"])
    def test_constant_tissue_datasets_keep_their_own_table(self, dataset: str) -> None:
        assert sample_meta_path(dataset) == layout.sample_meta(dataset)


class TestMainCmp:
    def test_it_writes_the_shared_table_from_the_model_list(self, workdir) -> None:
        process_sample_meta.main_cmp()

        table = pd.read_parquet(str(layout.cmp_sample_meta()))
        assert list(table.columns) == CMP_META_COLUMNS
        assert dict(zip(table["sample_id"], table["tissue"], strict=True)) == {
            "CVCL_0023": "Lung",
            "CVCL_1870": "Haematopoietic and Lymphoid",
        }

    def test_the_sample_id_of_a_cell_line_is_its_accession(self, workdir) -> None:
        process_sample_meta.main_cmp()

        table = pd.read_parquet(str(layout.cmp_sample_meta()))
        assert table["sample_id"].tolist() == table["cellosaurus_id"].tolist()
        assert dict(zip(table["sample_id"], table["cell_line_name"], strict=True)) == {
            "CVCL_0023": "A549",
            "CVCL_1870": "MEC-1",
        }

    def test_it_writes_nothing_per_dataset(self, workdir) -> None:
        process_sample_meta.main_cmp()

        assert [p.name for p in layout.SAMPLE_META_DIR.iterdir()] == ["cmp.parquet"]


class TestMain:
    def test_a_cmp_dataset_is_refused_because_it_has_no_table_of_its_own(self, workdir) -> None:
        with pytest.raises(CurationError, match="shared CMP table"):
            process_sample_meta.main("GDSC2")

        assert not layout.sample_meta("GDSC2").exists()

    def test_a_constant_tissue_dataset_gets_its_own_table(self, workdir) -> None:
        transcriptomics = layout.modality("transcriptomics", "PDX_Bruna")
        transcriptomics.parent.mkdir(parents=True)
        ad.AnnData(
            X=np.zeros((1, 1), dtype=np.float32),
            obs=pd.DataFrame(index=pd.Index(["HCI001-X4C"])),
        ).write_h5ad(str(transcriptomics))

        process_sample_meta.main("PDX_Bruna")

        table = pd.read_parquet(str(layout.sample_meta("PDX_Bruna")))
        assert list(table.columns) == ["sample_id", "tissue"]
        assert list(table["sample_id"]) == ["HCI001-X4C"]
        assert list(table["tissue"]) == ["Breast"]


MINIMAL_SOURCES = """\
datasets:
  Inhouse:
    sample_metadata: standard
  Specimens:
    tissue: Blood
  Screen: {}
"""


@pytest.fixture
def minimal_workdir(monkeypatch, tmp_path) -> UPath:
    """A scratch repository root whose sources config holds one dataset per metadata route."""
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "sources.yaml").write_text(MINIMAL_SOURCES)
    monkeypatch.chdir(tmp_path)
    return UPath(tmp_path)


def _write_samples(frame: pd.DataFrame) -> None:
    raw_dir = layout.raw_dir("response", "Inhouse")
    raw_dir.mkdir(parents=True)
    frame.to_parquet(str(raw_dir / "samples.parquet"), index=False)


class TestStandardSampleMetadata:
    def test_a_standard_dataset_keeps_its_own_table(self, minimal_workdir) -> None:
        assert sample_meta_path("Inhouse") == layout.sample_meta("Inhouse")

    def test_a_dataset_with_neither_route_reads_the_shared_table(self, minimal_workdir) -> None:
        assert sample_meta_path("Screen") == layout.cmp_sample_meta()

    def test_the_standard_table_is_written_with_its_extra_columns(self, minimal_workdir) -> None:
        _write_samples(
            pd.DataFrame(
                {
                    "sample_id": ["S1", "S2"],
                    "tissue": ["Blood", "Bone Marrow"],
                    "patient_id": [7.0, None],
                    "sex": ["F", "M"],
                }
            )
        )

        process_sample_meta.main("Inhouse")

        table = pd.read_parquet(str(layout.sample_meta("Inhouse")))
        assert list(table.columns) == ["sample_id", "tissue", "patient_id", "sex"]
        assert table.to_dict("list") == {
            "sample_id": ["S1", "S2"],
            "tissue": ["Blood", "Bone Marrow"],
            "patient_id": ["7", ""],
            "sex": ["F", "M"],
        }

    def test_a_missing_samples_file_is_refused(self, minimal_workdir) -> None:
        with pytest.raises(CurationError, match=r"samples\.parquet is missing"):
            process_sample_meta.main("Inhouse")

        assert not layout.sample_meta("Inhouse").exists()

    def test_a_broken_samples_file_is_refused(self, minimal_workdir) -> None:
        _write_samples(pd.DataFrame({"sample_id": ["S1", "S1"], "tissue": ["Blood", "Blood"]}))

        with pytest.raises(CurationError, match="repeat"):
            process_sample_meta.main("Inhouse")

    def test_the_constant_tissue_route_is_unchanged(self, minimal_workdir) -> None:
        response = layout.modality("response", "Specimens")
        response.parent.mkdir(parents=True)
        ad.AnnData(
            X=np.zeros((2, 1), dtype=np.float32),
            obs=pd.DataFrame(index=pd.Index(["P1", "P2"])),
        ).write_h5ad(str(response))

        process_sample_meta.main("Specimens")

        table = pd.read_parquet(str(layout.sample_meta("Specimens")))
        assert table.to_dict("list") == {"sample_id": ["P1", "P2"], "tissue": ["Blood", "Blood"]}

    def test_a_dataset_with_neither_route_is_refused_naming_both(self, minimal_workdir) -> None:
        with pytest.raises(CurationError, match="sample_metadata: standard") as raised:
            process_sample_meta.main("Screen")

        assert "constant tissue" in str(raised.value)
        assert not layout.sample_meta("Screen").exists()
