"""Tests that the CMP process stages read the shared download and stay deterministic."""

import shutil
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from upath import UPath

from data_curation.cell_lines.download_model_list import cmp_model_list_path
from data_curation.core import layout
from data_curation.core.errors import CurationError
from data_curation.omics import process_cnv, process_mutations, process_proteomics, process_transcriptomics

SOURCES = Path(__file__).parents[2] / "config" / "sources.yaml"

MODELS = [
    {"model_id": "SIDM00001", "model_name": "A549", "RRID": "CVCL_0023"},
    {"model_id": "SIDM00002", "model_name": "MEC-1", "RRID": "CVCL_1870"},
]


@pytest.fixture
def workdir(monkeypatch, tmp_path, write_model_list) -> UPath:
    """A scratch repository root with the real sources config and a model list at the shared CMP location."""
    (tmp_path / "config").mkdir()
    shutil.copy(SOURCES, tmp_path / "config" / "sources.yaml")
    monkeypatch.chdir(tmp_path)
    model_list = write_model_list(MODELS, name="staged_model_list.csv")
    target = UPath(cmp_model_list_path())
    target.parent.mkdir(parents=True)
    target.write_bytes(model_list.read_bytes())
    return UPath(tmp_path)


def _cmp_file(kind: str, name: str) -> UPath:
    path = layout.cmp_raw(kind) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _write_cnv(path: UPath, genes: list[str]) -> None:
    rows = [
        ["", "", "SIDM00001", "SIDM00002", "SIDM99999"],
        ["", "", "A549", "MEC-1", "other"],
        ["gene_id", "symbol", "", "", ""],
        *([f"SIDG{i}", gene, "0", "1", "2"] for i, gene in enumerate(genes)),
    ]
    pd.DataFrame(rows).to_csv(str(path), index=False, header=False)


class TestCnv:
    def test_it_reads_the_shared_download_for_any_cmp_dataset(self, workdir) -> None:
        _write_cnv(_cmp_file("cnv", "cnv_gistic_20191101.csv"), ["TP53", "KRAS"])

        process_cnv.main("GDSC1")
        process_cnv.main("CCLE")

        first = ad.read_h5ad(str(layout.omics("cnv", "GDSC1")))
        second = ad.read_h5ad(str(layout.omics("cnv", "CCLE")))
        assert list(first.obs_names) == ["CVCL_0023", "CVCL_1870"]
        assert list(first.var_names) == ["TP53", "KRAS"]
        assert first.to_df().equals(second.to_df())
        assert not layout.raw_dir("cnv", "GDSC1").exists()

    def test_the_file_choice_is_deterministic_when_several_match(self, workdir) -> None:
        _write_cnv(_cmp_file("cnv", "cnv_gistic_b.csv"), ["BRCA1"])
        _write_cnv(_cmp_file("cnv", "cnv_gistic_a.csv"), ["TP53"])

        assert process_cnv._find_cnv_file(layout.cmp_raw("cnv")).name == "cnv_gistic_a.csv"

    def test_a_missing_download_is_reported_with_the_directory_contents(self, workdir) -> None:
        _cmp_file("cnv", "unrelated.txt").write_text("x")

        with pytest.raises(FileNotFoundError, match="unrelated.txt"):
            process_cnv._find_cnv_file(layout.cmp_raw("cnv"))


class TestMutations:
    def test_it_builds_a_binary_matrix_from_the_shared_download(self, workdir) -> None:
        pd.DataFrame(
            {
                "model_id": ["SIDM00001", "SIDM00001", "SIDM00002", "SIDM00002", "SIDM99999"],
                "gene_symbol": ["TP53", "KRAS", "TP53", "BRAF", "TP53"],
                "coding": ["t", "t", "t", "t", "t"],
                "effect": ["missense", "silent", "missense", "nonsense", "missense"],
            }
        ).to_csv(str(_cmp_file("mutations", "mutations_all_20260724.csv")), index=False)

        process_mutations.main("GDSC2")

        adata = ad.read_h5ad(str(layout.omics("mutations", "GDSC2")))
        frame = pd.DataFrame(adata.X, index=adata.obs_names, columns=adata.var_names)
        assert frame.loc["CVCL_0023", "TP53"] == 1
        assert "KRAS" not in frame.columns
        assert frame.loc["CVCL_1870", "BRAF"] == 1
        assert "SIDM99999" not in adata.obs_names

    def test_the_file_choice_is_deterministic_when_several_match(self, workdir) -> None:
        for name in ("mutations_all_2.csv", "mutations_all_1.csv"):
            _cmp_file("mutations", name).write_text("x")

        assert process_mutations._find_mutations_file(layout.cmp_raw("mutations")).name == "mutations_all_1.csv"

    def test_the_fallback_pattern_still_applies(self, workdir) -> None:
        _cmp_file("mutations", "my_mutation_dump.csv").write_text("x")

        assert process_mutations._find_mutations_file(layout.cmp_raw("mutations")).name == "my_mutation_dump.csv"


class TestProteomics:
    def test_it_pivots_the_shared_download(self, workdir) -> None:
        pd.DataFrame(
            {
                "model_id": ["SIDM00001", "SIDM00001", "SIDM00002"],
                "symbol": ["TP53", "KRAS", "TP53"],
                "protein_intensity": [1.0, 2.0, 3.0],
            }
        ).to_csv(str(_cmp_file("proteomics", "abc-proteomics_latest.csv.gz")), index=False)

        process_proteomics.main("CTRPv2")

        adata = ad.read_h5ad(str(layout.omics("proteomics", "CTRPv2")))
        assert adata.shape == (2, 2)
        assert adata.to_df().loc["CVCL_1870", "TP53"] == pytest.approx(3.0)

    def test_the_file_choice_is_deterministic_when_several_match(self, workdir) -> None:
        for name in ("z-proteomics.csv", "a-proteomics.csv"):
            _cmp_file("proteomics", name).write_text("x")

        assert process_proteomics._find_proteomics_file(layout.cmp_raw("proteomics")).name == "a-proteomics.csv"


class TestTranscriptomics:
    def test_a_cmp_dataset_reads_the_shared_rnaseq(self, workdir, monkeypatch) -> None:
        seen: list[UPath] = []

        def fake(raw_dir: UPath, model_list: UPath) -> ad.AnnData:
            seen.append(raw_dir)
            return ad.AnnData(X=np.ones((1, 1), dtype=np.float32))

        monkeypatch.setattr(process_transcriptomics, "_process_cmp_rnaseq", fake)

        process_transcriptomics.main("GDSC1")

        assert seen == [layout.cmp_raw("rnaseq")]
        assert ad.read_h5ad(str(layout.omics("transcriptomics", "GDSC1"))).uns["expression_unit"] == "log2_tpm_plus_1"

    def test_another_dataset_reads_its_own_raw_directory(self, workdir, monkeypatch) -> None:
        seen: list[UPath] = []

        def fake(raw_dir: UPath) -> ad.AnnData:
            seen.append(raw_dir)
            return ad.AnnData(X=np.ones((1, 1), dtype=np.float32))

        monkeypatch.setitem(process_transcriptomics.PROCESSORS, "BeatAML2", fake)

        process_transcriptomics.main("BeatAML2")

        assert seen == [layout.raw_dir("transcriptomics", "BeatAML2")]

    def test_an_unknown_dataset_is_rejected(self, workdir) -> None:
        with pytest.raises(CurationError, match="No transcriptomics processor"):
            process_transcriptomics.main("NOPE")

    def test_the_matrix_files_are_picked_in_sorted_order(self, workdir) -> None:
        """Both the TPM and the count file are chosen by the shared, deterministic finder."""
        for name in ("rnaseq_merged_rsem_tpm_2.csv", "rnaseq_merged_rsem_tpm_1.csv"):
            _cmp_file("rnaseq", name).write_text("x")
        from data_curation.core.paths import find_first

        assert find_first(layout.cmp_raw("rnaseq"), ("*rsem_tpm*", "*tpm*")).name == "rnaseq_merged_rsem_tpm_1.csv"
