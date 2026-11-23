"""Tests for the GDSC and CCLE transcriptomics processors, which read each dataset's own expression files."""

import gzip

import numpy as np
import pandas as pd
import pytest
from upath import UPath

from data_curation.core.errors import CurationError
from data_curation.omics import COUNT_LAYER
from data_curation.omics.transcriptomics.ccle import process_ccle_transcriptomics
from data_curation.omics.transcriptomics.gdsc import process_gdsc_transcriptomics

MODELS = [
    {"model_id": "SIDM00001", "model_name": "A549", "RRID": "CVCL_0023", "COSMIC_ID": 905949, "CCLE_ID": "A549_LUNG"},
    {
        "model_id": "SIDM00002",
        "model_name": "MEC-1",
        "RRID": "CVCL_1870",
        "COSMIC_ID": 1331031,
        "CCLE_ID": "MEC1_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE",
    },
]
MODEL_COLUMNS = ["model_id", "model_name", "RRID", "tissue", "synonyms", "COSMIC_ID", "CCLE_ID"]


@pytest.fixture
def model_list(write_model_list) -> UPath:
    return write_model_list(MODELS, columns=MODEL_COLUMNS)


@pytest.fixture
def cellosaurus(tmp_dir: UPath) -> UPath:
    """An empty Cellosaurus table: the processors under test only need it for the name fallback."""
    columns = ["name", "cellosaurus_id", "synonyms", "problem_description", "discontinued", "misspelling"]
    frame = pd.DataFrame({column: pd.Series(dtype=str) for column in columns})
    frame["problematic"] = pd.Series(dtype=bool)
    path = tmp_dir / "cellosaurus.parquet"
    frame.to_parquet(str(path), index=False)
    return path


#: Cell Model Passports gene annotation of the fixtures: symbol, Ensembl ID, HGNC accession.
ANNOTATION = [
    ("TP53", "ENSG1", "HGNC:1"),
    ("KRAS", "ENSG2", "HGNC:2"),
    ("EGFR", "ENSG3", "HGNC:3"),
    ("AARS1", "ENSG4", "HGNC:4"),
    ("ICAM4", "ENSG5", "HGNC:5"),
]


@pytest.fixture
def gene_identifiers(tmp_dir: UPath) -> UPath:
    path = tmp_dir / "gene_identifiers.csv"
    pd.DataFrame(ANNOTATION, columns=pd.Index(["hgnc_symbol", "ensembl_gene_id", "hgnc_id"])).to_csv(
        str(path), index=False
    )
    return path


@pytest.fixture
def raw_dir(tmp_dir: UPath) -> UPath:
    path = tmp_dir / "raw"
    path.mkdir()
    return path


class TestGdsc:
    @staticmethod
    def _write(raw_dir: UPath, rows: list[list], samples: list[str]) -> None:
        header = ["GENE_SYMBOLS", "GENE_title", *(f"DATA.{sample}" for sample in samples)]
        pd.DataFrame(rows, columns=pd.Index(header)).to_csv(
            str(raw_dir / "Cell_line_RMA_proc_basalExp.txt"), sep="\t", index=False
        )

    def test_samples_resolve_by_cosmic_id_and_values_stay_as_they_are(
        self, raw_dir, model_list, gene_identifiers
    ) -> None:
        self._write(
            raw_dir, [["TP53", "tumor protein", 5.5, 6.5], ["KRAS", "KRAS proto", 8.0, 9.0]], ["905949", "1331031"]
        )

        adata = process_gdsc_transcriptomics(raw_dir, model_list, gene_identifiers)

        assert list(adata.obs_names) == ["CVCL_0023", "CVCL_1870"]
        assert list(adata.obs["cell_line_name"]) == ["A549", "MEC-1"]
        assert list(adata.var_names) == ["TP53", "KRAS"]
        assert adata.var_names.name == "gene_symbol"
        assert np.asarray(adata.X).tolist() == [[5.5, 8.0], [6.5, 9.0]]
        assert not adata.layers

    def test_unresolved_cosmic_ids_are_dropped(self, raw_dir, model_list, gene_identifiers) -> None:
        self._write(raw_dir, [["TP53", "t", 5.5, 6.5, 7.5]], ["905949", "42", "1331031"])

        adata = process_gdsc_transcriptomics(raw_dir, model_list, gene_identifiers)

        assert list(adata.obs_names) == ["CVCL_0023", "CVCL_1870"]
        assert np.asarray(adata.X).tolist() == [[5.5], [7.5]]

    def test_rows_without_a_symbol_and_genes_with_missing_values_are_dropped(
        self, raw_dir, model_list, gene_identifiers
    ) -> None:
        rows = [
            ["TP53", "t", 5.5, 6.5],
            [np.nan, "no symbol", 1.0, 2.0],
            ["KRAS", "k", np.nan, 9.0],
            ["EGFR", "e", 3.0, 4.0],
        ]
        self._write(raw_dir, rows, ["905949", "1331031"])

        adata = process_gdsc_transcriptomics(raw_dir, model_list, gene_identifiers)

        assert list(adata.var_names) == ["TP53", "EGFR"]
        assert np.asarray(adata.X).tolist() == [[5.5, 3.0], [6.5, 4.0]]

    def test_a_repeated_symbol_keeps_its_first_row(self, raw_dir, model_list, gene_identifiers) -> None:
        self._write(raw_dir, [["TP53", "a", 1.0, 2.0], ["TP53", "b", 3.0, 4.0]], ["905949", "1331031"])

        adata = process_gdsc_transcriptomics(raw_dir, model_list, gene_identifiers)

        assert list(adata.var_names) == ["TP53"]
        assert np.asarray(adata.X).tolist() == [[1.0], [2.0]]

    def test_a_file_without_sample_columns_is_rejected(self, raw_dir, model_list, gene_identifiers) -> None:
        pd.DataFrame({"GENE_SYMBOLS": ["TP53"], "GENE_title": ["t"]}).to_csv(
            str(raw_dir / "Cell_line_RMA_proc_basalExp.txt"), sep="\t", index=False
        )

        with pytest.raises(CurationError, match="not a GDSC expression matrix"):
            process_gdsc_transcriptomics(raw_dir, model_list, gene_identifiers)

    def test_a_missing_file_is_reported(self, raw_dir, model_list, gene_identifiers) -> None:
        with pytest.raises(FileNotFoundError, match="Cell_line_RMA_proc_basalExp"):
            process_gdsc_transcriptomics(raw_dir, model_list, gene_identifiers)


def _write_tpm(raw_dir: UPath, genes: list[str], samples: list[str], values: list[list[float]]) -> None:
    """Write the RSEM TPM file: Ensembl IDs, a transcript column, space-padded values."""
    lines = ["\t".join(["gene_id", "transcript_ids", *samples])]
    for gene, row in zip(genes, values, strict=True):
        cells = ["" if np.isnan(value) else f"{value:>8.2f}" for value in row]
        lines.append("\t".join([gene, "ENST1,ENST2", *cells]))
    with gzip.open(str(raw_dir / "CCLE_RNAseq_rsem_genes_tpm_20180929.txt.gz"), "wt") as handle:
        handle.write("\n".join(lines) + "\n")


def _write_gct(
    raw_dir: UPath, genes: list[str], symbols: list[str], samples: list[str], values: list[list[float]]
) -> None:
    """Write the RNA-SeQC read counts as GCT 1.2."""
    lines = ["#1.2", f"{len(genes)}\t{len(samples)}", "\t".join(["Name", "Description", *samples])]
    for gene, symbol, row in zip(genes, symbols, values, strict=True):
        cells = ["" if np.isnan(value) else str(int(value)) for value in row]
        lines.append("\t".join([gene, symbol, *cells]))
    with gzip.open(str(raw_dir / "CCLE_RNAseq_genes_counts_20180929.gct.gz"), "wt") as handle:
        handle.write("\n".join(lines) + "\n")


A549, MEC1 = "A549_LUNG", "MEC1_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE"


class TestGdscSymbols:
    def test_an_out_of_date_symbol_takes_the_annotation_symbol_of_its_hgnc_accession(
        self, raw_dir, model_list, gene_identifiers
    ) -> None:
        rows = [["AARS", "alanyl-tRNA synthetase [Source:HGNC Symbol;Acc:4]", 1.0], ["TP53", "t [Acc:1]", 2.0]]
        TestGdsc._write(raw_dir, rows, ["905949"])

        adata = process_gdsc_transcriptomics(raw_dir, model_list, gene_identifiers)

        assert list(adata.var_names) == ["AARS1", "TP53"]


class TestCcle:
    def test_x_is_log2_tpm_and_the_counts_layer_holds_the_read_counts(
        self, raw_dir, model_list, cellosaurus, gene_identifiers
    ) -> None:
        _write_tpm(raw_dir, ["ENSG1.5", "ENSG2.3"], [A549, MEC1], [[0.0, 3.0], [7.0, 15.0]])
        _write_gct(raw_dir, ["ENSG1.5", "ENSG2.3"], ["TP53", "KRAS"], [A549, MEC1], [[10, 20], [30, 40]])

        adata = process_ccle_transcriptomics(raw_dir, model_list, cellosaurus, gene_identifiers)

        assert list(adata.obs_names) == ["CVCL_0023", "CVCL_1870"]
        assert list(adata.obs["cell_line_name"]) == ["A549", "MEC-1"]
        assert list(adata.var_names) == ["TP53", "KRAS"]
        assert adata.var_names.name == "gene_symbol"
        np.testing.assert_allclose(np.asarray(adata.X), [[0.0, 3.0], [2.0, 4.0]])
        assert np.asarray(adata.layers[COUNT_LAYER]).tolist() == [[10.0, 30.0], [20.0, 40.0]]
        assert adata.uns["layer_units"] == {COUNT_LAYER: "rnaseqc_read_count"}

    def test_unannotated_genes_the_counts_file_lacks_are_dropped(
        self, raw_dir, model_list, cellosaurus, gene_identifiers
    ) -> None:
        _write_tpm(raw_dir, ["ENSG1.5", "ENSG9.1", "ENSG2.3"], [A549], [[1.0], [2.0], [3.0]])
        _write_gct(raw_dir, ["ENSG2.3", "ENSG1.5"], ["KRAS", "TP53"], [A549], [[4], [5]])

        adata = process_ccle_transcriptomics(raw_dir, model_list, cellosaurus, gene_identifiers)

        # TPM file order, restricted to the shared genes; counts follow the same genes.
        assert list(adata.var_names) == ["TP53", "KRAS"]
        np.testing.assert_allclose(np.asarray(adata.X), [[1.0, 2.0]])
        assert np.asarray(adata.layers[COUNT_LAYER]).tolist() == [[5.0, 4.0]]

    def test_cell_lines_missing_from_one_file_are_dropped(
        self, raw_dir, model_list, cellosaurus, gene_identifiers
    ) -> None:
        _write_tpm(raw_dir, ["ENSG1.5"], [A549, MEC1], [[1.0, 3.0]])
        _write_gct(raw_dir, ["ENSG1.5"], ["TP53"], [MEC1], [[8]])

        adata = process_ccle_transcriptomics(raw_dir, model_list, cellosaurus, gene_identifiers)

        assert list(adata.obs_names) == ["CVCL_1870"]
        np.testing.assert_allclose(np.asarray(adata.X), [[2.0]])
        assert np.asarray(adata.layers[COUNT_LAYER]).tolist() == [[8.0]]

    def test_cell_lines_without_an_accession_are_dropped(
        self, raw_dir, model_list, cellosaurus, gene_identifiers
    ) -> None:
        _write_tpm(raw_dir, ["ENSG1.5"], [A549, "GHOST_SKIN"], [[1.0, 3.0]])
        _write_gct(raw_dir, ["ENSG1.5"], ["TP53"], [A549, "GHOST_SKIN"], [[8, 9]])

        adata = process_ccle_transcriptomics(raw_dir, model_list, cellosaurus, gene_identifiers)

        assert list(adata.obs_names) == ["CVCL_0023"]
        assert np.asarray(adata.layers[COUNT_LAYER]).tolist() == [[8.0]]

    def test_a_gene_missing_a_value_in_either_matrix_is_dropped(
        self, raw_dir, model_list, cellosaurus, gene_identifiers
    ) -> None:
        genes = ["ENSG1.5", "ENSG2.3", "ENSG3.1"]
        _write_tpm(raw_dir, genes, [A549, MEC1], [[1.0, 1.0], [np.nan, 2.0], [3.0, 3.0]])
        _write_gct(raw_dir, genes, ["TP53", "KRAS", "EGFR"], [A549, MEC1], [[1, 1], [2, 2], [np.nan, 3]])

        adata = process_ccle_transcriptomics(raw_dir, model_list, cellosaurus, gene_identifiers)

        assert list(adata.var_names) == ["TP53"]
        assert adata.shape == (2, 1)

    def test_repeated_symbols_are_made_unique(self, raw_dir, model_list, cellosaurus, gene_identifiers) -> None:
        _write_tpm(raw_dir, ["ENSG8.5", "ENSG9.3"], [A549], [[1.0], [3.0]])
        _write_gct(raw_dir, ["ENSG8.5", "ENSG9.3"], ["Y_RNA", "Y_RNA"], [A549], [[1], [2]])

        with pytest.warns(UserWarning, match="not unique"):
            adata = process_ccle_transcriptomics(raw_dir, model_list, cellosaurus, gene_identifiers)

        assert list(adata.var_names) == ["Y_RNA", "Y_RNA-1"]

    def test_files_sharing_no_genes_are_rejected(self, raw_dir, model_list, cellosaurus, gene_identifiers) -> None:
        _write_tpm(raw_dir, ["ENSG1.5"], [A549], [[1.0]])
        _write_gct(raw_dir, ["ENSG2.3"], ["TP53"], [A549], [[1]])

        with pytest.raises(CurationError, match="share no genes"):
            process_ccle_transcriptomics(raw_dir, model_list, cellosaurus, gene_identifiers)

    def test_a_repeated_gene_id_is_rejected(self, raw_dir, model_list, cellosaurus, gene_identifiers) -> None:
        _write_tpm(raw_dir, ["ENSG1.5", "ENSG1.5"], [A549], [[1.0], [2.0]])
        _write_gct(raw_dir, ["ENSG1.5"], ["TP53"], [A549], [[1]])

        with pytest.raises(CurationError, match="repeats gene IDs"):
            process_ccle_transcriptomics(raw_dir, model_list, cellosaurus, gene_identifiers)

    def test_a_counts_file_in_the_wrong_format_is_rejected(
        self, raw_dir, model_list, cellosaurus, gene_identifiers
    ) -> None:
        _write_tpm(raw_dir, ["ENSG1.5"], [A549], [[1.0]])
        with gzip.open(str(raw_dir / "CCLE_RNAseq_genes_counts_20180929.gct.gz"), "wt") as handle:
            handle.write("#1.2\n1\t1\nid\tsymbol\t" + A549 + "\nENSG1.5\tTP53\t3\n")

        with pytest.raises(CurationError, match="lacks the columns"):
            process_ccle_transcriptomics(raw_dir, model_list, cellosaurus, gene_identifiers)


class TestCcleSymbols:
    def test_an_out_of_date_symbol_takes_the_annotation_symbol_of_its_ensembl_id(
        self, raw_dir, model_list, cellosaurus, gene_identifiers
    ) -> None:
        _write_tpm(raw_dir, ["ENSG4.2", "ENSG1.5"], [A549], [[1.0], [3.0]])
        _write_gct(raw_dir, ["ENSG4.2", "ENSG1.5"], ["AARS", "TP53"], [A549], [[1], [2]])

        adata = process_ccle_transcriptomics(raw_dir, model_list, cellosaurus, gene_identifiers)

        assert list(adata.var_names) == ["AARS1", "TP53"]

    def test_a_rename_onto_a_symbol_another_gene_holds_is_not_made(
        self, raw_dir, model_list, cellosaurus, gene_identifiers
    ) -> None:
        # ENSG2 is KRAS in the annotation, but another gene of the file already is KRAS.
        _write_tpm(raw_dir, ["ENSG9.1", "ENSG2.3"], [A549], [[1.0], [3.0]])
        _write_gct(raw_dir, ["ENSG9.1", "ENSG2.3"], ["KRAS", "KRAS2"], [A549], [[1], [2]])

        adata = process_ccle_transcriptomics(raw_dir, model_list, cellosaurus, gene_identifiers)

        assert list(adata.var_names) == ["KRAS", "KRAS2"]

    def test_an_annotated_gene_the_counts_file_lacks_is_kept_without_counts(
        self, raw_dir, model_list, cellosaurus, gene_identifiers
    ) -> None:
        _write_tpm(raw_dir, ["ENSG1.5", "ENSG5.1"], [A549], [[1.0], [3.0]])
        _write_gct(raw_dir, ["ENSG1.5"], ["TP53"], [A549], [[7]])

        adata = process_ccle_transcriptomics(raw_dir, model_list, cellosaurus, gene_identifiers)

        assert list(adata.var_names) == ["TP53", "ICAM4"]
        np.testing.assert_allclose(np.asarray(adata.X), [[1.0, 2.0]])
        counts = np.asarray(adata.layers[COUNT_LAYER])
        assert counts[0, 0] == 7.0
        assert np.isnan(counts[0, 1])
