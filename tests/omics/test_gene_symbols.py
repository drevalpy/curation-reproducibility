"""Tests for renaming out-of-date gene symbols through the Cell Model Passports annotation."""

import pandas as pd
import pytest
from upath import UPath

from data_curation.core.errors import CurationError
from data_curation.omics.gene_symbols import harmonize_symbols, read_gene_annotation, symbols_by_key


class TestHarmonizeSymbols:
    def test_a_current_symbol_is_kept_even_when_its_id_maps_elsewhere(self) -> None:
        result = harmonize_symbols(pd.Series(["TP53"]), pd.Series(["OTHER"]), {"TP53", "OTHER"})

        assert result.tolist() == ["TP53"]

    def test_an_out_of_date_symbol_is_renamed(self) -> None:
        result = harmonize_symbols(pd.Series(["AARS", "TP53"]), pd.Series(["AARS1", "TP53"]), {"AARS1", "TP53"})

        assert result.tolist() == ["AARS1", "TP53"]

    def test_an_unannotated_gene_keeps_its_symbol(self) -> None:
        result = harmonize_symbols(pd.Series(["LOC1", ""]), pd.Series([None, None]), {"TP53"})

        assert result.tolist() == ["LOC1", ""]

    def test_a_rename_onto_another_genes_current_symbol_is_skipped(self) -> None:
        result = harmonize_symbols(pd.Series(["KRAS", "KRAS2"]), pd.Series([None, "KRAS"]), {"KRAS"})

        assert result.tolist() == ["KRAS", "KRAS2"]

    def test_a_gene_without_its_own_symbol_takes_the_annotations(self) -> None:
        result = harmonize_symbols(pd.Series([None]), pd.Series(["ICAM4"]), {"ICAM4"})

        assert result.tolist() == ["ICAM4"]


class TestSymbolsByKey:
    @pytest.fixture
    def annotation(self, tmp_dir: UPath) -> UPath:
        path = tmp_dir / "gene_identifiers.csv"
        pd.DataFrame(
            {"hgnc_symbol": ["AARS1", "TP53"], "ensembl_gene_id": ["ENSG4", None], "hgnc_id": ["HGNC:4", "HGNC:1"]}
        ).to_csv(str(path), index=False)
        return path

    def test_genes_are_looked_up_by_the_named_key_column(self, annotation: UPath) -> None:
        by_ensembl = symbols_by_key(
            pd.Series(["ENSG4", "ENSG1"]), pd.Series(["AARS", "P53"]), "ensembl_gene_id", annotation
        )
        by_hgnc = symbols_by_key(pd.Series(["HGNC:4", "HGNC:1"]), pd.Series(["AARS", "P53"]), "hgnc_id", annotation)

        assert by_ensembl.tolist() == ["AARS1", "P53"]
        assert by_hgnc.tolist() == ["AARS1", "TP53"]

    def test_an_annotation_without_the_key_columns_is_rejected(self, tmp_dir: UPath) -> None:
        path = tmp_dir / "bad.csv"
        pd.DataFrame({"hgnc_symbol": ["TP53"]}).to_csv(str(path), index=False)

        with pytest.raises(CurationError, match="lacks the columns"):
            read_gene_annotation(path)
