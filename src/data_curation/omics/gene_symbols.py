"""Name the genes of the GDSC and CCLE expression files the way Cell Model Passports does.

The two releases predate many HGNC renames: the CCLE 2019 files use GENCODE v19
(2013) symbols and the GDSC1000 array annotation dates from 2015, so ``AARS1`` is
still ``AARS`` and ``SKIC8`` still ``WDR61``. drevalpy's gene lists, and the Cell
Model Passports mutations, CNV and proteomics of the same cell lines, use the
newer names. Each gene is therefore renamed through the Cell Model Passports gene
annotation (``gene_identifiers.csv``), keyed by a stable identifier the file
carries: the Ensembl gene ID for CCLE, the HGNC accession for GDSC.

A symbol the annotation already lists as current is kept, so a gene is only renamed
when its own name is out of date. A rename that would take a symbol another gene of
the file already holds is not made either, which keeps one gene per symbol.
"""

import logging

import pandas as pd
from upath import UPath

from data_curation.core import layout
from data_curation.core.errors import CurationError

logger = logging.getLogger(__name__)

#: Local name of the Cell Model Passports gene annotation.
GENE_IDENTIFIERS_PATH = layout.cmp_raw("gene_identifiers") / "gene_identifiers.csv"

SYMBOL_COLUMN = "hgnc_symbol"
KEY_COLUMNS = ("ensembl_gene_id", "hgnc_id")


def read_gene_annotation(path: UPath = GENE_IDENTIFIERS_PATH) -> pd.DataFrame:
    """Read the symbol and the identifier columns of the Cell Model Passports gene annotation.

    Raises:
        CurationError: If the file lacks one of the columns.
    """
    table = pd.read_csv(str(path), dtype=str)
    missing = [column for column in (SYMBOL_COLUMN, *KEY_COLUMNS) if column not in table.columns]
    if missing:
        msg = f"{path} lacks the columns {missing}"
        raise CurationError(msg)
    return table[[SYMBOL_COLUMN, *KEY_COLUMNS]].dropna(subset=[SYMBOL_COLUMN])


def harmonize_symbols(original: pd.Series, mapped: pd.Series, current: set[str]) -> pd.Series:
    """Rename out-of-date symbols to the annotation's, without merging two genes into one symbol.

    Args:
        original: The file's own symbol per gene (empty string or NaN when it has none).
        mapped: The annotation's symbol per gene, NaN where the identifier is not annotated;
            aligned to *original*.
        current: Every symbol the annotation lists.

    Returns:
        The symbol per gene: *original* when it is current, unannotated, or when its
        replacement is already another gene's current symbol; *mapped* otherwise.
    """
    original = original.fillna("").astype(str)
    is_current = original.isin(current)
    rename = ~is_current & mapped.notna()
    clashes = rename & mapped.isin(set(original[is_current]))
    rename &= ~clashes
    logger.info(
        "Renaming %d genes to their Cell Model Passports symbol (%d renames skipped: symbol already taken)",
        int(rename.sum()),
        int(clashes.sum()),
    )
    return original.where(~rename, mapped)


def symbols_by_key(
    keys: pd.Series, original: pd.Series, key_column: str, path: UPath = GENE_IDENTIFIERS_PATH
) -> pd.Series:
    """Harmonize the symbols of genes identified by an annotation key column.

    Args:
        keys: Identifier per gene, e.g. an unversioned Ensembl gene ID or ``"HGNC:5"``.
        original: The file's own symbol per gene, aligned to *keys*.
        key_column: ``"ensembl_gene_id"`` or ``"hgnc_id"``.
        path: Cell Model Passports ``gene_identifiers.csv``.

    Returns:
        The harmonized symbol per gene; see :func:`harmonize_symbols`.
    """
    annotation = read_gene_annotation(path)
    keyed = annotation.dropna(subset=[key_column]).drop_duplicates(key_column)
    lookup = dict(zip(keyed[key_column], keyed[SYMBOL_COLUMN], strict=True))
    mapped = keys.map(lookup)
    return harmonize_symbols(original, mapped, set(annotation[SYMBOL_COLUMN]))
