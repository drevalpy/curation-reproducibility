"""Process the GDSC1000 basal expression matrix into a standardized AnnData file.

The GDSC cell lines' own expression profile is the RMA-normalised Affymetrix
Human Genome U219 array of Iorio et al. (2016), distributed as
``Cell_line_RMA_proc_basalExp.txt``:

- Tab-separated, one row per gene, one column per cell line.
- ``GENE_SYMBOLS`` and ``GENE_title`` annotate the gene; the remaining columns are
  ``DATA.<COSMIC_ID>``, one per cell line.
- Values are log2 RMA intensities. They are kept as they are in ``X``: the array
  has no counts or TPM, so there is no counts layer.

Probe sets without a gene symbol and genes with a missing value are dropped. A
symbol out of date since 2015 is replaced by the Cell Model Passports symbol of
the HGNC accession in ``GENE_title`` (``[Source:HGNC Symbol;Acc:11858]``); see
``omics/gene_symbols.py``.
Cell lines are identified by COSMIC ID, which the Cell Model Passports model list
resolves to Cellosaurus accessions.
"""

import logging
from collections.abc import Hashable
from typing import Any

import anndata as ad
import pandas as pd
from upath import UPath

from data_curation.cell_lines.native_ids import resolve_cosmic_ids
from data_curation.core.errors import CurationError
from data_curation.core.paths import find_file
from data_curation.omics.gene_symbols import GENE_IDENTIFIERS_PATH, symbols_by_key
from data_curation.omics.samples import to_sample_anndata

logger = logging.getLogger(__name__)

#: Annotation column holding the gene symbol, and the second annotation column.
SYMBOL_COLUMN = "GENE_SYMBOLS"
TITLE_COLUMN = "GENE_title"

#: Prefix of the per-cell-line columns, followed by the COSMIC ID.
SAMPLE_PREFIX = "DATA."


#: HGNC accession in ``GENE_title``, e.g. ``"tetraspanin 6 [Source:HGNC Symbol;Acc:11858]"``.
HGNC_ACCESSION = r"Acc:(\d+)\]"


def _read_expression(path: UPath, gene_identifiers_path: UPath) -> pd.DataFrame:
    """Read the matrix as samples x genes, indexed by COSMIC ID, with complete genes only.

    Raises:
        CurationError: If the file lacks the symbol column or any ``DATA.<COSMIC_ID>`` column.
    """
    header = pd.read_csv(str(path), sep="\t", nrows=0).columns
    sample_columns = [str(column) for column in header if str(column).startswith(SAMPLE_PREFIX)]
    if SYMBOL_COLUMN not in header or not sample_columns:
        msg = (
            f"{path} is not a GDSC expression matrix: expected {SYMBOL_COLUMN!r} and {SAMPLE_PREFIX}<COSMIC_ID> columns"
        )
        raise CurationError(msg)

    dtypes: dict[Hashable, Any] = {
        SYMBOL_COLUMN: "str",
        TITLE_COLUMN: "str",
        **dict.fromkeys(sample_columns, "float32"),
    }
    table = pd.read_csv(str(path), sep="\t", dtype=dtypes)
    logger.info("Raw GDSC expression matrix: %d genes x %d cell lines", len(table), len(sample_columns))

    symbols = table[SYMBOL_COLUMN].str.strip()
    named = symbols.notna() & (symbols != "")
    complete = ~table[sample_columns].isna().any(axis=1)
    logger.info(
        "Dropping %d probe sets without a gene symbol and %d genes with missing values",
        int((~named).sum()),
        int((named & ~complete).sum()),
    )
    keep = named & complete
    values = table.loc[keep, sample_columns]
    hgnc_ids = "HGNC:" + table.loc[keep, TITLE_COLUMN].str.extract(HGNC_ACCESSION)[0]
    values.index = pd.Index(
        symbols_by_key(hgnc_ids, symbols[keep], "hgnc_id", gene_identifiers_path), name="gene_symbol"
    )

    duplicated = values.index.duplicated(keep="first")
    if duplicated.any():
        logger.warning("Dropping %d repeated gene symbols", int(duplicated.sum()))
        values = values.loc[~duplicated]

    samples = values.T
    samples.index = pd.Index([column.removeprefix(SAMPLE_PREFIX) for column in samples.index])
    return samples


def process_gdsc_transcriptomics(
    raw_dir: UPath, model_list_path: UPath, gene_identifiers_path: UPath = GENE_IDENTIFIERS_PATH
) -> ad.AnnData:
    """Process the GDSC1000 RMA basal expression into an AnnData keyed by Cellosaurus.

    Args:
        raw_dir: Raw transcriptomics directory (contains ``Cell_line_RMA_proc_basalExp.txt``).
        model_list_path: Cell Model Passports ``model_list.csv``, which maps COSMIC IDs to accessions.
        gene_identifiers_path: Cell Model Passports gene annotation, for the gene symbols.

    Returns:
        AnnData with ``obs`` = Cellosaurus accessions, ``var`` = gene symbols and ``X`` =
        log2 RMA intensities. There is no counts layer.

    Raises:
        CurationError: If the expression file is not a GDSC basal expression matrix.
    """
    samples = _read_expression(find_file(raw_dir, "Cell_line_RMA_proc_basalExp.txt"), gene_identifiers_path)
    ids = resolve_cosmic_ids(samples.index, model_list_path)
    adata = to_sample_anndata(samples, ids, var_name="gene_symbol")
    logger.info("Processed GDSC transcriptomics: %d cell lines x %d genes", adata.n_obs, adata.n_vars)
    return adata
