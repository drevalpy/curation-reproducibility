"""Process the CCLE 2019 RNA-seq release into a standardized AnnData file.

The Broad's legacy CCLE release (Ghandi et al., 2019) ships two matrices of the
same RNA-seq run, both with genes as rows and CCLE names (``A549_LUNG``) as columns:

- ``CCLE_RNAseq_rsem_genes_tpm_*.txt.gz``: RSEM TPM, keyed by versioned Ensembl gene
  ID (``ENSG00000000003.10``). It carries no gene symbols.
- ``CCLE_RNAseq_genes_counts_*.gct.gz``: GCT 1.2 integer RNA-SeQC gene read counts,
  keyed by the same IDs with the gene symbol in its ``Description`` column. These
  are read counts, not RSEM expected counts.

``X`` is log2(TPM + 1) and the read counts are kept as the ``counts`` layer. Genes
are named by the Cell Model Passports symbol of their Ensembl ID where the
file's own GENCODE v19 symbol is out of date (see ``omics/gene_symbols.py``). The
counts file covers 1,618 fewer genes than the TPM file; of those, the ones the
annotation names are kept with missing counts, since several are in drevalpy's
gene lists (``ICAM4``, ``PGAM2``). Only the cell lines in both files are kept, and
genes with a missing TPM, or a missing count where the counts file has the gene,
are dropped.
"""

import logging
from collections.abc import Hashable
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
from upath import UPath

from data_curation.cell_lines.cellosaurus import CELLOSAURUS_PATH
from data_curation.cell_lines.native_ids import resolve_ccle_names
from data_curation.core.errors import CurationError
from data_curation.core.paths import find_first
from data_curation.omics import COUNT_LAYER
from data_curation.omics.gene_symbols import GENE_IDENTIFIERS_PATH, symbols_by_key
from data_curation.omics.samples import to_sample_anndata

logger = logging.getLogger(__name__)

#: Unit of ``X``, recorded in ``uns["expression_unit"]``.
EXPRESSION_UNIT = "log2_tpm_plus_1"

#: Unit of the counts layer: RNA-SeQC gene read counts.
COUNTS_UNIT = "rnaseqc_read_count"

#: GCT 1.2 has a version line and a dimensions line before the header.
GCT_PREAMBLE_ROWS = 2

#: Leading annotation columns of the TPM file and the GCT file.
TPM_ID_COLUMN = "gene_id"
GCT_ID_COLUMN = "Name"
GCT_SYMBOL_COLUMN = "Description"


def _read_genes_by_samples(path: UPath, annotations: list[str], skiprows: int = 0) -> pd.DataFrame:
    """Read a genes x samples file into float32 values with the annotation columns first.

    Args:
        path: Tab-separated file, optionally gzipped.
        annotations: Names of the leading non-numeric columns.
        skiprows: Lines before the header.

    Raises:
        CurationError: If the header lacks an annotation column.
    """
    header = pd.read_csv(str(path), sep="\t", skiprows=skiprows, nrows=0).columns
    missing = [column for column in annotations if column not in header]
    if missing:
        msg = f"{path} lacks the columns {missing}; found {list(header[:4])}"
        raise CurationError(msg)
    dtypes: dict[Hashable, Any] = {column: ("str" if column in annotations else "float32") for column in header}
    return pd.read_csv(str(path), sep="\t", skiprows=skiprows, dtype=dtypes)


def _check_unique_ids(tpm: pd.DataFrame, counts: pd.DataFrame) -> None:
    """Raise if either file repeats a gene ID.

    Raises:
        CurationError: If an ID repeats.
    """
    for name, table, column in (("TPM", tpm, TPM_ID_COLUMN), ("counts", counts, GCT_ID_COLUMN)):
        if table[column].duplicated().any():
            msg = f"The CCLE {name} file repeats gene IDs in {column!r}"
            raise CurationError(msg)


def _align(tpm: pd.DataFrame, counts: pd.DataFrame, gene_identifiers_path: UPath) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Restrict both matrices to the shared cell lines and the named genes, in TPM file order.

    A gene is kept when the counts file has it, or when the Cell Model Passports
    annotation names its Ensembl ID; the latter have no counts.

    Returns:
        The TPM matrix and the counts matrix (missing where the counts file lacks the
        gene), both genes x samples and indexed by gene symbol (the Ensembl ID where
        neither file nor annotation has one; repeats are made unique downstream).

    Raises:
        CurationError: If the files share no genes or no samples, or repeat an ID.
    """
    _check_unique_ids(tpm, counts)
    tpm = tpm.set_index(TPM_ID_COLUMN).drop(columns="transcript_ids", errors="ignore")
    counts = counts.set_index(GCT_ID_COLUMN)
    own_symbols = counts.pop(GCT_SYMBOL_COLUMN).reindex(tpm.index)

    samples = tpm.columns[tpm.columns.isin(counts.columns)]
    in_counts = pd.Series(tpm.index.isin(counts.index), index=tpm.index)
    if not in_counts.any() or samples.empty:
        msg = "The CCLE TPM and counts files share no genes or no cell lines"
        raise CurationError(msg)

    ensembl = pd.Series(tpm.index.str.split(".").str[0], index=tpm.index)
    symbols = symbols_by_key(ensembl, own_symbols, "ensembl_gene_id", gene_identifiers_path)
    named = symbols.str.strip().astype(bool)
    keep = in_counts | named
    logger.info(
        "TPM file: %d genes x %d cell lines; counts file: %d x %d; kept %d genes (%d without counts) x %d",
        *tpm.shape,
        *counts.shape,
        int(keep.sum()),
        int((keep & ~in_counts).sum()),
        len(samples),
    )

    tpm = tpm.loc[keep, samples]
    counts = counts.reindex(index=tpm.index, columns=samples)
    complete = tpm.notna().all(axis=1) & (counts.notna().all(axis=1) | ~in_counts[keep])
    if not complete.all():
        logger.info("Dropping %d genes with missing values", int((~complete).sum()))
    tpm, counts = tpm.loc[complete], counts.loc[complete]
    names = symbols.loc[tpm.index]
    names = names.where(names.str.strip().astype(bool), pd.Series(tpm.index, index=tpm.index))
    tpm.index = counts.index = pd.Index(names.astype(str))
    return tpm, counts


def process_ccle_transcriptomics(
    raw_dir: UPath,
    model_list_path: UPath,
    cellosaurus_path: UPath = CELLOSAURUS_PATH,
    gene_identifiers_path: UPath = GENE_IDENTIFIERS_PATH,
) -> ad.AnnData:
    """Process the CCLE 2019 RNA-seq release into an AnnData keyed by Cellosaurus.

    Args:
        raw_dir: Raw transcriptomics directory (contains the TPM and the read-count files).
        model_list_path: Cell Model Passports ``model_list.csv``, which maps CCLE names to accessions.
        cellosaurus_path: Parsed Cellosaurus table, for the CCLE names the model list lacks.
        gene_identifiers_path: Cell Model Passports gene annotation, for the gene symbols.

    Returns:
        AnnData with ``obs`` = Cellosaurus accessions, ``var`` = gene symbols, ``X`` = log2(TPM + 1)
        and the ``counts`` layer holding the RNA-SeQC read counts (``uns["layer_units"]``).

    Raises:
        CurationError: If a file lacks its expected columns or the two files share no genes or cell lines.
    """
    tpm = _read_genes_by_samples(find_first(raw_dir, ("*rsem_genes_tpm*",)), [TPM_ID_COLUMN, "transcript_ids"])
    counts = _read_genes_by_samples(
        find_first(raw_dir, ("*genes_counts*",)), [GCT_ID_COLUMN, GCT_SYMBOL_COLUMN], skiprows=GCT_PREAMBLE_ROWS
    )
    tpm, counts = _align(tpm, counts, gene_identifiers_path)

    ids = resolve_ccle_names(tpm.columns, model_list_path, cellosaurus_path)
    adata = to_sample_anndata(
        np.log2(tpm + 1).T,
        ids,
        var_name="gene_symbol",
        layers={COUNT_LAYER: counts.T},
    )
    adata.uns["layer_units"] = {COUNT_LAYER: COUNTS_UNIT}
    logger.info("Processed CCLE transcriptomics: %d cell lines x %d genes", adata.n_obs, adata.n_vars)
    adata.uns["expression_unit"] = EXPRESSION_UNIT
    return adata
