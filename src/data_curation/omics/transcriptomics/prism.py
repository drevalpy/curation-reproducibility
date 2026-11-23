"""Process PRISM/DepMap CCLE RNA-seq expression into a standardized AnnData file.

PRISM cell lines are identified by DepMap IDs (``ACH-*``). The co-distributed
gene expression file ``CCLE_expression.csv`` from the matching DepMap quarterly
release (19Q4 / 20Q2) provides log2(TPM+1) values.

Layout of ``CCLE_expression.csv``:
- Row index: DepMap IDs (e.g. ``ACH-000001``).
- Column headers: ``GENE_SYMBOL (ENTREZ_ID)`` (e.g. ``A1BG (1)``).
- Values: log2(TPM + 1) RNA-seq expression.

We strip the Entrez suffix from column headers to produce plain gene symbols as
``var_names``, matching the convention of other expression matrices in the
pipeline. Duplicate gene symbols (rare, e.g. from PAR loci) are deduplicated by
keeping the first occurrence.

DepMap IDs are mapped to Cellosaurus accessions through the ``BROAD_ID`` column
of the Cell Model Passports model list, as for the PRISM responses.
"""

import logging
import re

import anndata as ad
import pandas as pd
from upath import UPath

from data_curation.cell_lines.download_model_list import cmp_model_list_path
from data_curation.cell_lines.sidm_mapping import build_id_to_cellosaurus_map
from data_curation.core.paths import find_file
from data_curation.omics.samples import to_sample_anndata

logger = logging.getLogger(__name__)

#: Pattern that matches the ``(ENTREZ_ID)`` suffix appended to column names.
_ENTREZ_SUFFIX = re.compile(r"\s*\([^)]+\)\s*$")

#: Unit of ``X``, recorded in ``uns["expression_unit"]``.
EXPRESSION_UNIT = "log2_tpm_plus_1"


def process_prism_transcriptomics(raw_dir: UPath) -> ad.AnnData:
    """Process PRISM DepMap CCLE expression into an AnnData keyed by Cellosaurus.

    Args:
        raw_dir: Raw transcriptomics directory (contains ``CCLE_expression.csv``).

    Returns:
        AnnData with ``obs`` = Cellosaurus accessions, ``var`` = gene symbols,
        ``X`` = log2(TPM+1) values.
    """
    expr_file = find_file(raw_dir, "CCLE_expression.csv")
    logger.info("Reading PRISM CCLE expression from %s", expr_file)

    df = pd.read_csv(str(expr_file), index_col=0)
    logger.info("Raw expression matrix: %d cell lines × %d genes", *df.shape)
    df.columns = pd.Index([_ENTREZ_SUFFIX.sub("", column).strip() for column in df.columns])

    dup_mask = df.columns.duplicated(keep="first")
    if dup_mask.any():
        logger.warning("Dropping %d duplicate gene symbols", dup_mask.sum())
        df = df.loc[:, ~dup_mask]

    ids = build_id_to_cellosaurus_map(cmp_model_list_path(), "BROAD_ID")
    adata = to_sample_anndata(df, ids, var_name="gene_symbol")
    adata.uns["expression_unit"] = EXPRESSION_UNIT
    logger.info("Processed PRISM transcriptomics: %d cell lines x %d genes", adata.n_obs, adata.n_vars)
    return adata
