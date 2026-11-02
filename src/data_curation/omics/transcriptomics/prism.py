"""Process PRISM/DepMap CCLE RNA-seq expression into a standardized AnnData file.

PRISM cell lines are identified by DepMap IDs (``ACH-*``). The co-distributed
gene expression file ``CCLE_expression.csv`` from the matching DepMap quarterly
release (19Q4 / 20Q2) provides log2(TPM+1) values.

Layout of ``CCLE_expression.csv``:
- Row index: DepMap IDs (e.g. ``ACH-000001``).
- Column headers: ``GENE_SYMBOL (ENSEMBL_ID)`` (e.g. ``A1BG (1)``).
- Values: log2(TPM + 1) RNA-seq expression.

We strip the Ensembl suffix from column headers to produce plain gene symbols as
``var_names``, matching the convention of other expression matrices in the
pipeline. Duplicate gene symbols (rare, e.g. from PAR loci) are deduplicated by
keeping the first occurrence.

Cell-line identity resolution follows the same route as
``response/harmonize/identity.py``: DepMap IDs are mapped to Cellosaurus
accessions via the ``BROAD_ID`` column of the Cell Model Passports model list.
"""

from __future__ import annotations

import logging
import re

import anndata as ad
import numpy as np
import pandas as pd
from upath import UPath

from data_curation.cell_lines.download_model_list import cmp_model_list_path
from data_curation.cell_lines.sidm_mapping import _valid_cvcl
from data_curation.core import layout
from data_curation.core.paths import find_file

logger = logging.getLogger(__name__)

#: Pattern that matches the ``(ENSEMBL_ID)`` suffix appended to column names.
_ENSEMBL_SUFFIX = re.compile(r"\s*\([^)]+\)\s*$")

#: Expression unit stored in ``uns``.
EXPRESSION_UNIT = "log2_tpm_plus_1"


def _strip_ensembl_suffix(name: str) -> str:
    """Remove the ``(ENSEMBL_ID)`` suffix from a gene column name."""
    return _ENSEMBL_SUFFIX.sub("", name).strip()


def _build_depmap_to_cellosaurus(model_list_path: UPath) -> dict[str, str]:
    """Build a DepMap ACH-* ID -> Cellosaurus accession map.

    Uses the same ``BROAD_ID`` column that
    ``response/harmonize/identity._resolve_by_depmap_id`` relies on.
    """
    models = pd.read_csv(str(model_list_path), low_memory=False)
    mapping: dict[str, str] = {}
    for _, row in models.iterrows():
        broad = row.get("BROAD_ID")
        if pd.isna(broad):
            continue
        cvcl = _valid_cvcl(row)
        if cvcl is None:
            continue
        mapping[str(broad).strip()] = cvcl
    return mapping


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

    # Strip Ensembl suffixes from column headers to get plain gene symbols.
    df.columns = pd.Index([_strip_ensembl_suffix(c) for c in df.columns])

    # Deduplicate gene symbols (keep first occurrence).
    dup_mask = df.columns.duplicated(keep="first")
    if dup_mask.any():
        logger.warning("Dropping %d duplicate gene symbols", dup_mask.sum())
        df = df.loc[:, ~dup_mask]

    # Resolve DepMap IDs -> Cellosaurus accessions.
    depmap_to_cvcl = _build_depmap_to_cellosaurus(cmp_model_list_path())
    resolvable = [idx for idx in df.index if idx in depmap_to_cvcl]
    unresolved = len(df.index) - len(resolvable)
    if unresolved:
        logger.warning(
            "Dropping %d cell lines with no Cellosaurus mapping (of %d total)",
            unresolved,
            len(df.index),
        )
    df = df.loc[resolvable]

    cellosaurus_ids = [depmap_to_cvcl[dep] for dep in df.index]

    # Build AnnData: obs = samples (cell lines), var = genes.
    adata = ad.AnnData(
        X=df.values.astype(np.float32),
        obs=pd.DataFrame(index=pd.Index(cellosaurus_ids)),
        var=pd.DataFrame(index=pd.Index(df.columns, name="gene_symbol")),
    )
    adata.obs_names.name = layout.SAMPLE_INDEX_NAME

    # Deduplicate cell lines that have the same Cellosaurus accession.
    dup_obs = adata.obs_names.duplicated(keep="first")
    if dup_obs.any():
        logger.warning(
            "Dropping %d duplicate Cellosaurus IDs (multiple DepMap lines map to same accession)",
            dup_obs.sum(),
        )
        adata = adata[~dup_obs].copy()

    logger.info(
        "Processed PRISM transcriptomics: %d cell lines × %d genes",
        adata.n_obs,
        adata.n_vars,
    )
    return adata
