"""Process standard-format expression (``expression.parquet``) into AnnData.

The long table of the ``standard`` contract (:mod:`data_curation.core.standard`)
is pivoted to samples x genes and processed like the public RNA-seq: ``X`` is
log2(TPM+1), raw counts are kept as a layer when present, and genes with any
missing value in either matrix are dropped. Several Ensembl genes can share a
symbol; per sample the first row of a symbol in file order is kept, matching the
first-occurrence rule of the public processors.
"""

import logging

import anndata as ad
import numpy as np
import pandas as pd
from upath import UPath

from data_curation.core import layout, standard
from data_curation.omics import COUNT_LAYER

logger = logging.getLogger(__name__)

#: Unit of ``X``, the same as the public RNA-seq datasets.
EXPRESSION_UNIT = "log2_tpm_plus_1"


def _pivot(values: np.ndarray, sample_codes: np.ndarray, gene_codes: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Scatter long values into a samples x genes matrix, NaN where a pair is absent."""
    matrix = np.full(shape, np.nan, dtype=np.float64)
    matrix[sample_codes, gene_codes] = values
    return matrix


def process_standard_transcriptomics(raw_dir: UPath, counts_unit: str) -> ad.AnnData:
    """Build the samples x genes AnnData of a standard dataset.

    Args:
        raw_dir: Raw transcriptomics directory holding ``expression.parquet``.
        counts_unit: Unit recorded for the counts layer in ``uns["layer_units"]``.

    Returns:
        AnnData with samples as obs, gene symbols as var, log2(TPM+1) as ``X`` and
        the counts as layer ``counts`` when the table has a ``counts`` column.

    Raises:
        CurationError: If ``expression.parquet`` is missing or breaks the contract.
    """
    long = standard.read("expression", raw_dir)
    assert long is not None  # noqa: S101 - only the optional "published" kind can be absent
    sample_codes, samples = pd.factorize(long["sample_id"])
    gene_codes, genes = pd.factorize(long["gene_symbol"])
    shape = (len(samples), len(genes))

    first = ~pd.Series(sample_codes.astype(np.int64) * shape[1] + gene_codes).duplicated(keep="first").to_numpy()
    if not first.all():
        logger.warning("Dropping %d repeated (sample_id, gene_symbol) rows; keeping the first", int((~first).sum()))
    sample_codes, gene_codes = sample_codes[first], gene_codes[first]

    matrices = {"tpm": _pivot(long["tpm"].to_numpy()[first], sample_codes, gene_codes, shape)}
    if "counts" in long.columns:
        matrices[COUNT_LAYER] = _pivot(long["counts"].to_numpy()[first], sample_codes, gene_codes, shape)
    del long

    complete = np.logical_and.reduce([~np.isnan(matrix).any(axis=0) for matrix in matrices.values()])
    if not complete.all():
        logger.warning("Dropping %d genes with missing values", int((~complete).sum()))

    tpm = matrices.pop("tpm")[:, complete]
    adata = ad.AnnData(
        X=np.log2(tpm + 1.0).astype(np.float32),
        obs=pd.DataFrame(index=pd.Index(samples.astype(str), name=layout.SAMPLE_INDEX_NAME)),
        var=pd.DataFrame(index=pd.Index(genes[complete].astype(str), name="gene_symbol")),
        layers={name: matrix[:, complete].astype(np.float32) for name, matrix in matrices.items()},
    )
    adata.var_names_make_unique()
    adata.uns["expression_unit"] = EXPRESSION_UNIT
    if COUNT_LAYER in adata.layers:
        adata.uns["layer_units"] = {COUNT_LAYER: counts_unit}
    logger.info("Processed standard transcriptomics: %d samples x %d genes", adata.n_obs, adata.n_vars)
    return adata
