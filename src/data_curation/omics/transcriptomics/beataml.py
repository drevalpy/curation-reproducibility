"""Process BeatAML2 RNA-seq expression into a standardized AnnData file.

The published matrix is genes x RNA-seq samples with log2-normalized values.
RNA-seq samples (``dbgap_rnaseq_sample``, e.g. ``BA2392R``) are mapped to
patient subjects (``dbgap_subject_id``) via the official sample mapping
workbook, because the response data is keyed by subject.

Raw integer counts (``beataml_counts.txt``) are stored as the ``counts`` layer.
The counts file covers 63,677 genes; it is reindexed to the 22,843-gene
normalized-expression set so both matrices share a consistent gene axis.
"""

import logging

import anndata as ad
import numpy as np
import pandas as pd
from upath import UPath

from data_curation.core import layout
from data_curation.core.paths import find_file
from data_curation.omics import COUNT_LAYER

logger = logging.getLogger(__name__)

_METADATA_COLUMNS = ["stable_id", "description", "biotype"]
_GENE_SYMBOL_COLUMN = "display_label"


def _build_rnaseq_to_subject_map(mapping_file: UPath) -> dict[str, str]:
    """Build dbgap_rnaseq_sample -> dbgap_subject_id from the sample mapping workbook."""
    mapping = pd.read_excel(str(mapping_file), sheet_name="sample_map")
    pairs = mapping[["dbgap_rnaseq_sample", "dbgap_subject_id"]].dropna().drop_duplicates()
    return {str(row.dbgap_rnaseq_sample): str(row.dbgap_subject_id) for row in pairs.itertuples()}


def _read_matrix(path: UPath) -> pd.DataFrame:
    """Read a BeatAML2 genes-x-samples TSV, dropping metadata columns and indexing by gene symbol."""
    df = pd.read_csv(str(path), sep="\t", low_memory=False)
    df = df.drop(columns=[c for c in _METADATA_COLUMNS if c in df.columns])
    return df.set_index(_GENE_SYMBOL_COLUMN)


def process_beataml_transcriptomics(raw_dir: UPath) -> ad.AnnData:
    """Process BeatAML2 normalized expression and raw counts into AnnData keyed by subject."""
    expression_file = find_file(raw_dir, "*norm_exp*")
    counts_file = find_file(raw_dir, "*counts*")
    mapping_file = find_file(raw_dir, "*sample_mapping*")
    logger.info("Reading BeatAML2 expression from %s", expression_file)

    df = _read_matrix(expression_file)
    counts_df = _read_matrix(counts_file)

    # Gene symbols can repeat (e.g. multiple Ensembl IDs mapping to one symbol).
    # Deduplicate now so both frames have a unique index before the reindex below.
    df = df[~df.index.duplicated(keep="first")]
    counts_df = counts_df[~counts_df.index.duplicated(keep="first")]

    rnaseq_to_subject = _build_rnaseq_to_subject_map(mapping_file)

    valid_cols = [col for col in df.columns if col in rnaseq_to_subject]
    missing = len(df.columns) - len(valid_cols)
    if missing:
        logger.warning("Dropping %d RNA-seq samples with no subject mapping", missing)
    df = df[valid_cols]

    # Reindex counts to the normalized-expression gene set (counts covers a superset)
    # and restrict to the same samples in the same order.
    counts_df = counts_df.reindex(index=df.index, columns=valid_cols)

    subject_ids = [rnaseq_to_subject[col] for col in valid_cols]

    # Transpose: samples become obs, genes become var
    adata = ad.AnnData(
        X=df.values.astype(np.float32).T,
        obs=pd.DataFrame(index=pd.Index(subject_ids)),
        var=pd.DataFrame(index=pd.Index(df.index, name="gene_symbol")),
        layers={COUNT_LAYER: counts_df.values.astype(np.float32).T},
    )
    adata.obs_names.name = layout.SAMPLE_INDEX_NAME
    adata.var_names_make_unique()

    # Several RNA-seq specimens can belong to one subject; keep the first
    adata = adata[~adata.obs_names.duplicated(keep="first")].copy()

    return adata
