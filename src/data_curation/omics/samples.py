"""Turn a sample x feature matrix keyed by native IDs into a sample-keyed AnnData.

The dataset-specific omics files (GDSC, CCLE) key their samples by the producer's
identifier. Each processor resolves those to Cellosaurus accessions (see
:mod:`data_curation.cell_lines.native_ids`) and hands the matrix and the mapping
here, so that unmapped samples, duplicate accessions and the ``obs`` layout are
handled the same way for every source, and the same way as
:func:`data_curation.omics.cmp_matrix.to_cell_line_anndata`.
"""

import logging
from collections.abc import Mapping

import anndata as ad
import numpy as np
import pandas as pd

from data_curation.core import layout

logger = logging.getLogger(__name__)


def to_sample_anndata(
    values: pd.DataFrame,
    ids: Mapping[str, tuple[str, str]],
    *,
    var_name: str,
    layers: Mapping[str, pd.DataFrame] | None = None,
) -> ad.AnnData:
    """Build a Cellosaurus-keyed AnnData from a native-ID-keyed matrix.

    Samples without an entry in *ids* are dropped. When two samples resolve to
    the same accession the first one in file order is kept, as for the Cell Model
    Passports matrices. Repeated feature names are made unique.

    Args:
        values: Samples x features, indexed by native sample ID.
        ids: Native sample ID -> (Cellosaurus accession, cell-line name).
        var_name: Name of the ``var`` index, e.g. ``"gene_symbol"`` or ``"cpg_island"``.
        layers: Further matrices aligned to *values* (same index and columns).

    Returns:
        AnnData with ``obs_names`` the accessions (index name ``sample_id``), a
        ``cell_line_name`` column, and float32 ``X`` and layers.
    """
    native = [str(sample) for sample in values.index]
    keep: list[int] = []
    seen: set[str] = set()
    duplicates: list[str] = []
    for position, sample in enumerate(native):
        if sample not in ids:
            continue
        accession = ids[sample][0]
        if accession in seen:
            duplicates.append(accession)
            continue
        seen.add(accession)
        keep.append(position)
    unmapped = sum(sample not in ids for sample in native)
    if unmapped:
        logger.warning("Dropping %d of %d samples without a Cellosaurus accession", unmapped, len(native))
    if duplicates:
        logger.warning(
            "Dropping %d samples whose accession an earlier sample already has: %s",
            len(duplicates),
            sorted(set(duplicates))[:10],
        )

    adata = ad.AnnData(
        X=values.iloc[keep].to_numpy(dtype=np.float32),
        obs=pd.DataFrame(
            {"cell_line_name": [ids[native[position]][1] for position in keep]},
            index=pd.Index([ids[native[position]][0] for position in keep]),
        ),
        var=pd.DataFrame(index=pd.Index([str(column) for column in values.columns], name=var_name)),
        layers={name: layer.iloc[keep].to_numpy(dtype=np.float32) for name, layer in (layers or {}).items()},
    )
    adata.obs_names.name = layout.SAMPLE_INDEX_NAME
    adata.var_names_make_unique()
    return adata
