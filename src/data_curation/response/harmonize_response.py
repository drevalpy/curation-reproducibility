"""Rename fitted curves onto canonical identifiers and attach the annotations.

The cheap second half of the old ``process_response`` stage. It takes the
native-keyed curve metrics from ``fit_curves``, renames both axes to the
identifiers drevalpy expects - sample IDs (Cellosaurus accessions for the
cell-line datasets) and PubChem CIDs - resolves the duplicates that renaming can
create, and attaches ``canonical_smiles``, ``cell_line_name`` (cell-line datasets
only) and the published metric layers. The replicate-fits table ``fit_curves`` stored in
``uns["replicate_fits"]`` is re-keyed onto the same identifiers
(:func:`~data_curation.response.replicate_fits.canonicalize`), without collapsing.

Everything identity-related lives on this side of the split, so re-running it is
minutes rather than the hours ``fit_curves`` costs.
"""

import logging

import anndata as ad
import pandas as pd
from upath import UPath

from data_curation.cell_lines.download_model_list import cmp_model_list_path
from data_curation.core import layout
from data_curation.core.config import dataset_config
from data_curation.response.harmonize.collapse import collapse_axis, log_collapses
from data_curation.response.harmonize.identity import (
    drop_unresolved,
    resolve_drugs,
    resolve_samples,
)
from data_curation.response.harmonize.published import add_published_layers
from data_curation.response.harmonize.structures import apply_structures, read_structures
from data_curation.response.replicate_fits import KEY, canonicalize

logger = logging.getLogger(__name__)

# Unmapped identifiers are expected in small numbers (a screen occasionally
# includes a line no reference database carries), but a large share means the
# mapping itself regressed. Fail rather than silently shrink the dataset.
UNMAPPED_FAILURE_FRACTION = 0.10

#: drevalpy identifies drugs by PubChem CID (``drevalpy/data/utils.py``) and samples
#: by the ``sample_id`` index (:data:`data_curation.core.layout.SAMPLE_INDEX_NAME`);
#: naming the var index records that in the file rather than only in the pipeline
#: that wrote it.
VAR_INDEX_NAME = "pubchem_id"

#: The response ``X`` matrix holds pEC50.
RESPONSE_X_COLUMN = "pEC50"


def harmonize(
    curves: ad.AnnData,
    *,
    fmt: str,
    model_list_path: UPath,
    drug_mapping: pd.DataFrame,
) -> ad.AnnData:
    """Rename both axes of the fitted curves, and their replicate-fits table, onto canonical identifiers.

    Args:
        curves: Curve metrics keyed by native identifiers, optionally with the replicate-fits
            table in ``uns``.
        fmt: The dataset's response format.
        model_list_path: Cell Model Passports model list for cell-line identity.
        drug_mapping: The ``drug_mapping/<ds>.parquet`` table, already on canonical drug IDs and
            SMILES (:func:`~data_curation.response.harmonize.structures.apply_structures`).

    Returns:
        The metrics keyed by sample ID (Cellosaurus accession for cell lines) and
        PubChem CID, with ``canonical_smiles`` attached, ``cell_line_name`` for
        the cell-line datasets, and the re-keyed replicate-fits table.
    """
    table = curves.uns.get(KEY)
    sample_natives = curves.obs_names
    drug_natives = curves.var_names
    if table is not None:
        sample_natives = sample_natives.union(pd.Index(table["native_sample"].astype(str).unique()))
        drug_natives = drug_natives.union(pd.Index(table["native_drug"].astype(str).unique()))
    identity = resolve_samples(sample_natives, fmt, model_list_path)
    drug_identity = resolve_drugs(drug_natives, drug_mapping)

    samples = drop_unresolved(
        identity.canonical.loc[curves.obs_names], axis_name="sample", failure_fraction=UNMAPPED_FAILURE_FRACTION
    )
    drugs = drop_unresolved(
        drug_identity.loc[curves.var_names], axis_name="drug", failure_fraction=UNMAPPED_FAILURE_FRACTION
    )

    subset = curves[list(samples.index), list(drugs.index)].copy()
    subset, obs_collapses = collapse_axis(subset, samples, axis=0)
    log_collapses(obs_collapses, axis_name="sample")
    subset, var_collapses = collapse_axis(subset, drugs, axis=1)
    log_collapses(var_collapses, axis_name="drug")

    _annotate(subset, cell_line_names=identity.cell_line_names, drug_mapping=drug_mapping)
    if table is not None:
        subset.uns[KEY] = _canonical_table(table, identity.canonical, drug_identity)
    return subset


def _canonical_table(table: pd.DataFrame, sample_identity: pd.Series, drug_identity: pd.Series) -> pd.DataFrame:
    """Re-key the replicate-fits table, bounding the unresolved share of its own identifiers."""
    sample_map = drop_unresolved(
        sample_identity.loc[table["native_sample"].astype(str).unique()],
        axis_name="replicate-fits sample",
        failure_fraction=UNMAPPED_FAILURE_FRACTION,
    )
    drug_map = drop_unresolved(
        drug_identity.loc[table["native_drug"].astype(str).unique()],
        axis_name="replicate-fits drug",
        failure_fraction=UNMAPPED_FAILURE_FRACTION,
    )
    out = canonicalize(table, sample_map, drug_map)
    logger.info("Replicate-fits table: %d of %d rows resolved, %s", len(out), len(table), _status_counts(out))
    return out


def _status_counts(table: pd.DataFrame) -> dict[str, int]:
    """Rows per fit status, for the log."""
    return {str(status): int(count) for status, count in table["status"].value_counts().items()}


def _annotate(adata: ad.AnnData, *, cell_line_names: dict[str, str], drug_mapping: pd.DataFrame) -> None:
    """Attach the axis annotations and record the identifier conventions.

    ``cell_line_name`` is written only when the samples are cell lines, i.e. when
    the identity route produced names.
    """
    smiles = dict(zip(drug_mapping["pubchem_id"].astype(str), drug_mapping["canonical_smiles"].fillna(""), strict=True))
    adata.var["canonical_smiles"] = [smiles.get(str(drug), "") for drug in adata.var_names]
    if cell_line_names:
        adata.obs["cell_line_name"] = [cell_line_names.get(str(sample), "") for sample in adata.obs_names]

    adata.obs_names.name = layout.SAMPLE_INDEX_NAME
    adata.var_names.name = VAR_INDEX_NAME
    adata.uns["x_column"] = RESPONSE_X_COLUMN


def main(dataset: str) -> None:
    """Harmonize a dataset's fitted curves into the published response file."""
    raw_dir = layout.raw_dir("response", dataset)
    curves_path = layout.curves(dataset)
    output_path = layout.harmonized(dataset)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fmt = dataset_config(dataset, "response")["format"]

    drug_mapping = pd.read_parquet(str(layout.drug_mapping(dataset)))
    # Identifiers are stored as strings in the Parquet file; the cast is kept as
    # a defensive guard for any future reader that passes a non-string column.
    drug_mapping["pubchem_id"] = drug_mapping["pubchem_id"].astype(str)
    # One canonical ID and one SMILES per drug across all datasets.
    drug_mapping = apply_structures(drug_mapping, read_structures())

    model_list_path = cmp_model_list_path()

    curves = ad.read_h5ad(str(curves_path))
    logger.info("Read %s native-keyed curve metrics for %s", curves.shape, dataset)

    adata = harmonize(curves, fmt=fmt, model_list_path=model_list_path, drug_mapping=drug_mapping)
    logger.info("Harmonized to %s canonical identifiers", adata.shape)

    add_published_layers(adata, fmt, raw_dir, model_list_path, drug_mapping)

    adata.write_h5ad(str(output_path))
    logger.info("Wrote %s", output_path)
