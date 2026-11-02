"""Subsample an assembled MuData file into a tiny toy dataset.

Full-dataset runs are far too slow to iterate on - a single ElasticNet fold on
GDSC2 takes minutes over 36k genes - so this stage produces a dataset small
enough to run every drevalpy split mode in seconds while keeping the same file
layout as the real ones: the same modality names, the same global ``obs``
columns, the same response layers.

Each view also keeps every gene of the gene lists drevalpy ships for it
(``GENE_LISTS_BY_VIEW``), so zoo presets that name a list run on the toy dataset.
The response's replicate-fits table (``uns["replicate_fits"]``) is cut down to the
toy's samples and drugs, since slicing an AnnData leaves ``uns`` untouched.

Selection logic lives in :mod:`data_curation.assembly.toy_selection`; this module
only reads the parameters, applies the selection and writes the result.
"""

import logging
from collections.abc import Mapping, Sequence
from typing import Any

import anndata as ad
import mudata as md
import pandas as pd

from data_curation.assembly.record_toy_gene_lists import shipped_gene_lists
from data_curation.assembly.toy_selection import (
    TISSUE_COLUMN,
    eligible_drugs,
    eligible_samples,
    required_genes,
    select_drugs,
    select_features,
    select_samples,
)
from data_curation.assembly.write import write_mudata
from data_curation.core import layout
from data_curation.core.config import processing_config
from data_curation.core.errors import CurationError
from data_curation.response import replicate_fits

logger = logging.getLogger(__name__)

md.set_options(pull_on_update=False)

ASSEMBLED_DIR = layout.ASSEMBLED

# The response modality is subset by drug rather than by variance, so it is
# excluded from the per-modality feature selection.
RESPONSE_KEY = "response"

# Global obs columns carried over from the source dataset. Only ``tissue`` is
# required; the cell-line columns exist when the source samples are cell lines.
GLOBAL_OBS_COLUMNS = ("cellosaurus_id", "cell_line_name", TISSUE_COLUMN)


GeneLists = Mapping[str, Mapping[str, Sequence[str]]]


def _subset_modality(
    adata: ad.AnnData, samples: list[str], n_features: int, view_lists: Mapping[str, Sequence[str]], view: str
) -> ad.AnnData:
    """Restrict a modality to the selected samples, its top-variance features and its list genes."""
    required = required_genes(view, view_lists, pd.Index(adata.var_names))
    rows = adata[samples].to_memory() if adata.isbacked else adata[samples]
    features = select_features(rows, n_features, required)
    logger.info("%s: kept %d of %d features (%d from gene lists)", view, len(features), adata.n_vars, len(required))
    return rows[:, features].copy()


def _subset_response(response: ad.AnnData, samples: list[str], drugs: list[str]) -> ad.AnnData:
    """Restrict the response to the selected samples and drugs, its replicate-fits table included."""
    subset = response[samples, drugs].copy()
    table = subset.uns.get(replicate_fits.KEY)
    if table is not None:
        subset.uns[replicate_fits.KEY] = replicate_fits.restrict(table, subset.obs_names, subset.var_names)
    return subset


def _global_obs(source_obs: pd.DataFrame, samples: list[str]) -> pd.DataFrame:
    """Carry the source global ``obs`` over to the selected samples.

    The index keeps the ``sample_id`` name every modality carries: ``mudata`` joins
    the global ``obs`` onto each modality on read, and a global index named
    differently from the modality index raises "cannot join with no overlapping
    index names". Columns the source lacks are not invented.
    """
    index = pd.Index(samples, name=layout.SAMPLE_INDEX_NAME)
    aligned = source_obs.reindex(index)
    obs = pd.DataFrame(index=index)
    for column in GLOBAL_OBS_COLUMNS:
        if column in aligned.columns:
            obs[column] = aligned[column].astype("string").astype("object")
    return obs


def _anndata_modalities(source: md.MuData) -> dict[str, ad.AnnData]:
    """Return the source modalities, rejecting nested MuData containers.

    ``MuData.mod`` is typed to allow nested MuData objects; the assembled
    datasets never contain one, and the selection logic assumes plain AnnData.

    Raises:
        TypeError: If a modality is itself a MuData.
    """
    modalities: dict[str, ad.AnnData] = {}
    for key, modality in source.mod.items():
        if not isinstance(modality, ad.AnnData):
            msg = f"Modality '{key}' is a nested MuData, which is not supported"
            raise TypeError(msg)
        modalities[key] = modality
    return modalities


def build_toy(source: md.MuData, toy_params: dict[str, Any], gene_lists: GeneLists) -> md.MuData:
    """Build the toy MuData from an assembled source dataset.

    Args:
        source: An assembled dataset, in memory.
        toy_params: The ``toy`` block of ``params.yaml``.
        gene_lists: Gene lists to keep per modality, as ``{view: {list name: genes}}``; see
            :func:`shipped_gene_lists`. A modality without an entry keeps only its top-variance features.

    Returns:
        A new MuData holding the selected samples, drugs and features.

    Raises:
        CurationError: If the source has no response modality to select drugs from, or a
            modality lacks genes of one of its lists.
    """
    source_modalities = _anndata_modalities(source)
    if RESPONSE_KEY not in source_modalities:
        msg = f"Source dataset has no '{RESPONSE_KEY}' modality to subsample drugs from"
        raise CurationError(msg)

    response = source_modalities[RESPONSE_KEY]
    candidate_lines = eligible_samples(source.obs, source_modalities)
    candidate_drugs = eligible_drugs(response)

    samples = select_samples(
        response=response[:, candidate_drugs],
        candidates=candidate_lines,
        tissues=source.obs[TISSUE_COLUMN],
        n_tissues=int(toy_params["n_tissues"]),
        per_tissue=int(toy_params["samples_per_tissue"]),
    )
    drugs = select_drugs(response, candidate_drugs, samples, int(toy_params["n_drugs"]))

    n_features = int(toy_params["n_features_per_modality"])
    modalities: dict[str, ad.AnnData] = {}
    for key, adata in source_modalities.items():
        if key == RESPONSE_KEY:
            modalities[key] = _subset_response(adata, samples, drugs)
        else:
            modalities[key] = _subset_modality(adata, samples, n_features, gene_lists.get(key, {}), key)

    toy = md.MuData(modalities)
    toy.uns["dataset_name"] = str(toy_params["name"])
    toy.obs = _global_obs(source.obs, samples)
    return toy


def main() -> None:
    """Write the toy dataset described by the ``toy`` block of ``params.yaml``."""
    toy_params = processing_config("toy")

    source_path = ASSEMBLED_DIR / f"{toy_params['source']}.h5mu"
    output_path = ASSEMBLED_DIR / f"{toy_params['name']}.h5mu"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    source = md.read_h5mu(str(source_path))
    toy = build_toy(source, toy_params, shipped_gene_lists())
    write_mudata(toy, output_path)
