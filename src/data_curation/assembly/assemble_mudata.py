"""Assemble per-omics AnnData files into a single MuData (.h5mu) file.

Combines all available omics modalities for a dataset into the MuData
format expected by drevalpy. Drug ontology annotations from the central
dictionary are attached to the response modality on the way through, and
sample annotations are consolidated into the global ``mdata.obs``.

The response modality already carries its identifier conventions and
``uns["x_column"]`` from ``harmonize_response``, which is where the axes are
named, so nothing is re-annotated here. Its replicate-fits table
(``response.uns["replicate_fits"]``, see :mod:`data_curation.response.replicate_fits`) is
restricted to the response axes, which collapsing duplicates may have narrowed.
"""

import anndata as ad
import mudata as md
import pandas as pd
from upath import UPath

from data_curation.assembly.write import write_mudata
from data_curation.cell_lines.meta import CELL_LINE_COLUMNS, read_sample_meta, sample_meta_path
from data_curation.core import layout
from data_curation.core.errors import CurationError
from data_curation.response import replicate_fits

md.set_options(pull_on_update=False)

OMICS_MODALITY_MAP: dict[str, str] = {
    "transcriptomics": "gene_expression",
    "mutations": "mutations",
    "cnv": "copy_number_variation_gistic",
    "proteomics": "proteomics",
    "response": "response",
}


# drevalpy requires only "tissue" (drevalpy/data/utils.py); "cellosaurus_id" and
# "cell_line_name" are optional and only present for cell-line datasets. Everything
# else the metadata table carries rides along so downstream code can stratify or
# filter on it - "cancer_type" is a finer split variable than "tissue",
# "patient_id" exposes shared-donor leakage, and "problematic" marks contaminated
# or misidentified lines. The global columns are whatever the table holds.


def _attach_drug_ontology(response: ad.AnnData) -> None:
    """Attach drug ontology annotations to a response modality in place.

    Reindexing on ``var_names`` subsets the central dictionary to this dataset's
    drugs and guarantees row alignment. Columns already present are not
    overwritten, so re-assembly stays idempotent.
    """
    if not layout.DRUG_ONTOLOGY.exists():
        return

    ontology = pd.read_parquet(str(layout.DRUG_ONTOLOGY))
    new_columns = [c for c in ontology.columns if c not in response.var.columns]
    if not new_columns:
        return

    aligned = ontology[new_columns].reindex(response.var_names)
    for column in new_columns:
        response.var[column] = aligned[column].to_numpy()


def _modality_cell_line_names(mdata: md.MuData) -> pd.Series:
    """Collect cell line names from the per-modality ``obs`` frames.

    Used only for cell-line datasets, as a fallback for accessions missing from
    the metadata table. The response modality is consulted first because its names
    come straight from the screen; remaining gaps are filled from the omics
    modalities.
    """
    ordered = [k for k in mdata.mod if k == "response"] + [k for k in mdata.mod if k != "response"]
    collected: dict[str, str] = {}
    for key in ordered:
        obs = mdata.mod[key].obs
        if "cell_line_name" not in obs.columns:
            continue
        for accession, name in zip(obs.index, obs["cell_line_name"].to_numpy(), strict=True):
            if accession not in collected and pd.notna(name):
                collected[str(accession)] = str(name)
    return pd.Series(collected, dtype="object")


def _build_global_obs(mdata: md.MuData, meta_path: UPath) -> None:
    """Populate ``mdata.obs`` with sample metadata and drop per-modality copies.

    The metadata table is joined onto the union sample index, which is the only
    index that spans every modality: for GDSC2 it holds 1675 cell lines while
    ``gene_expression`` covers 1584, so resolving tissue inside a single modality
    would leave holes. The global columns are those of the table, so a dataset
    whose samples are not cell lines gets no ``cellosaurus_id`` or
    ``cell_line_name``.
    """
    meta = read_sample_meta(meta_path)

    # The global index must carry the same name as every modality's: mudata joins
    # the global obs onto each modality on read, and a named global index against
    # an unnamed or differently named modality index raises "cannot join with no
    # overlapping index names".
    index = pd.Index(mdata.obs_names, name=layout.SAMPLE_INDEX_NAME)
    global_obs = meta.reindex(index)

    if all(column in global_obs.columns for column in CELL_LINE_COLUMNS):
        # A cell line's sample ID is its accession, including the response-only
        # lines the metadata table does not know.
        global_obs["cellosaurus_id"] = index
        missing = global_obs["cell_line_name"].isna()
        if missing.any():
            fallback = _modality_cell_line_names(mdata).reindex(index[missing])
            global_obs.loc[missing, "cell_line_name"] = fallback.to_numpy()
        global_obs["cell_line_name"] = global_obs["cell_line_name"].fillna(pd.Series(index, index=index))

    # The dtype contract is pinned in meta._finalize_cmp: "problematic" is a real
    # bool in the Parquet file and can be coerced directly.
    if "problematic" in global_obs.columns:
        flags = global_obs["problematic"].astype("boolean")
        global_obs["problematic"] = flags.fillna(value=False).astype(bool)

    # Samples missing from the metadata table (response-only cell lines) leave
    # every table-derived column entirely unset, which reads back as float NaN.
    # h5py cannot write an all-NaN object column as strings, so normalize the text
    # columns to empty strings and keep the column present: downstream code can
    # then treat "absent annotation" uniformly instead of having to branch on
    # whether the column exists at all.
    for column in global_obs.columns:
        if column == "problematic":
            continue
        global_obs[column] = global_obs[column].astype("object").where(global_obs[column].notna(), "").astype(str)

    mdata.obs = global_obs

    for modality in mdata.mod.values():
        obs = pd.DataFrame(modality.obs)
        kept = [c for c in obs.columns if c not in global_obs.columns]
        if len(kept) != len(obs.columns):
            modality.obs = obs[kept]


def _subset_to_screened(modalities: dict[str, ad.AnnData]) -> dict[str, ad.AnnData]:
    """Restrict the omics modalities to the samples this dataset screened.

    The Cell Model Passports omics are dataset-independent: every screen reads
    the same matrices, so without subsetting each assembled file embeds all 1584
    profiled cell lines regardless of which its response covers. Subsetting is
    both smaller and more honest about what the dataset contains.

    Samples with a response but no omics are kept in the union index, so
    ``mdata.obs`` still spans them and their features read as missing rather than
    the sample vanishing.

    Args:
        modalities: Modality name to AnnData, including ``response``.

    Returns:
        The modalities, with the non-response ones subset.
    """
    response = modalities.get("response")
    if response is None:
        return modalities

    screened = set(response.obs_names)
    subset: dict[str, ad.AnnData] = {}
    for name, adata in modalities.items():
        if name == "response":
            subset[name] = adata
            continue
        keep = [obs_name in screened for obs_name in adata.obs_names]
        subset[name] = adata[keep].copy() if not all(keep) else adata
    return subset


def _restrict_replicate_fits(response: ad.AnnData) -> None:
    """Restrict the response's replicate-fits table, if it has one, to the response axes.

    The table is keyed by the same canonical identifiers as the response axes, so
    restricting it keeps exactly the rows of samples and drugs the modality holds.
    """
    table = response.uns.get(replicate_fits.KEY)
    if table is not None:
        response.uns[replicate_fits.KEY] = replicate_fits.restrict(table, response.obs_names, response.var_names)


def main(dataset: str) -> None:
    """Assemble all available omics for a dataset into a MuData file."""
    output_path = layout.assembled(dataset)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    modalities: dict[str, ad.AnnData] = {}

    for layout_key, mudata_key in OMICS_MODALITY_MAP.items():
        h5ad_path = layout.modality(layout_key, dataset)
        if h5ad_path.exists():
            modalities[mudata_key] = ad.read_h5ad(str(h5ad_path))

    if not modalities:
        msg = f"No omics data found for dataset: {dataset}"
        raise CurationError(msg)

    if "response" in modalities:
        _attach_drug_ontology(modalities["response"])

    modalities = _subset_to_screened(modalities)

    if "response" in modalities:
        _restrict_replicate_fits(modalities["response"])

    mdata = md.MuData(modalities)
    mdata.uns["dataset_name"] = dataset
    _build_global_obs(mdata, sample_meta_path(dataset))

    write_mudata(mdata, output_path)
