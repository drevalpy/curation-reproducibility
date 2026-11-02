"""Selection logic for the subsampled toy dataset.

Full-dataset runs are too slow to iterate on, so ``subsample_toy`` carves a tiny
but structurally complete dataset out of an assembled MuData file. "Structurally
complete" is the point: every split mode drevalpy offers has to work on it, so

* samples are drawn only from sample IDs present in *every* modality and
  carrying a known tissue, so leave-sample-out and leave-tissue-out both have
  something to split on. ``splitters/lto.py`` runs ``KFold(n_splits=5)`` over
  unique tissues, which puts a hard floor of 6 tissues on the selection;
* drugs keep a numeric PubChem CID, a canonical SMILES and single-compound
  status, so drug featurizers have usable input for leave-drug-out;
* features are ranked by variance, preferring features that are complete over
  the selected samples because ``_valid_pair_mask`` in drevalpy drops a
  training pair on *any* NaN feature;
* on top of that ranking every view keeps all genes of the gene lists drevalpy
  ships for it, so a preset that names a list works on the toy dataset
  unchanged. A list gene the source lacks is an error, not a silent gap.

Every tie is broken deterministically, so the stage output is reproducible.
"""

from collections.abc import Mapping, Sequence

import anndata as ad
import numpy as np
import pandas as pd

from data_curation.core.errors import CurationError

# Global obs column carrying the tissue label, written by MuData assembly.
TISSUE_COLUMN = "tissue"

# How many missing genes of a gene list an error message names.
_MISSING_PREVIEW = 5

# Response var columns used to qualify a drug.
SMILES_COLUMN = "canonical_smiles"
COMBINATION_COLUMN = "is_combination"


def eligible_samples(mdata_obs: pd.DataFrame, modalities: Mapping[str, ad.AnnData]) -> pd.Index:
    """Return sample IDs present in every modality and carrying a known tissue.

    Args:
        mdata_obs: Global ``obs`` frame of the source MuData file.
        modalities: The source modalities, keyed as in ``mdata.mod``.

    Returns:
        Sample IDs in the order they appear in ``mdata_obs``.
    """
    shared = pd.Index(mdata_obs.index)
    for adata in modalities.values():
        shared = shared.intersection(pd.Index(adata.obs_names))
    known_tissue = mdata_obs.index[mdata_obs[TISSUE_COLUMN].notna()]
    return pd.Index([c for c in mdata_obs.index if c in shared and c in known_tissue])


def eligible_drugs(response: ad.AnnData) -> pd.Index:
    """Return drugs with a numeric PubChem CID, a SMILES string and no combination flag.

    Args:
        response: Response modality of the source MuData file.

    Returns:
        Qualifying drug identifiers, in source order.
    """
    identifiers = pd.Index(response.var_names)
    as_series = pd.Series(identifiers, index=identifiers, dtype="string")
    numeric = as_series.str.fullmatch(r"\d+").fillna(value=False).astype(bool)
    has_smiles = pd.Series(response.var[SMILES_COLUMN].notna().to_numpy(), index=identifiers)
    if COMBINATION_COLUMN in response.var.columns:
        flags = pd.Series(response.var[COMBINATION_COLUMN].to_numpy(), index=identifiers)
        single = ~flags.fillna(value=False).astype(bool)
    else:
        single = pd.Series(data=True, index=identifiers)
    return identifiers[(numeric & has_smiles & single).to_numpy()]


def _measurement_counts(response: ad.AnnData, axis: int) -> pd.Series:
    """Count non-missing response measurements along one axis."""
    values = np.asarray(response.X, dtype=np.float64)
    counts = np.isfinite(values).sum(axis=axis)
    labels = response.obs_names if axis == 1 else response.var_names
    return pd.Series(counts, index=pd.Index(labels))


def select_samples(
    response: ad.AnnData,
    candidates: pd.Index,
    tissues: pd.Series,
    n_tissues: int,
    per_tissue: int,
) -> list[str]:
    """Pick ``per_tissue`` samples from each of the ``n_tissues`` best-covered tissues.

    Tissues are ranked by how many candidate samples they contribute, and samples
    within a tissue by how many response measurements they carry, so the
    toy stays dense enough for every model to fit.

    Args:
        response: Response modality restricted to the eligible drugs.
        candidates: Sample IDs eligible for selection.
        tissues: Tissue label per sample.
        n_tissues: Number of distinct tissues to keep.
        per_tissue: Number of samples to keep per tissue.

    Returns:
        Selected sample IDs, grouped by tissue in ranked order.

    Raises:
        CurationError: If fewer than ``n_tissues`` tissues have ``per_tissue``
            eligible samples.
    """
    density = _measurement_counts(response, axis=1).reindex(candidates).fillna(value=0)
    frame = pd.DataFrame({"tissue": tissues.reindex(candidates).astype(str), "density": density})

    sizes = frame.groupby("tissue").size()
    usable = sorted(sizes[sizes >= per_tissue].index)
    if len(usable) < n_tissues:
        msg = (
            f"Only {len(usable)} tissues have at least {per_tissue} eligible samples, "
            f"but {n_tissues} are required for the toy dataset"
        )
        raise CurationError(msg)

    ranked_tissues = sorted(usable, key=lambda t: (-sizes[t], t))[:n_tissues]

    selected: list[str] = []
    for tissue in sorted(ranked_tissues):
        block = frame[frame["tissue"] == tissue]
        ordered = sorted(block.index, key=lambda c: (-block.loc[c, "density"], str(c)))
        selected.extend(str(c) for c in ordered[:per_tissue])
    return selected


def select_drugs(response: ad.AnnData, candidates: pd.Index, samples: list[str], n_drugs: int) -> list[str]:
    """Pick the ``n_drugs`` candidate drugs measured on the most selected samples.

    Args:
        response: Response modality of the source MuData file.
        candidates: Drug identifiers eligible for selection.
        samples: Sample IDs already selected for the toy dataset.
        n_drugs: Number of drugs to keep.

    Returns:
        Selected drug identifiers.

    Raises:
        CurationError: If fewer than ``n_drugs`` candidates are available.
    """
    if len(candidates) < n_drugs:
        msg = f"Only {len(candidates)} eligible drugs available, but {n_drugs} are required for the toy dataset"
        raise CurationError(msg)

    subset = response[samples, candidates]
    density = _measurement_counts(subset, axis=0)
    ordered = sorted(density.index, key=lambda d: (-density[d], str(d)))
    return [str(d) for d in ordered[:n_drugs]]


def top_variance_features(adata: ad.AnnData, n_features: int) -> list[str]:
    """Return the ``n_features`` most variable features, preferring complete ones.

    Features with no missing value across the given samples are considered
    first, because drevalpy discards a training pair as soon as one feature is
    NaN. Only if too few complete features exist does the ranking fall back to
    all features, ranked by ``nanvar``. Constant features are useless to a
    scaler, so zero-variance features are excluded whenever alternatives remain.

    Args:
        adata: A modality already subset to the selected samples.
        n_features: Number of features to keep.

    Returns:
        Feature names, in the source ``var`` order.
    """
    values = np.asarray(adata.X, dtype=np.float64)
    names = pd.Index(adata.var_names)
    if len(names) <= n_features:
        return [str(v) for v in names]

    finite = np.isfinite(values)
    complete = finite.all(axis=0)

    # Variance is only defined where at least two values are observed; anything
    # else is ranked last rather than left as a NaN.
    variance = np.full(values.shape[1], -1.0)
    measurable = finite.sum(axis=0) >= 2  # noqa: PLR2004
    if measurable.any():
        variance[measurable] = np.nanvar(values[:, measurable], axis=0)
    variance = np.nan_to_num(variance, nan=-1.0, posinf=-1.0, neginf=-1.0)

    candidates = np.flatnonzero(complete & (variance > 0))
    if len(candidates) < n_features:
        candidates = np.flatnonzero(variance > 0)
    if len(candidates) < n_features:
        candidates = np.arange(len(names))

    order = sorted(candidates, key=lambda i: (-variance[i], str(names[i])))
    chosen = sorted(order[:n_features])
    return [str(names[i]) for i in chosen]


def required_genes(view: str, gene_lists: Mapping[str, Sequence[str]], available: pd.Index) -> list[str]:
    """Return every gene of the given lists, checking the source holds all of them.

    Args:
        view: Name of the modality the lists belong to, for the error message.
        gene_lists: Genes per gene-list name for this view.
        available: Feature names of the source modality.

    Returns:
        The genes of all lists, each once, in first-appearance order.

    Raises:
        CurationError: If any list names a gene the source modality lacks. The message
            names every affected list, its missing count and the first few genes.
    """
    present = set(available)
    problems: list[str] = []
    for name, genes in gene_lists.items():
        missing = [gene for gene in dict.fromkeys(genes) if gene not in present]
        if missing:
            preview = ", ".join(missing[:_MISSING_PREVIEW]) + (", ..." if len(missing) > _MISSING_PREVIEW else "")
            problems.append(f"'{name}' lacks {len(missing)} of {len(set(genes))} genes ({preview})")
    if problems:
        msg = (
            f"The source dataset's '{view}' modality misses genes of drevalpy gene lists, so the toy dataset "
            f"could not hold them: {'; '.join(problems)}. Pick a source that has them or fix the lists."
        )
        raise CurationError(msg)
    return list(dict.fromkeys(gene for genes in gene_lists.values() for gene in genes))


def select_features(adata: ad.AnnData, n_features: int, required: Sequence[str]) -> list[str]:
    """Return the top-variance features together with every required one.

    Args:
        adata: A modality already subset to the selected samples.
        n_features: Number of top-variance features to keep.
        required: Features to keep regardless of their variance, all present in ``adata``.

    Returns:
        Feature names, in the source ``var`` order.
    """
    keep = set(top_variance_features(adata, n_features)) | set(required)
    return [str(name) for name in adata.var_names if name in keep]
