"""Keep the replicate-fits table (``response.uns["replicate_fits"]``) on the response's identifiers.

``drevalpy.curation.curate()`` builds the table in ``fit_curves``: one row per dose series of a
(sample, drug) pair, keyed by the native labels that were fitted (``sample_id``/``pubchem_id``
equal ``native_sample``/``native_drug`` there). This module holds the two steps that follow:

* :func:`canonicalize`, in ``harmonize_response``, renames the keys to the canonical sample IDs
  (Cellosaurus accessions for cell lines) and PubChem CIDs. No collapsing happens: when several
  native identifiers map onto one canonical one, each keeps its rows, and
  ``native_sample``/``native_drug`` tell them apart.
* :func:`restrict`, in ``assemble_mudata`` and ``subsample_toy``, keeps the table aligned with a
  response modality's axes.

drevalpy's ``drevalpy.curation._replicate_fits`` writes the format and
``drevalpy.data._replicate_fits`` reads it.
"""

import pandas as pd

from data_curation.core.layout import SAMPLE_INDEX_NAME

#: Key of the table in ``response.uns``.
KEY = "replicate_fits"

#: Canonical drug column, matching the response ``var`` index name.
DRUG_COLUMN = "pubchem_id"


def canonicalize(table: pd.DataFrame, sample_map: pd.Series, drug_map: pd.Series) -> pd.DataFrame:
    """Re-key the table on canonical identifiers and drop rows whose sample or drug is unresolved.

    Args:
        table: The table as ``curate()`` wrote it, keyed by native identifiers.
        sample_map: Canonical sample ID per resolved native sample.
        drug_map: Canonical drug ID per resolved native drug.

    Returns:
        The table with canonical ``sample_id`` and ``pubchem_id``, unresolved rows removed, its
        categorical columns holding only the categories still used.
    """
    samples = table["native_sample"].astype(str).map(sample_map)
    drugs = table["native_drug"].astype(str).map(drug_map)
    keep = (samples.notna() & drugs.notna()).to_numpy()
    out = table.loc[keep].copy()
    out[SAMPLE_INDEX_NAME] = pd.Categorical(samples[keep].astype(str).to_numpy())
    out[DRUG_COLUMN] = pd.Categorical(drugs[keep].astype(str).to_numpy())
    return _without_unused_categories(out.reset_index(drop=True))


def restrict(table: pd.DataFrame, samples: pd.Index, drugs: pd.Index) -> pd.DataFrame:
    """Keep the rows whose canonical sample and drug are in *samples* and *drugs*."""
    keep = table[SAMPLE_INDEX_NAME].astype(str).isin(samples) & table[DRUG_COLUMN].astype(str).isin(drugs)
    return _without_unused_categories(table.loc[keep.to_numpy()].reset_index(drop=True))


def _without_unused_categories(table: pd.DataFrame) -> pd.DataFrame:
    """Drop the categories no row uses any more."""
    for column in table.columns:
        if isinstance(table[column].dtype, pd.CategoricalDtype):
            table[column] = table[column].cat.remove_unused_categories()
    return table
