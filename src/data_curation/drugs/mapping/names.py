"""Recover a drug's name from the source dataset that screened it.

The ontology names drugs from PubChem, which can only name a real CID. Drugs
whose lookup failed carry a dataset-prefixed pseudo-ID (``GDSC_1195``,
``CTRP_660221``) and so came out as ``unknown``, even though the source dataset
records a perfectly good name. Combinations are the starkest case: they are
pseudo-IDs by design, so every one of them was nameless.

The drug mapping CSVs written by ``map_drugs`` keep the source ``DRUG_NAME``
alongside the assigned ID, which is all that is needed to fill the gap.
"""

import logging

import pandas as pd
from upath import UPath

from data_curation.response.harmonize.structures import apply_structures

logger = logging.getLogger(__name__)


def load_source_drug_names(mapping_dir: UPath, structures: pd.DataFrame | None = None) -> pd.Series:
    """Collect the source drug name of every mapped drug, keyed by drug ID.

    Args:
        mapping_dir: Directory holding the per-dataset drug mapping CSVs.
        structures: If given, the structure table; drug IDs are then rewritten onto
            canonical IDs before the names are collected.

    Returns:
        Series of source drug names indexed by ``pubchem_id``, one row per
        distinct ID. Where datasets disagree on the name of a shared CID the
        alphabetically first is taken, so the result does not depend on file
        ordering.
    """
    frames = []
    for path in sorted(mapping_dir.glob("*.parquet")):
        frame = pd.read_parquet(str(path))
        frame["pubchem_id"] = frame["pubchem_id"].astype(str)
        if structures is not None:
            frame = apply_structures(frame, structures)
        frame = frame[["pubchem_id", "DRUG_NAME"]].copy()
        frame["DRUG_NAME"] = frame["DRUG_NAME"].astype(str).str.strip()
        frames.append(frame)

    if not frames:
        msg = f"No drug mapping files found in {mapping_dir}"
        raise FileNotFoundError(msg)

    combined = pd.concat(frames, ignore_index=True)
    combined = combined[combined["DRUG_NAME"] != ""]
    combined = combined.sort_values(["pubchem_id", "DRUG_NAME"])
    combined = combined.drop_duplicates("pubchem_id")
    return combined.set_index("pubchem_id")["DRUG_NAME"]


def fill_missing_names(names: pd.Series, source_names: pd.Series, unknown: str) -> pd.Series:
    """Substitute the source dataset's name wherever PubChem supplied none.

    PubChem's name wins when it exists, so drugs that resolved to a real CID are
    untouched and keep their canonical name.

    Args:
        names: Current ``drug_name`` column, indexed by drug ID.
        source_names: Source names keyed by drug ID, from ``load_source_drug_names``.
        unknown: Placeholder value marking a missing name.

    Returns:
        ``names`` with the placeholder replaced by the source name where one is
        known.
    """
    missing = names == unknown
    fallback = source_names.reindex(names.index).fillna(unknown)
    filled = names.mask(missing, fallback)

    recovered = int((missing & (filled != unknown)).sum())
    still_missing = int((filled == unknown).sum())
    logger.info(
        "Recovered %d drug name(s) from the source datasets, %d still unnamed",
        recovered,
        still_missing,
    )
    return filled
