"""Map PRISM compounds to PubChem CIDs, using the registered structures to verify the name.

PRISM is mapped apart from the other datasets because it is the only one whose mapper reads another
dataset's raw files (the sibling release, for compounds one release leaves unnamed) and so needs its own
DVC dependencies; see ``prism_structures.resolve_prism_drug`` for the resolution rule.
"""

import logging
import re
import time

import pandas as pd
from upath import UPath

from data_curation.cli._progress import track
from data_curation.core import layout
from data_curation.core.config import dataset_config
from data_curation.core.paths import find_file
from data_curation.drugs import map_drugs
from data_curation.drugs.prism_structures import connectivity_key_from_smiles, resolve_prism_drug

logger = logging.getLogger(__name__)

# "BRD-K88510285" of "BRD-K88510285-001-50-9": the compound, without salt form and batch.
_BROAD_STEM = re.compile(r"^BRD-[A-Z]\d+")


def _treatment_drugs(raw_dir: UPath) -> pd.DataFrame:
    """One row per ``broad_id`` of a release's replicate treatment info, with name and smiles columns."""
    treatment_info = pd.read_csv(str(find_file(raw_dir, "*replicate-treatment-info*")))
    # One row per broad_id is sufficient; smiles and name are drug-level.
    return (
        treatment_info[["broad_id", "name", "smiles"]]
        .drop_duplicates(subset="broad_id")
        .dropna(subset=["broad_id"])
        .reset_index(drop=True)
    )


def _stem(broad_id: object) -> str:
    match = _BROAD_STEM.match(str(broad_id).strip())
    return match.group(0) if match else ""


def _sibling_row(broad_id: str, sibling: pd.DataFrame) -> tuple[pd.Series | None, str]:
    """Find the sibling release's row for a compound: exact ``broad_id``, else a same-stem row.

    A stem match is only trusted when every named row of that stem agrees on the name.

    Returns:
        The row (None when nothing matches) and how it matched: "exact", "stem" or "".
    """
    named = sibling[sibling["name"].notna()]
    exact = named[named["broad_id"].astype(str).str.strip() == broad_id]
    if not exact.empty:
        return exact.iloc[0], "exact"
    stem = _stem(broad_id)
    same_stem = named[named["broad_id"].map(_stem) == stem] if stem else named.iloc[0:0]
    if not same_stem.empty and same_stem["name"].str.strip().nunique() == 1:
        return same_stem.iloc[0], "stem"
    return None, ""


def fill_from_sibling(drugs: pd.DataFrame, sibling: pd.DataFrame) -> pd.DataFrame:
    """Take the name and smiles of compounds a release leaves blank from the sibling release.

    Args:
        drugs: ``broad_id``, ``name`` and ``smiles`` per compound of the release being mapped.
        sibling: The same columns for the release to borrow from.

    Returns:
        A copy of *drugs* where every compound with neither name nor smiles takes both from its match
        in *sibling*. Compounds without a match stay blank.
    """
    filled = drugs.copy()
    blank = filled["name"].isna() & filled["smiles"].isna()
    for position in filled.index[blank]:
        broad_id = str(filled.at[position, "broad_id"]).strip()
        row, how = _sibling_row(broad_id, sibling)
        if row is None:
            logger.warning("No name or structure for %s, and none in the sibling release", broad_id)
            continue
        logger.info("Filled %s from sibling release %s match %s: '%s'", broad_id, how, row["broad_id"], row["name"])
        filled.at[position, "name"] = row["name"]
        filled.at[position, "smiles"] = row["smiles"]
    return filled


def _map_prism(raw_dir: UPath, dataset: str) -> pd.DataFrame:
    """Map PRISM drugs by name, verified (and where needed replaced) by the registered structures.

    ``broad_id`` is the native identifier the preparer emits. The treatment-info ``smiles`` field can list
    several structures, so it is parsed and compared with the name's PubChem record rather than sent to
    PubChem. Compounds a release leaves unnamed borrow from ``drug_mapping.name_fallback_from`` if configured.
    """
    drugs = _treatment_drugs(raw_dir)
    sibling_dataset = dataset_config(dataset).get("drug_mapping", {}).get("name_fallback_from")
    if sibling_dataset:
        drugs = fill_from_sibling(drugs, _treatment_drugs(layout.raw_dir("response", sibling_dataset)))

    smiles_by_cid: dict[int, str | None] = {}

    def lookup_name(name: str) -> int | None:
        cid, smiles = map_drugs._resolve_by_name(name, "")
        if cid is not None:
            smiles_by_cid[cid] = smiles
        return cid

    def lookup_inchikey(key: str) -> int | None:
        time.sleep(map_drugs._REQUEST_DELAY)
        cid, smiles = map_drugs._lookup_pubchem(key, namespace="inchikey")
        if cid is not None:
            smiles_by_cid.setdefault(cid, smiles)
        return cid

    def cid_connectivity(cid: int) -> str | None:
        return connectivity_key_from_smiles(smiles_by_cid.get(cid))

    results = []
    for _, row in track(list(drugs.iterrows()), "Mapping PRISM drugs", total=len(drugs)):
        broad_id = str(row["broad_id"]).strip()
        drug_name = str(row["name"]).strip() if pd.notna(row.get("name")) else broad_id
        smiles = str(row["smiles"]).strip() if pd.notna(row.get("smiles")) else ""

        resolution = resolve_prism_drug(
            drug_name,
            smiles,
            lookup_name=lookup_name,
            lookup_inchikey=lookup_inchikey,
            cid_connectivity=cid_connectivity,
            cid_smiles=smiles_by_cid.get,
        )
        if resolution.note:
            logger.info("PRISM '%s' (%s): %s (%s)", drug_name, broad_id, resolution.method, resolution.note)
        results.append(map_drugs._record(broad_id, drug_name, resolution.cid, smiles, "PRISM"))

    return pd.DataFrame(results)


def main(dataset: str) -> None:
    """Map a PRISM release's drugs to PubChem CIDs.

    Raises:
        CurationError: If a CID absorbs unrelated drugs or too many PubChem lookups fail.
    """
    map_drugs.write_mapping(dataset, lambda raw_dir: _map_prism(raw_dir, dataset))
