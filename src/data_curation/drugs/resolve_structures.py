"""Give every drug exactly one canonical ID and one SMILES across all datasets.

``map_drugs`` takes ``canonical_smiles`` from whichever source its dataset offers
(PubChem's connectivity SMILES without stereo, the study's own SMILES, a PRISM
multi-structure string), so one CID carried a different SMILES per dataset, and a
molecule could sit under two CIDs (sildenafil, XAV-939, pemetrexed). This stage
settles both once, for the union of the drugs in ``data/03_drugs/mapping/``:

* **SMILES.** The CID's isomeric PubChem SMILES, rewritten with RDKit's canonical
  writer, so the string does not depend on how PubChem happens to order atoms.
* **Identity.** CIDs with the same InChIKey are one molecule; every one of them is
  remapped onto the lowest numeric CID of the group, and the group shares that CID's
  SMILES.
* **No CID.** Pseudo-IDs (``GDSC_<id>``, combinations, ...) and CIDs PubChem returns
  nothing for keep the study's SMILES, canonicalized if RDKit parses it and left
  unchanged otherwise (PRISM's comma-separated multi-structure strings do not parse).

The raw PubChem responses are cached in ``pubchem_structures_cache.json`` (a persistent
output, so DVC leaves it in place when the stage re-runs) and a re-run only asks for CIDs
that are not cached yet. Delete a CID's entry to force a new lookup.

A failing PubChem request aborts the stage rather than falling back to the study's SMILES:
a silent fallback would make the table depend on whether PubChem was reachable.
"""

import json
import logging
import time

import pandas as pd
from rdkit import Chem, rdBase
from upath import UPath

from data_curation.cli._progress import track
from data_curation.core import layout
from data_curation.drugs.ontology.pubchem import _PUG_REST, _batched, _request
from data_curation.response.harmonize.structures import STRUCTURES_PATH

logger = logging.getLogger(__name__)

#: Defined here rather than in ``core/layout.py``: every stage, the fits included, depends on that file.
CACHE_PATH = layout.DRUGS / "pubchem_structures_cache.json"

#: One PubChem request per this many CIDs; the same size the InChIKey lookup uses.
_BATCH = 150
#: Pause between requests; PubChem asks for no more than 5 per second.
_REQUEST_DELAY = 0.25

#: Property table columns. ``SMILES`` is PubChem's isomeric SMILES; ``CanonicalSMILES`` and
#: ``IsomericSMILES`` are deprecated names (``ConnectivitySMILES``, the stereo-free one, is
#: what pubchempy's ``canonical_smiles`` returns).
_PROPERTIES = "SMILES,InChIKey"

Cache = dict[str, dict[str, str] | None]


def is_cid(drug_id: str) -> bool:
    """Whether a drug ID is a numeric PubChem CID, as opposed to a dataset pseudo-ID."""
    return drug_id.isdigit()


def canonicalize(smiles: str) -> str:
    """Rewrite a SMILES with RDKit's canonical writer, stereochemistry included.

    Returns:
        The canonical SMILES, or *smiles* unchanged if it is empty or RDKit cannot parse it.
    """
    if not smiles:
        return ""
    with rdBase.BlockLogs():
        mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol) if mol is not None else smiles


def load_study_structures(mapping_dir: UPath) -> pd.DataFrame:
    """Collect the study-side SMILES of every drug ID in the mapping files.

    Args:
        mapping_dir: Directory of ``<dataset>.parquet`` drug mappings.

    Returns:
        One row per distinct ``pubchem_id`` with the ``study_smiles`` of the first
        dataset (in name order) that has a non-empty one.

    Raises:
        FileNotFoundError: If there are no mapping files.
    """
    frames = []
    for path in sorted(mapping_dir.glob("*.parquet")):
        frame = pd.read_parquet(str(path))[["pubchem_id", "canonical_smiles"]]
        frame["pubchem_id"] = frame["pubchem_id"].astype(str)
        frame["canonical_smiles"] = frame["canonical_smiles"].fillna("").astype(str).str.strip()
        frames.append(frame)
    if not frames:
        msg = f"No drug mapping files found in {mapping_dir}"
        raise FileNotFoundError(msg)

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values("canonical_smiles", key=lambda s: s == "", kind="stable")
    first = combined.drop_duplicates("pubchem_id").set_index("pubchem_id")["canonical_smiles"]
    return first.rename("study_smiles").sort_index().reset_index()


def _read_cache(path: UPath) -> Cache:
    return json.loads(path.read_text()) if path.exists() else {}


def _write_cache(path: UPath, cache: Cache) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, sort_keys=True, indent=0))


def _parse_properties(payload: dict) -> Cache:
    """Extract ``{cid: {smiles, inchikey}}`` from one property table; a CID without a record maps to ``None``."""
    parsed: Cache = {}
    for record in payload.get("PropertyTable", {}).get("Properties", []):
        smiles = record.get("SMILES") or record.get("IsomericSMILES")
        parsed[str(record["CID"])] = {"smiles": smiles, "inchikey": record.get("InChIKey", "")} if smiles else None
    return parsed


def fetch_structures(cids: list[str], cache_path: UPath) -> Cache:
    """Look up PubChem's isomeric SMILES and InChIKey, using and extending the cache.

    Args:
        cids: Numeric PubChem CIDs as strings.
        cache_path: JSON file of previous responses. It is rewritten after every request, so an
            interrupted run resumes where it stopped.

    Returns:
        ``{cid: {"smiles", "inchikey"}}`` for every requested CID, ``None`` where PubChem
        has no record.

    Raises:
        PubChemUnavailable: If a request fails on every retry.
    """
    cache = _read_cache(cache_path)
    missing = [cid for cid in cids if cid not in cache]
    logger.info("%d CIDs requested, %d cached, %d to fetch", len(cids), len(cids) - len(missing), len(missing))

    for batch in track(_batched(missing, _BATCH), "Fetching PubChem structures"):
        payload = _request(f"{_PUG_REST}/compound/cid/{','.join(batch)}/property/{_PROPERTIES}/JSON")
        found = _parse_properties(payload)
        cache.update({cid: found.get(cid) for cid in batch})
        _write_cache(cache_path, cache)
        time.sleep(_REQUEST_DELAY)

    return {cid: cache[cid] for cid in cids}


def _group_by_inchikey(structures: dict[str, dict[str, str] | None]) -> dict[str, str]:
    """Map every CID that PubChem knows to the lowest CID sharing its InChIKey."""
    groups: dict[str, list[int]] = {}
    for cid, record in structures.items():
        if record and record["inchikey"]:
            groups.setdefault(record["inchikey"], []).append(int(cid))
    return {str(cid): str(min(group)) for group in groups.values() for cid in group}


def resolve_structures(study: pd.DataFrame, fetched: dict[str, dict[str, str] | None]) -> pd.DataFrame:
    """Build the structure table from the study SMILES and PubChem's records.

    Args:
        study: Output of :func:`load_study_structures`.
        fetched: Output of :func:`fetch_structures` for the CIDs in *study*.

    Returns:
        One row per source drug ID with ``pubchem_id``, ``canonical_id``, ``canonical_smiles``
        and ``inchikey`` (empty without a PubChem record).
    """
    lowest = _group_by_inchikey(fetched)
    smiles_of: dict[str, str] = {}
    for cid, record in fetched.items():
        if record:
            smiles_of[cid] = canonicalize(record["smiles"])

    rows = []
    for drug_id, study_smiles in zip(study["pubchem_id"], study["study_smiles"], strict=True):
        record = fetched.get(drug_id)
        canonical_id = lowest.get(drug_id, drug_id)
        if record:
            smiles, inchikey = smiles_of[canonical_id], record["inchikey"]
        else:
            smiles, inchikey = canonicalize(study_smiles), ""
            if is_cid(drug_id):
                logger.warning("PubChem has no record for CID %s; keeping the study SMILES", drug_id)
        rows.append(
            {"pubchem_id": drug_id, "canonical_id": canonical_id, "canonical_smiles": smiles, "inchikey": inchikey}
        )
    return pd.DataFrame(rows)


def _log_remaps(table: pd.DataFrame) -> None:
    remapped = table[table["pubchem_id"] != table["canonical_id"]]
    logger.info("%d of %d drug IDs remapped onto a lower CID with the same InChIKey", len(remapped), len(table))
    for row in remapped.itertuples():
        logger.info("  %s -> %s (%s)", row.pubchem_id, row.canonical_id, row.inchikey)


def main() -> None:
    """Resolve one canonical ID and SMILES per drug and write ``structures.parquet``."""
    study = load_study_structures(layout.DRUG_MAPPING_DIR)
    cids = [drug_id for drug_id in study["pubchem_id"] if is_cid(drug_id)]
    logger.info("%d distinct drug IDs, %d of them CIDs", len(study), len(cids))

    fetched = fetch_structures(sorted(cids, key=int), CACHE_PATH)
    table = resolve_structures(study, fetched)
    _log_remaps(table)

    STRUCTURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(str(STRUCTURES_PATH), index=False)
    logger.info("Wrote %d structures to %s", len(table), STRUCTURES_PATH)
