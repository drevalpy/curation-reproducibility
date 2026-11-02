"""Detect combination-therapy drugs and resolve their component compounds.

Two datasets encode combinations in the drug name itself:

- CTRPv2: ``JQ-1:UNC0638 (2:1 mol/mol)`` - colon-separated, with a trailing
  molar-ratio suffix that also contains a colon.
- PDX_Bruna: ``Olaparib(1495) + Temozolomide(1375)`` - plus-separated, with the
  dataset's own drug ID inline.

A combination has no single PubChem CID, so it must never be resolved to one.
Some CTRPv2 combinations carry a mixture SMILES that PubChem happily matches to
an unrelated single compound, which silently conflated three combinations with
real drugs. Combinations therefore keep their dataset-prefixed pseudo-ID and
record their components in a separate column instead.
"""

import logging
import re
from collections.abc import Callable

import pandas as pd
from upath import UPath

logger = logging.getLogger(__name__)

COMBINATION_COLUMNS = ["is_combination", "n_components", "component_pubchem_ids"]

#: Separator for the pipe-joined component ID list.
ID_SEPARATOR = "|"
#: Placeholder for a component whose identity could not be established.
UNRESOLVED = "unknown"

_MIN_COMPONENTS = 2

# " (2:1 mol/mol)", " (40:17 mol/mol)": stripped before splitting, otherwise the
# ratio's own colon would be read as a component separator.
_RATIO_SUFFIX = re.compile(r"\s*\(\s*\d+(?:\.\d+)?\s*:\s*\d+(?:\.\d+)?[^()]*\)\s*$")
# "Olaparib(1495)": an inline dataset drug ID is not part of the compound name.
_INLINE_ID = re.compile(r"\(\s*\d+\s*\)")


def split_components(drug_name: str) -> list[str]:
    """Split a combination drug name into its component compound names.

    Args:
        drug_name: Drug name as it appears in the raw dataset.

    Returns:
        The component names, or an empty list if the name is a single agent.
    """
    base = _RATIO_SUFFIX.sub("", str(drug_name).strip())
    separator = "+" if "+" in base else ":" if ":" in base else ""
    if not separator:
        return []

    parts = [_INLINE_ID.sub("", part).strip() for part in base.split(separator)]
    components = [part for part in parts if part]
    return components if len(components) >= _MIN_COMPONENTS else []


def is_combination(drug_name: str) -> bool:
    """Report whether a drug name describes a combination of two or more compounds."""
    return bool(split_components(drug_name))


def _single_agent_ids(frame: pd.DataFrame) -> dict[str, str]:
    """Index the single agents of a mapping frame by lowercased name."""
    names = frame["DRUG_NAME"].astype(str)
    ids = frame["pubchem_id"].astype(str)
    return {name.strip().lower(): drug_id for name, drug_id in zip(names, ids, strict=True) if not is_combination(name)}


def _resolve_component(name: str, single_agents: dict[str, str], lookup: Callable[[str], int | None]) -> str:
    """Resolve one component name to an ID, preferring the in-dataset mapping.

    Most components are also screened as single agents in the same dataset, so
    reusing that ID both avoids an API call and keeps the component pointing at
    the dataset's own column for that compound.
    """
    known = single_agents.get(name.lower())
    if known:
        return known

    cid = lookup(name)
    if cid is not None:
        logger.info("Resolved combination component '%s' -> CID %s", name, cid)
        return str(cid)

    logger.warning("No PubChem match for combination component '%s'", name)
    return UNRESOLVED


def annotate_combinations(frame: pd.DataFrame, lookup: Callable[[str], int | None]) -> pd.DataFrame:
    """Add the combination columns to a drug mapping frame.

    Args:
        frame: Drug mapping with DRUG_NAME and pubchem_id columns.
        lookup: Fallback name -> CID resolver for components that are not
            screened as single agents in this dataset.

    Returns:
        A copy of ``frame`` with ``is_combination``, ``n_components`` and
        pipe-separated ``component_pubchem_ids`` appended. Single agents get
        ``False``, ``0`` and an empty string.
    """
    single_agents = _single_agent_ids(frame)
    resolved: dict[str, str] = {}

    flags: list[bool] = []
    counts: list[int] = []
    component_ids: list[str] = []
    for drug_name in frame["DRUG_NAME"].astype(str):
        components = split_components(drug_name)
        for name in components:
            if name.lower() not in resolved:
                resolved[name.lower()] = _resolve_component(name, single_agents, lookup)
        flags.append(bool(components))
        counts.append(len(components))
        component_ids.append(ID_SEPARATOR.join(resolved[name.lower()] for name in components))

    annotated = frame.copy()
    annotated["is_combination"] = flags
    annotated["n_components"] = counts
    annotated["component_pubchem_ids"] = component_ids

    if any(flags):
        logger.info("Annotated %d combination(s) among %d drugs", sum(flags), len(annotated))
    return annotated


def load_combination_annotations(mapping_dir: UPath) -> pd.DataFrame:
    """Collect the combination columns of every drug mapping file, keyed by drug ID.

    Args:
        mapping_dir: Directory holding the per-dataset drug mapping Parquet files.

    Returns:
        DataFrame indexed by ``pubchem_id`` with the combination columns, one row
        per distinct ID. A drug screened by several datasets appears once;
        combination rows win over single-agent rows so a shared ID is never
        downgraded.
    """
    frames = []
    for path in sorted(mapping_dir.glob("*.parquet")):
        frame = pd.read_parquet(str(path))[["pubchem_id", *COMBINATION_COLUMNS]]
        frame["pubchem_id"] = frame["pubchem_id"].astype(str)
        frames.append(frame)

    if not frames:
        msg = f"No drug mapping files found in {mapping_dir}"
        raise FileNotFoundError(msg)

    combined = pd.concat(frames, ignore_index=True)
    combined["component_pubchem_ids"] = combined["component_pubchem_ids"].fillna("")
    combined = combined.sort_values("is_combination", ascending=False)
    return combined.drop_duplicates("pubchem_id").set_index("pubchem_id")
