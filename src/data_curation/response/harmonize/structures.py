"""Apply the one-structure-per-drug table to a drug mapping.

``map_drugs`` picks ``pubchem_id`` and ``canonical_smiles`` per dataset, so the same
compound can come out with a different SMILES in every study, and the same molecule
can sit under two CIDs. ``resolve_drug_structures`` settles both once, globally, into
``data/03_drugs/structures.parquet``. This module is the reading side: it rewrites a
mapping frame onto the canonical ID and its SMILES, so everything downstream of the
mapping (harmonization, the ontology) sees one ID and one SMILES per drug.
"""

import pandas as pd

from data_curation.core import layout
from data_curation.core.errors import CurationError

#: Defined here rather than in ``core/layout.py``: every stage, the fits included, depends on that file.
STRUCTURES_PATH = layout.DRUGS / "structures.parquet"

#: Separator of ``component_pubchem_ids``, as written by ``drugs/mapping/combinations.py``.
_COMPONENT_SEPARATOR = "|"

STRUCTURE_COLUMNS = ["pubchem_id", "canonical_id", "canonical_smiles", "inchikey"]


def read_structures() -> pd.DataFrame:
    """Read ``structures.parquet`` with string identifiers.

    Raises:
        CurationError: If the table has not been built.
    """
    path = STRUCTURES_PATH
    if not path.exists():
        msg = f"{path} does not exist; run the resolve_drug_structures stage first"
        raise CurationError(msg)
    structures = pd.read_parquet(str(path))
    for column in ("pubchem_id", "canonical_id"):
        structures[column] = structures[column].astype(str)
    return structures


def canonical_id_map(structures: pd.DataFrame) -> dict[str, str]:
    """Map each source drug ID to its canonical ID."""
    return dict(zip(structures["pubchem_id"], structures["canonical_id"], strict=True))


def apply_structures(drug_mapping: pd.DataFrame, structures: pd.DataFrame) -> pd.DataFrame:
    """Rewrite a drug mapping onto canonical IDs and their single SMILES.

    ``pubchem_id`` and ``component_pubchem_ids`` (when present) are replaced by canonical
    IDs and ``canonical_smiles`` by the structure table's SMILES. Component IDs that are
    not drug IDs (the ``unknown`` placeholder) are kept as they are.

    Args:
        drug_mapping: A ``drug_mapping/<ds>.parquet`` table.
        structures: The table from :func:`read_structures`.

    Returns:
        A copy of *drug_mapping*. Several rows can now share a ``pubchem_id``: those are
        native drugs that were remapped onto one canonical ID.

    Raises:
        CurationError: If the mapping holds an ID the structure table does not know,
            i.e. the table is stale relative to the mapping.
    """
    canonical = canonical_id_map(structures)
    smiles = dict(zip(structures["pubchem_id"], structures["canonical_smiles"], strict=True))

    ids = drug_mapping["pubchem_id"].astype(str)
    unknown = sorted(set(ids) - set(canonical))
    if unknown:
        msg = (
            f"{len(unknown)} drug ID(s) in the mapping are missing from {STRUCTURES_PATH}: {unknown[:10]}. "
            "Re-run the resolve_drug_structures stage."
        )
        raise CurationError(msg)

    result = drug_mapping.copy()
    result["pubchem_id"] = ids.map(canonical)
    result["canonical_smiles"] = ids.map(smiles).fillna("")
    if "component_pubchem_ids" in result.columns:
        result["component_pubchem_ids"] = canonical_components(result["component_pubchem_ids"], canonical)
    return result


def canonical_components(components: pd.Series, canonical: dict[str, str]) -> pd.Series:
    """Put pipe-joined component IDs onto canonical IDs; non-drug entries such as ``unknown`` are kept."""
    return components.fillna("").map(
        lambda joined: _COMPONENT_SEPARATOR.join(
            canonical.get(part, part) for part in joined.split(_COMPONENT_SEPARATOR) if part
        )
    )
