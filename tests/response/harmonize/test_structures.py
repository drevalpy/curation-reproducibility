"""Rewriting a drug mapping onto canonical IDs and SMILES."""

import pandas as pd
import pytest

from data_curation.core.errors import CurationError
from data_curation.response.harmonize.identity import resolve_drugs
from data_curation.response.harmonize.structures import apply_structures, canonical_components

STRUCTURES = pd.DataFrame(
    {
        "pubchem_id": ["5212", "135398744", "GDSC_9"],
        "canonical_id": ["5212", "5212", "GDSC_9"],
        "canonical_smiles": ["LOW", "LOW", "X"],
        "inchikey": ["K", "K", ""],
    }
)


def _mapping(**columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame({"DRUG_ID": ["a", "b", "c"], "pubchem_id": ["5212", "135398744", "GDSC_9"], **columns})


def test_ids_and_smiles_are_replaced() -> None:
    result = apply_structures(_mapping(canonical_smiles=["old", "old2", ""]), STRUCTURES)
    assert result["pubchem_id"].tolist() == ["5212", "5212", "GDSC_9"]
    assert result["canonical_smiles"].tolist() == ["LOW", "LOW", "X"]


def test_the_native_ids_survive_so_both_natives_resolve_to_the_canonical_id() -> None:
    result = apply_structures(_mapping(canonical_smiles=["", "", ""]), STRUCTURES)
    resolved = resolve_drugs(pd.Index(["a", "b", "c"]), result)
    assert resolved.tolist() == ["5212", "5212", "GDSC_9"]


def test_the_input_is_not_modified() -> None:
    mapping = _mapping(canonical_smiles=["old", "old", "old"])
    apply_structures(mapping, STRUCTURES)
    assert mapping["pubchem_id"].tolist() == ["5212", "135398744", "GDSC_9"]


def test_component_ids_are_remapped_and_placeholders_kept() -> None:
    mapping = _mapping(canonical_smiles=["", "", ""], component_pubchem_ids=["135398744|unknown", "", ""])
    result = apply_structures(mapping, STRUCTURES)
    assert result["component_pubchem_ids"].tolist() == ["5212|unknown", "", ""]


def test_a_mapping_id_missing_from_the_table_is_an_error() -> None:
    mapping = _mapping(canonical_smiles=["", "", ""])
    mapping.loc[0, "pubchem_id"] = "42"
    with pytest.raises(CurationError, match="resolve_drug_structures"):
        apply_structures(mapping, STRUCTURES)


def test_canonical_components() -> None:
    components = pd.Series(["135398744|135398744", "7"])
    assert canonical_components(components, {"135398744": "5212"}).tolist() == ["5212|5212", "7"]
