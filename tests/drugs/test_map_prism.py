"""PRISM drug mapping: structure-aware resolution and filling blank compounds from the sibling release."""

import pandas as pd
import pytest
from upath import UPath

from data_curation.core.errors import CurationError
from data_curation.drugs import map_drugs, map_prism
from data_curation.drugs.prism_structures import structures_from_field

L_LACTIC = "C[C@H](O)C(=O)O"
D_LACTIC = "C[C@@H](O)C(=O)O"
FLAT_LACTIC = "CC(O)C(=O)O"
ETHANOL = "CCO"


def _treatment_info(path: UPath, rows: list[tuple[str, str | None, str | None]]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows, columns=pd.Index(["broad_id", "name", "smiles"]))
    frame["dose"] = 1.0
    frame.to_csv(str(path / "secondary-screen-replicate-treatment-info.csv"), index=False)


class _FakePubChem:
    """Stands in for ``map_drugs._resolve_by_name`` / ``_lookup_pubchem``; records every query."""

    def __init__(self, names: dict[str, tuple[int, str]], inchikeys: dict[str, tuple[int, str]] | None = None) -> None:
        self.names = names
        self.inchikeys = inchikeys or {}
        self.queries: list[tuple[str, str]] = []

    def resolve_by_name(self, name: str, _synonyms: str) -> tuple[int | None, str | None]:
        self.queries.append(("name", name))
        return self.names.get(name, (None, None))

    def lookup(self, identifier: str, namespace: str = "name") -> tuple[int | None, str | None]:
        self.queries.append((namespace, identifier))
        return self.inchikeys.get(identifier, (None, None))


@pytest.fixture(autouse=True)
def _offline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(map_drugs.time, "sleep", lambda _s: None)
    map_drugs._LOOKUPS.reset()


@pytest.fixture
def pubchem(monkeypatch: pytest.MonkeyPatch) -> _FakePubChem:
    fake = _FakePubChem(
        names={
            "lactate": (612, FLAT_LACTIC),
            "mislabelled": (1, ETHANOL),
            "imatinib": (5291, "CC"),
        },
        inchikeys={structures_from_field(L_LACTIC)[0].inchikey: (107689, FLAT_LACTIC)},
    )
    monkeypatch.setattr(map_drugs, "_resolve_by_name", fake.resolve_by_name)
    monkeypatch.setattr(map_drugs, "_lookup_pubchem", fake.lookup)
    return fake


def _configure(monkeypatch: pytest.MonkeyPatch, dirs: dict[str, UPath], fallback: str | None = None) -> None:
    block = {"drug_mapping": {"name_fallback_from": fallback}} if fallback else {}
    monkeypatch.setattr(map_prism, "dataset_config", lambda _d: block)
    monkeypatch.setattr(map_prism.layout, "raw_dir", lambda _kind, dataset: dirs[dataset])


def _ids(frame: pd.DataFrame) -> dict[str, str]:
    return dict(zip(frame["DRUG_ID"], frame["pubchem_id"], strict=True))


def test_name_verified_by_the_registered_structures(
    monkeypatch: pytest.MonkeyPatch, tmp_dir: UPath, pubchem: _FakePubChem
) -> None:
    _treatment_info(tmp_dir / "A", [("BRD-K1-001", "lactate", f"{L_LACTIC}, {L_LACTIC}, {D_LACTIC}, ")])
    _configure(monkeypatch, {"A": tmp_dir / "A"})

    mapping = map_prism._map_prism(tmp_dir / "A", "A")

    assert _ids(mapping) == {"BRD-K1-001": "612"}
    assert [namespace for namespace, _ in pubchem.queries] == ["name"]
    assert mapping["canonical_smiles"].iloc[0].startswith(L_LACTIC)


def test_structure_replaces_a_name_that_disagrees_or_is_unknown(
    monkeypatch: pytest.MonkeyPatch, tmp_dir: UPath, pubchem: _FakePubChem
) -> None:
    _treatment_info(
        tmp_dir / "A",
        [("BRD-K1-001", "mislabelled", f"{FLAT_LACTIC}, {L_LACTIC}"), ("BRD-K2-001", "nameless", L_LACTIC)],
    )
    _configure(monkeypatch, {"A": tmp_dir / "A"})

    mapping = map_prism._map_prism(tmp_dir / "A", "A")

    assert _ids(mapping) == {"BRD-K1-001": "107689", "BRD-K2-001": "107689"}
    assert ("inchikey", structures_from_field(L_LACTIC)[0].inchikey) in pubchem.queries


def test_unresolvable_and_combination_drugs_keep_pseudo_ids(
    monkeypatch: pytest.MonkeyPatch, tmp_dir: UPath, pubchem: _FakePubChem
) -> None:
    _treatment_info(
        tmp_dir / "A", [("BRD-K1-001", "nothing", ETHANOL), ("BRD-K2-001", "JQ-1:UNC0638 (2:1 mol/mol)", ETHANOL)]
    )
    _configure(monkeypatch, {"A": tmp_dir / "A"})

    mapping = map_prism._map_prism(tmp_dir / "A", "A")

    assert _ids(mapping) == {"BRD-K1-001": "PRISM_BRD-K1-001", "BRD-K2-001": "PRISM_BRD-K2-001"}
    assert ("name", "JQ-1:UNC0638 (2:1 mol/mol)") not in pubchem.queries


def test_blank_compounds_take_name_and_structure_from_the_sibling_release(
    monkeypatch: pytest.MonkeyPatch, tmp_dir: UPath, pubchem: _FakePubChem
) -> None:
    _treatment_info(tmp_dir / "old", [("BRD-K5-001", "imatinib", "CC"), ("BRD-K6-001", "lactate", FLAT_LACTIC)])
    _treatment_info(
        tmp_dir / "new",
        [
            ("BRD-K5-001", None, None),
            ("BRD-K6-009", None, None),
            ("BRD-K7-001", None, None),
            ("BRD-K1-001", "lactate", L_LACTIC),
        ],
    )
    _configure(monkeypatch, {"old": tmp_dir / "old", "new": tmp_dir / "new"}, fallback="old")

    mapping = map_prism._map_prism(tmp_dir / "new", "new")

    assert _ids(mapping) == {
        "BRD-K5-001": "5291",  # exact broad_id
        "BRD-K6-009": "612",  # same compound, different batch
        "BRD-K7-001": "PRISM_BRD-K7-001",  # nowhere to be found
        "BRD-K1-001": "612",
    }
    assert mapping.set_index("DRUG_ID").loc["BRD-K5-001", "DRUG_NAME"] == "imatinib"


def test_no_fallback_configured_leaves_blank_compounds_alone(
    monkeypatch: pytest.MonkeyPatch, tmp_dir: UPath, pubchem: _FakePubChem
) -> None:
    _treatment_info(tmp_dir / "new", [("BRD-K5-001", None, None)])
    _configure(monkeypatch, {"new": tmp_dir / "new"})

    mapping = map_prism._map_prism(tmp_dir / "new", "new")

    assert _ids(mapping) == {"BRD-K5-001": "PRISM_BRD-K5-001"}
    assert mapping["DRUG_NAME"].tolist() == ["BRD-K5-001"]


def test_exact_match_beats_stem_and_conflicting_stems_are_not_trusted() -> None:
    sibling = pd.DataFrame(
        {
            "broad_id": ["BRD-K1-001", "BRD-K1-002", "BRD-K2-001", "BRD-K2-002"],
            "name": ["alpha", "alpha-other", "beta", "gamma"],
            "smiles": ["C", "CC", "CCC", "CCCC"],
        }
    )
    drugs = pd.DataFrame(
        {
            "broad_id": ["BRD-K1-002", "BRD-K2-009", "BRD-K1-001", "BRD-K3-001"],
            "name": [None, None, "kept", None],
            "smiles": [None, None, "C", None],
        }
    )

    filled = map_prism.fill_from_sibling(drugs, sibling).set_index("broad_id")

    assert filled.loc["BRD-K1-002", "name"] == "alpha-other"
    assert pd.isna(filled.loc["BRD-K2-009", "name"])
    assert filled.loc["BRD-K1-001", "name"] == "kept"
    assert pd.isna(filled.loc["BRD-K3-001", "name"])


def test_only_compounds_missing_both_name_and_smiles_are_filled() -> None:
    sibling = pd.DataFrame({"broad_id": ["BRD-K1-001"], "name": ["alpha"], "smiles": ["C"]})
    drugs = pd.DataFrame({"broad_id": ["BRD-K1-001"], "name": [None], "smiles": ["CC"]})

    assert map_prism.fill_from_sibling(drugs, sibling)["name"].isna().all()


def test_main_writes_the_mapping_through_the_shared_checks(
    monkeypatch: pytest.MonkeyPatch, tmp_dir: UPath, pubchem: _FakePubChem
) -> None:
    _treatment_info(tmp_dir / "A", [("BRD-K1-001", "lactate", L_LACTIC)])
    _configure(monkeypatch, {"A": tmp_dir / "A"})
    output = tmp_dir / "mapping" / "A.parquet"
    monkeypatch.setattr(map_drugs.layout, "raw_dir", lambda *_a: tmp_dir / "A")
    monkeypatch.setattr(map_drugs.layout, "drug_mapping", lambda _d: output)
    monkeypatch.setattr(map_drugs, "processing_config", lambda _k: {"max_failure_fraction": 0.5})

    map_prism.main("A")

    mapping = pd.read_parquet(str(output))
    assert mapping["pubchem_id"].tolist() == ["612"]
    assert (tmp_dir / "mapping" / "A_pubchem_failures.tsv").exists()


def test_the_generic_mapper_points_prism_datasets_to_their_own_command(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(map_drugs, "dataset_config", lambda *_a: {"format": "prism"})

    with pytest.raises(CurationError, match="map-prism"):
        map_drugs.main("PRISM_20Q2")
