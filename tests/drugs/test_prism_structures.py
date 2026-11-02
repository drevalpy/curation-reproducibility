"""PRISM structure parsing and the name/structure resolution rule, with fake PubChem lookups."""

import numpy as np
from rdkit import Chem

from data_curation.drugs import prism_structures as ps

L_LACTIC = "C[C@H](O)C(=O)O"
D_LACTIC = "C[C@@H](O)C(=O)O"
FLAT_LACTIC = "CC(O)C(=O)O"
ETHANOL = "CCO"
SODIUM_ACETATE = "CC(=O)[O-].[Na+]"
ACETIC_ACID = "CC(=O)O"


def _key(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None
    return ps.connectivity_key(mol)


def _full_key(smiles: str) -> str:
    return ps.structures_from_field(smiles)[0].inchikey


class _Fake:
    """Fake PubChem: name -> CID, InChIKey -> CID, CID -> connectivity key; records every call."""

    def __init__(self, names=None, inchikeys=None, connectivity=None) -> None:
        self.names = names or {}
        self.inchikeys = inchikeys or {}
        self.connectivity = connectivity or {}
        self.calls: list[tuple[str, str]] = []

    def lookup_name(self, name: str) -> int | None:
        self.calls.append(("name", name))
        return self.names.get(name)

    def lookup_inchikey(self, key: str) -> int | None:
        self.calls.append(("inchikey", key))
        return self.inchikeys.get(key)

    def cid_connectivity(self, cid: int) -> str | None:
        self.calls.append(("cid", str(cid)))
        return self.connectivity.get(cid)

    def resolve(self, name: str, field: object) -> ps.Resolution:
        return ps.resolve_prism_drug(
            name,
            field,
            lookup_name=self.lookup_name,
            lookup_inchikey=self.lookup_inchikey,
            cid_connectivity=self.cid_connectivity,
        )


def test_identical_copies_collapse() -> None:
    assert ps.parse_smiles_field(f"{ETHANOL}, {ETHANOL}, OCC") == [ETHANOL]


def test_empty_entries_dropped() -> None:
    assert ps.parse_smiles_field(f"{ETHANOL}, , ,") == [ETHANOL]
    assert ps.parse_smiles_field("") == []
    assert ps.parse_smiles_field(np.nan) == []
    assert ps.parse_smiles_field(None) == []


def test_invalid_smiles_dropped() -> None:
    assert ps.parse_smiles_field(f"not-a-smiles, {ETHANOL}, C1CC") == [ETHANOL]


def test_order_is_preserved() -> None:
    assert ps.parse_smiles_field(f"{ACETIC_ACID}, {ETHANOL}, {ACETIC_ACID}") == [ACETIC_ACID, ETHANOL]


def test_stereo_variants_share_connectivity() -> None:
    assert _key(L_LACTIC) == _key(D_LACTIC) == _key(FLAT_LACTIC)
    structures = ps.structures_from_field(f"{FLAT_LACTIC}, {L_LACTIC}, {D_LACTIC}")
    assert {s.connectivity for s in structures} == {_key(FLAT_LACTIC)}
    assert len({s.inchikey for s in structures}) == 3


def test_most_stereo_specific_structure_first() -> None:
    structures = ps.structures_from_field(f"{FLAT_LACTIC}, {L_LACTIC}")
    assert [s.n_stereo for s in structures] == [1, 0]


def test_double_bond_stereo_counts() -> None:
    assert ps.structures_from_field("C/C=C/C")[0].n_stereo == 1
    assert ps.structures_from_field("CC=CC")[0].n_stereo == 0


def test_salt_stripped_to_parent() -> None:
    assert _key(SODIUM_ACETATE) == _key(ACETIC_ACID)
    assert _key("CC(=O)O.O") == _key(ACETIC_ACID)
    assert _key("CCO") != _key(ACETIC_ACID)


def test_connectivity_key_from_smiles() -> None:
    assert ps.connectivity_key_from_smiles(SODIUM_ACETATE) == _key(ACETIC_ACID)
    assert ps.connectivity_key_from_smiles("nonsense") is None
    assert ps.connectivity_key_from_smiles(None) is None


def test_name_verified_by_matching_structure() -> None:
    fake = _Fake(names={"lactate": 612}, connectivity={612: _key(FLAT_LACTIC)})
    result = fake.resolve("lactate", f"{L_LACTIC}, {L_LACTIC}")
    assert result == ps.Resolution(612, ps.NAME_VERIFIED)
    assert not any(kind == "inchikey" for kind, _ in fake.calls)


def test_name_verified_against_salt_record() -> None:
    fake = _Fake(names={"acetate": 517045}, connectivity={517045: ps.connectivity_key_from_smiles(SODIUM_ACETATE)})
    assert fake.resolve("acetate", ACETIC_ACID).method == ps.NAME_VERIFIED


def test_name_verified_by_any_variant() -> None:
    fake = _Fake(names={"x": 1}, connectivity={1: _key(ETHANOL)})
    assert fake.resolve("x", f"{ACETIC_ACID}, {ETHANOL}").method == ps.NAME_VERIFIED


def test_mismatch_falls_back_to_most_specific_structure() -> None:
    fake = _Fake(
        names={"drug": 1},
        connectivity={1: _key(ETHANOL)},
        inchikeys={_full_key(L_LACTIC): 107689, _full_key(FLAT_LACTIC): 612},
    )
    result = fake.resolve("drug", f"{FLAT_LACTIC}, {L_LACTIC}")
    assert (result.cid, result.method) == (107689, ps.STRUCTURE)
    assert "name CID 1" in result.note
    assert [c for c in fake.calls if c[0] == "inchikey"] == [("inchikey", _full_key(L_LACTIC))]


def test_remaining_structures_tried_when_first_misses() -> None:
    fake = _Fake(inchikeys={_full_key(FLAT_LACTIC): 612})
    result = fake.resolve("unknown name", f"{L_LACTIC}, {FLAT_LACTIC}")
    assert (result.cid, result.method) == (612, ps.STRUCTURE)
    assert result.note == "name not found"
    assert len([c for c in fake.calls if c[0] == "inchikey"]) == 2


def test_inchikey_lookup_uses_standardized_parent() -> None:
    fake = _Fake(inchikeys={_full_key(ACETIC_ACID): 176})
    assert fake.resolve("acetate", SODIUM_ACETATE).cid == 176


def test_no_structures_uses_name_only() -> None:
    fake = _Fake(names={"drug": 5})
    assert fake.resolve("drug", ", ,") == ps.Resolution(5, ps.NAME_ONLY, "no valid structure")
    assert fake.resolve("drug", np.nan).method == ps.NAME_ONLY
    assert fake.resolve("drug", "garbage").cid == 5
    assert not any(kind in {"inchikey", "cid"} for kind, _ in fake.calls)


def test_no_structures_and_no_name_is_unresolved() -> None:
    result = _Fake().resolve("drug", "")
    assert (result.cid, result.method) == (None, ps.UNRESOLVED)


def test_unmatched_name_kept_when_structure_not_in_pubchem() -> None:
    fake = _Fake(names={"drug": 1}, connectivity={1: _key(ETHANOL)})
    result = fake.resolve("drug", ACETIC_ACID)
    assert (result.cid, result.method) == (1, ps.NAME_UNVERIFIED)


def test_unresolved_when_nothing_found() -> None:
    result = _Fake().resolve("drug", ACETIC_ACID)
    assert (result.cid, result.method) == (None, ps.UNRESOLVED)
    assert "structure not in PubChem" in result.note


def test_unknown_cid_connectivity_does_not_verify() -> None:
    fake = _Fake(names={"drug": 1}, inchikeys={_full_key(ETHANOL): 702})
    assert fake.resolve("drug", ETHANOL).cid == 702


def test_combination_keeps_pseudo_id_without_lookups() -> None:
    fake = _Fake(names={"JQ-1:UNC0638 (2:1 mol/mol)": 9})
    result = fake.resolve("JQ-1:UNC0638 (2:1 mol/mol)", ETHANOL)
    assert (result.cid, result.method) == (None, ps.COMBINATION)
    assert fake.calls == []


ACETONE = "CC(C)=O"
PROPEN_2_OL = "CC(=C)O"


def test_tautomer_key_merges_keto_enol_forms() -> None:
    assert _key(ACETONE) != _key(PROPEN_2_OL)
    assert ps.tautomer_key(ACETONE) == ps.tautomer_key(PROPEN_2_OL)
    assert ps.tautomer_key(ACETONE) != ps.tautomer_key(ETHANOL)
    assert ps.tautomer_key("nonsense") is None
    assert ps.tautomer_key(None) is None


def test_name_verified_by_tautomer_when_smiles_available() -> None:
    fake = _Fake(names={"drug": 3}, connectivity={3: _key(PROPEN_2_OL)}, inchikeys={_full_key(ACETONE): 180})
    result = ps.resolve_prism_drug(
        "drug",
        ACETONE,
        lookup_name=fake.lookup_name,
        lookup_inchikey=fake.lookup_inchikey,
        cid_connectivity=fake.cid_connectivity,
        cid_smiles=lambda _cid: PROPEN_2_OL,
    )
    assert result == ps.Resolution(3, ps.NAME_VERIFIED_TAUTOMER)
    assert fake.resolve("drug", ACETONE).cid == 180


def test_tautomer_check_rejects_different_compound() -> None:
    fake = _Fake(names={"drug": 3}, connectivity={3: _key(ETHANOL)}, inchikeys={_full_key(ACETONE): 180})
    result = ps.resolve_prism_drug(
        "drug",
        ACETONE,
        lookup_name=fake.lookup_name,
        lookup_inchikey=fake.lookup_inchikey,
        cid_connectivity=fake.cid_connectivity,
        cid_smiles=lambda _cid: ETHANOL,
    )
    assert (result.cid, result.method) == (180, ps.STRUCTURE)


def test_metal_complex_keeps_its_metal() -> None:
    cisplatin = "N.N.Cl[Pt]Cl"
    assert _key(cisplatin) != _key("N")
    assert "Pt" in Chem.MolToSmiles(ps.standardize_parent(Chem.MolFromSmiles(cisplatin)))
    assert _key("N.N.Cl[Pt]Cl.O") == _key(cisplatin)
    assert _key("[Na+].[Na+].Cl[Pt]Cl.N.N") == _key(cisplatin)


def test_metal_complex_keeps_name_cid_when_unverifiable() -> None:
    cisplatin = "N[Pt](N)(Cl)Cl"
    fake = _Fake(names={"cisplatin": 5702198}, connectivity={5702198: "X"}, inchikeys={_full_key(cisplatin): 2767})
    result = fake.resolve("cisplatin", cisplatin)
    assert (result.cid, result.method) == (5702198, ps.NAME_UNVERIFIED)
    assert not any(kind == "inchikey" for kind, _ in fake.calls)
