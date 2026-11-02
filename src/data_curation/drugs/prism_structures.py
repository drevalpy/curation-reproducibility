"""Structure-aware PubChem resolution for PRISM drugs.

The PRISM treatment-info ``smiles`` column is a comma-separated list of the structures the Broad registered
for a compound: mostly identical copies, sometimes stereo, E/Z or tautomer variants, sometimes empty entries.
PubChem rejects such a list, and a single structure resolves to a stereo- or salt-specific record that
differs from the name-based CID the other datasets get. The rule implemented here keeps the name CID
whenever a registered structure confirms it and only falls back to the structure when it does not.

Everything is pure except the three injected lookups, so the rule is testable without a network.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache, reduce

from rdkit import Chem, rdBase
from rdkit.Chem.MolStandardize import rdMolStandardize

from data_curation.drugs.mapping.combinations import is_combination

logger = logging.getLogger(__name__)

_CONNECTIVITY_LENGTH = 14  # first block of an InChIKey: skeleton only, no stereo or isotopes
_MAX_TAUTOMERS = 200
# Metals that are part of the drug (Pt, V, Zn, Hg, Sb, ...), as opposed to Na/K/Ca-type counter-ions. A
# metal complex split into fragments (cisplatin as ``N.N.Cl[Pt]Cl``) must keep its metal: the largest organic
# fragment would be ammonia.
_COMPLEX_METALS = frozenset({13, *range(21, 32), *range(39, 52), *range(57, 85)})
_COUNTER_IONS = frozenset({3, 11, 12, 19, 20, 37, 38, 55, 56, 8})  # alkali/alkaline-earth metals and water
_SPECIFIED_BOND_STEREO = (
    Chem.BondStereo.STEREOE,
    Chem.BondStereo.STEREOZ,
    Chem.BondStereo.STEREOCIS,
    Chem.BondStereo.STEREOTRANS,
)

#: Resolution methods, in decreasing order of confidence.
NAME_VERIFIED = "name_verified"
NAME_VERIFIED_TAUTOMER = "name_verified_tautomer"
STRUCTURE = "structure"
NAME_ONLY = "name_only"
NAME_UNVERIFIED = "name_unverified"
COMBINATION = "combination"
UNRESOLVED = "unresolved"


@dataclass(frozen=True)
class Resolution:
    """Outcome of resolving one PRISM drug.

    Attributes:
        cid: PubChem CID, or None when the drug keeps its pseudo-ID.
        method: One of the module-level method constants.
        note: Human-readable detail, e.g. what the name lookup returned when it disagreed with the structure.
    """

    cid: int | None
    method: str
    note: str = ""


@dataclass(frozen=True)
class Structure:
    """A registered structure reduced to its standardized parent.

    Attributes:
        smiles: Canonical SMILES of the structure as registered.
        connectivity: InChIKey first block of the parent.
        inchikey: Full InChIKey of the parent, stereo included.
        n_stereo: Number of specified stereo centres and double bonds.
        parent_smiles: Canonical SMILES of the standardized parent.
        has_metal: Whether the parent is a metal complex, whose InChI differs between bonded and ionic drawings.
    """

    smiles: str
    connectivity: str
    inchikey: str
    n_stereo: int
    parent_smiles: str
    has_metal: bool


def _parse(smiles: str) -> Chem.Mol | None:
    with rdBase.BlockLogs():
        return Chem.MolFromSmiles(smiles)


def parse_smiles_field(raw: object) -> list[str]:
    """Split a PRISM ``smiles`` field into distinct valid structures.

    Args:
        raw: The raw cell: comma-separated SMILES, possibly with empty entries, or NaN/None.

    Returns:
        Canonical SMILES of the entries RDKit can parse, deduplicated and in input order.
    """
    if not isinstance(raw, str):
        return []
    seen: dict[str, None] = {}
    for part in raw.split(","):
        mol = _parse(part.strip()) if part.strip() else None
        if mol is not None:
            seen.setdefault(Chem.MolToSmiles(mol), None)
    return list(seen)


def _has_complex_metal(mol: Chem.Mol) -> bool:
    return any(atom.GetAtomicNum() in _COMPLEX_METALS for atom in mol.GetAtoms())


def _is_counter_ion(fragment: Chem.Mol) -> bool:
    atoms = list(fragment.GetAtoms())
    return len(atoms) == 1 and atoms[0].GetAtomicNum() in _COUNTER_IONS


def standardize_parent(mol: Chem.Mol) -> Chem.Mol:
    """Reduce a molecule to its neutral parent: largest organic fragment, charges neutralized.

    Metal complexes (cisplatin, vanadium chelates) keep all fragments except alkali/alkaline-earth ions and
    water, because the metal is what makes the compound.

    Args:
        mol: Any molecule, possibly a salt or solvate.

    Returns:
        The standardized parent. Counter-ions, water and solvent molecules are dropped.
    """
    with rdBase.BlockLogs():
        if _has_complex_metal(mol):
            kept = [f for f in Chem.GetMolFrags(mol, asMols=True) if not _is_counter_ion(f)]
            parent = reduce(Chem.CombineMols, kept) if kept else mol
        else:
            parent = rdMolStandardize.LargestFragmentChooser(preferOrganic=True).choose(mol)
        return rdMolStandardize.Uncharger().uncharge(parent)


def _inchikey(mol: Chem.Mol) -> str:
    with rdBase.BlockLogs():
        return Chem.MolToInchiKey(mol) or ""


def connectivity_key(mol: Chem.Mol) -> str:
    """Return the stereo-free identity of a molecule's standardized parent.

    Args:
        mol: Any molecule.

    Returns:
        First 14 characters of the parent's InChIKey, or "" when InChI generation fails.
    """
    return _inchikey(standardize_parent(mol))[:_CONNECTIVITY_LENGTH]


def connectivity_key_from_smiles(smiles: str | None) -> str | None:
    """Connectivity key of a SMILES string, e.g. PubChem's connectivity SMILES for a CID.

    Args:
        smiles: A SMILES string, or None.

    Returns:
        The key, or None when the SMILES is missing, invalid or has no InChI.
    """
    mol = _parse(smiles) if smiles else None
    return (connectivity_key(mol) or None) if mol is not None else None


@lru_cache(maxsize=4096)
def tautomer_key(smiles: str | None) -> str | None:
    """Tautomer-insensitive identity of a SMILES string's standardized parent.

    InChI normalizes mobile hydrogens but not ring-chain or keto-enol forms, so registered tautomers of
    one compound (tetracyclines, tagatose) get different connectivity keys. This canonicalizes the tautomer
    first and ignores stereo.

    Args:
        smiles: A SMILES string, or None.

    Returns:
        First InChIKey block of the canonical tautomer of the parent, or None when it cannot be computed.
    """
    mol = _parse(smiles) if smiles else None
    if mol is None:
        return None
    with rdBase.BlockLogs():
        enumerator = rdMolStandardize.TautomerEnumerator()
        enumerator.SetMaxTautomers(_MAX_TAUTOMERS)
        canonical = enumerator.Canonicalize(standardize_parent(mol))
    return _inchikey(canonical)[:_CONNECTIVITY_LENGTH] or None


def count_specified_stereo(mol: Chem.Mol) -> int:
    """Count specified stereo centres and double-bond configurations."""
    centres = Chem.FindMolChiralCenters(mol, includeUnassigned=False, useLegacyImplementation=False)
    bonds = [bond for bond in mol.GetBonds() if bond.GetStereo() in _SPECIFIED_BOND_STEREO]
    return len(centres) + len(bonds)


def _structure(smiles: str) -> Structure | None:
    mol = _parse(smiles)
    if mol is None:
        return None
    parent = standardize_parent(mol)
    key = _inchikey(parent)
    if not key:
        return None
    return Structure(
        smiles,
        key[:_CONNECTIVITY_LENGTH],
        key,
        count_specified_stereo(parent),
        Chem.MolToSmiles(parent),
        _has_complex_metal(parent),
    )


def structures_from_field(smiles_field: object) -> list[Structure]:
    """Parse a PRISM ``smiles`` field into standardized structures, most stereo-specific first.

    Args:
        smiles_field: The raw cell.

    Returns:
        One entry per distinct parent InChIKey; ties keep the field order.
    """
    found = (_structure(smiles) for smiles in parse_smiles_field(smiles_field))
    distinct = {s.inchikey: s for s in found if s is not None}
    return sorted(distinct.values(), key=lambda s: -s.n_stereo)


def _by_structure(
    structures: list[Structure],
    lookup_inchikey: Callable[[str], int | None],
) -> int | None:
    for structure in structures:
        cid = lookup_inchikey(structure.inchikey)
        if cid is not None:
            return cid
    return None


def _verified_by(
    name_cid: int,
    structures: list[Structure],
    cid_connectivity: Callable[[int], str | None],
    cid_smiles: Callable[[int], str | None] | None,
) -> str | None:
    """Return the verification method confirming ``name_cid`` against the structures, if any."""
    if cid_connectivity(name_cid) in {s.connectivity for s in structures}:
        return NAME_VERIFIED
    record_smiles = cid_smiles(name_cid) if cid_smiles else None
    record_key = tautomer_key(record_smiles)
    if record_key is not None and any(record_key == tautomer_key(s.parent_smiles) for s in structures):
        return NAME_VERIFIED_TAUTOMER
    return None


def resolve_prism_drug(
    name: str,
    smiles_field: object,
    *,
    lookup_name: Callable[[str], int | None],
    lookup_inchikey: Callable[[str], int | None],
    cid_connectivity: Callable[[int], str | None],
    cid_smiles: Callable[[int], str | None] | None = None,
) -> Resolution:
    """Resolve one PRISM drug to a PubChem CID.

    The name CID wins when it agrees with a registered structure. Otherwise the most stereo-specific structure
    is looked up by InChIKey, then the remaining ones. A name CID that no structure could replace is kept but
    flagged, as the current pipeline does for drugs without a structure hit. Metal complexes are never
    overridden by structure: PubChem draws them ionic and the registry covalent, so the comparison is unreliable.

    Args:
        name: Drug name from the treatment info.
        smiles_field: Raw comma-separated ``smiles`` cell.
        lookup_name: Name -> CID or None.
        lookup_inchikey: Full InChIKey -> CID or None.
        cid_connectivity: CID -> connectivity key (see ``connectivity_key_from_smiles``) of PubChem's record,
            or None when unknown.
        cid_smiles: CID -> SMILES of PubChem's record. When given, a name CID whose connectivity differs is
            still accepted if it is the same compound as a structure after tautomer canonicalization.

    Returns:
        The resolution; ``cid`` is None when the drug keeps its pseudo-ID.
    """
    if is_combination(name):
        return Resolution(None, COMBINATION)

    structures = structures_from_field(smiles_field)
    name_cid = lookup_name(name)
    if not structures:
        return Resolution(name_cid, NAME_ONLY if name_cid else UNRESOLVED, "no valid structure")

    verified = _verified_by(name_cid, structures, cid_connectivity, cid_smiles) if name_cid is not None else None
    if verified:
        return Resolution(name_cid, verified)

    if name_cid is not None and any(s.has_metal for s in structures):
        return Resolution(name_cid, NAME_UNVERIFIED, "metal complex: structure comparison unreliable")

    detail = f"name CID {name_cid} does not match structure" if name_cid else "name not found"
    cid = _by_structure(structures, lookup_inchikey)
    if cid is not None:
        return Resolution(cid, STRUCTURE, detail)
    if name_cid is not None:
        return Resolution(name_cid, NAME_UNVERIFIED, f"{detail}; structure not in PubChem")
    return Resolution(None, UNRESOLVED, f"{detail}; structure not in PubChem")
