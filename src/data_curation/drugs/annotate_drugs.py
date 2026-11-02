"""Build the central drug ontology dictionary for all datasets.

Reads every dataset's drug mapping, takes the union of the drugs they screen,
and annotates each distinct drug once with public classification data:

- Probes & Drugs: ChEMBL protein family classification (five levels) and target
- ChEBI: therapeutic roles
- NCI Thesaurus: oncology drug classes

The result is a single parquet keyed by the response ``var_name``, which
``assemble_mudata`` merges onto the response modality of each MuData file.

Drugs whose PubChem lookup failed upstream carry a dataset-prefixed pseudo-ID
instead of a CID. They cannot be annotated, but they are still emitted with
empty values so the merge covers every drug and none silently disappears. Their
``drug_name`` is taken from the source dataset, since PubChem can only name a
real CID.

Combination therapies are among those pseudo-IDs by design (a mixture has no
single CID). They are marked ``drug_role = "combination"`` and carry their
component IDs, so they group together instead of joining the genuinely
unannotated compounds.
"""

import json
import logging

import pandas as pd

from data_curation.cli._progress import track
from data_curation.core import layout
from data_curation.core.config import processing_config
from data_curation.core.errors import CurationError
from data_curation.drugs.mapping.combinations import COMBINATION_COLUMNS, load_combination_annotations
from data_curation.drugs.mapping.names import fill_missing_names, load_source_drug_names
from data_curation.drugs.ontology.chebi import ChebiRoles
from data_curation.drugs.ontology.ncit import NcitClasses
from data_curation.drugs.ontology.pubchem import enforce_failure_budget, fetch_cross_references
from data_curation.drugs.ontology.targets import DrugTargets
from data_curation.response.harmonize.structures import (
    apply_structures,
    canonical_components,
    canonical_id_map,
    read_structures,
)

logger = logging.getLogger(__name__)

RAW_DIR = layout.RAW_DRUG_ONTOLOGY
DRUG_MAPPING_DIR = layout.DRUG_MAPPING_DIR
OUTPUT_PATH = layout.DRUG_ONTOLOGY
FAILURES_PATH = OUTPUT_PATH.with_name("pubchem_failures.tsv")

UNKNOWN = "unknown"
COMBINATION_ROLE = "combination"

# Columns that describe the drug's grouping, in increasing specificity.
GROUPING_COLUMNS = [
    "drug_role",
    "ncit_branch",
    "ncit_class",
    "target_class_l1",
    "target_class_l2",
    "target_class_l3",
    "target_class_l4",
    "target_class_l5",
    "target_gene",
]
_ALL_COLUMNS = [
    "drug_name",
    "inchikey",
    "chebi_id",
    "ncit_code",
    "drug_role",
    "drug_role_all",
    "ncit_class",
    "ncit_branch",
    "target_gene",
    "target_gene_all",
    "target_moa",
    *[f"target_class_l{i}" for i in range(1, 6)],
    "annotation_source",
]

# Fail loudly if the target layer collapses, which would indicate an upstream
# schema or release change rather than genuinely unannotatable compounds.
_MIN_TARGET_COVERAGE = 0.30


def collect_drugs(structures: pd.DataFrame) -> dict[str, list[str]]:
    """Collect the canonical drug IDs screened by each dataset, from the drug mappings.

    Read from ``drug_mapping/`` rather than the fitted response files. The two
    carry an identical drug universe (verified: 1222 IDs either way, zero
    difference on any dataset), but ``drug_mapping`` does not depend on curve
    fitting. Depending on the response files instead made this a global barrier -
    no dataset could be assembled until every CurveCurator run had finished, and
    re-fitting one dataset invalidated the ontology and so every assembly.

    Args:
        structures: The structure table; CIDs that share an InChIKey are one drug, listed
            under the lowest of them.

    Returns:
        Mapping of dataset name to its drug IDs.

    Raises:
        FileNotFoundError: If no drug mapping files exist.
    """
    per_dataset: dict[str, list[str]] = {}
    for path in sorted(DRUG_MAPPING_DIR.glob("*.parquet")):
        mapping = apply_structures(pd.read_parquet(str(path)), structures)
        per_dataset[path.stem] = [str(drug_id) for drug_id in mapping["pubchem_id"].unique()]

    if not per_dataset:
        msg = f"No drug mapping files found in {DRUG_MAPPING_DIR}"
        raise FileNotFoundError(msg)

    logger.info(
        "Found %d datasets: %s",
        len(per_dataset),
        ", ".join(f"{name} ({len(drugs)})" for name, drugs in per_dataset.items()),
    )
    return per_dataset


def _load_cid_to_ncit() -> dict[str, list[str]]:
    """Load the cached PubChem CID -> NCIt code mapping."""
    payload = json.loads((RAW_DIR / "cid2ncit.json").read_text())
    logger.info("CID -> NCIt mapping retrieved %s", payload.get("retrieved", "unknown"))
    return payload["cid_to_ncit"]


def _annotate(drug_ids: list[str], inchikeys: dict[str, str], failures: list[tuple[str, str]]) -> pd.DataFrame:
    """Annotate every drug ID, returning one row per ID.

    Args:
        drug_ids: Distinct response ``var_names`` across all datasets.
        inchikeys: InChIKey per drug ID, from the structure table, so the ontology and
            ``var`` agree with it.
        failures: Collects ``(query, reason)`` for PubChem lookups that failed;
            the affected drugs keep empty (``unknown``) PubChem-derived fields.

    Returns:
        DataFrame indexed by drug ID with the ontology columns.
    """
    cids = sorted({d for d in drug_ids if d.isdigit()}, key=int)
    logger.info("Annotating %d drugs (%d with a PubChem CID)", len(drug_ids), len(cids))

    cross_refs = fetch_cross_references(cids, failures)
    cid_to_ncit = _load_cid_to_ncit()

    chebi = ChebiRoles(RAW_DIR / "chebi_compounds.tsv.gz", RAW_DIR / "chebi_relation.tsv.gz")
    ncit = NcitClasses(RAW_DIR / "Thesaurus.txt")
    targets = DrugTargets(RAW_DIR / "pd_compounds.csv", RAW_DIR / "pd_targets.csv")

    records = []
    for drug_id in track(drug_ids, "Annotating drugs"):
        row: dict[str, str] = {"drug_id": drug_id}
        if not drug_id.isdigit():
            records.append(row)
            continue

        refs = cross_refs.get(drug_id, {})
        inchikey = inchikeys.get(drug_id, "")
        row["drug_name"] = refs.get("name", "")
        row["inchikey"] = inchikey
        row["chebi_id"] = refs.get("chebi_id", "")

        row.update(chebi.annotate(row["chebi_id"]))
        row.update(ncit.annotate(cid_to_ncit.get(drug_id, [])))
        row.update(targets.annotate(inchikey))

        sources = []
        if row.get("target_class_l1") or row.get("target_gene"):
            sources.append("ProbesAndDrugs")
        if row.get("drug_role"):
            sources.append("ChEBI")
        if row.get("ncit_code"):
            sources.append("NCIt")
        row["annotation_source"] = "|".join(sources)
        records.append(row)

    frame = pd.DataFrame(records).set_index("drug_id")
    frame = frame.reindex(columns=_ALL_COLUMNS)
    return frame.fillna("").replace("", UNKNOWN)


def _canonical_combinations(combinations: pd.DataFrame, structures: pd.DataFrame) -> pd.DataFrame:
    """Move the combination table onto canonical IDs, a combination row still winning over a single agent."""
    canonical = canonical_id_map(structures)
    moved = combinations.copy()
    moved.index = pd.Index([canonical.get(drug_id, drug_id) for drug_id in moved.index], name=moved.index.name)
    moved["component_pubchem_ids"] = canonical_components(moved["component_pubchem_ids"], canonical)
    moved = moved.sort_values("is_combination", ascending=False, kind="stable")
    return moved[~moved.index.duplicated(keep="first")]


def _add_combination_columns(frame: pd.DataFrame, structures: pd.DataFrame) -> pd.DataFrame:
    """Attach the combination flags from the drug mapping files.

    Combinations have no CID and so no ontology annotation, but they are a known
    kind of drug rather than an unannotatable one: giving them their own
    ``drug_role`` keeps them out of the ``unknown`` bucket.

    Args:
        frame: Annotated drugs indexed by drug ID.
        structures: The structure table, to put component IDs onto canonical IDs.

    Returns:
        ``frame`` with the combination columns appended and ``drug_role`` set to
        ``"combination"`` for every combination.
    """
    combinations = _canonical_combinations(load_combination_annotations(DRUG_MAPPING_DIR), structures)

    annotated = frame.copy()
    annotated["is_combination"] = False
    annotated["n_components"] = 0
    annotated["component_pubchem_ids"] = ""

    shared = annotated.index.intersection(combinations.index[combinations["is_combination"]])
    annotated.loc[shared, COMBINATION_COLUMNS] = combinations.loc[shared, COMBINATION_COLUMNS]
    annotated.loc[shared, "drug_role"] = COMBINATION_ROLE
    annotated["is_combination"] = annotated["is_combination"].astype(bool)
    annotated["n_components"] = annotated["n_components"].astype(int)

    missing = combinations.index[combinations["is_combination"]].difference(annotated.index)
    if len(missing):
        logger.info("%d mapped combination(s) are not screened in any response file", len(missing))
    logger.info("Marked %d of %d drugs as combinations", len(shared), len(annotated))
    return annotated


def _log_coverage(frame: pd.DataFrame, per_dataset: dict[str, list[str]]) -> None:
    """Log annotation coverage overall and per dataset, and enforce a floor.

    Raises:
        CurationError: If the share of drugs with a target classification is below the floor.
    """
    total = len(frame)
    logger.info("--- coverage over %d distinct drugs ---", total)
    for column in GROUPING_COLUMNS:
        known = int((frame[column] != UNKNOWN).sum())
        logger.info("  %-18s %4d (%.0f%%)", column, known, 100 * known / total)

    logger.info("--- coverage per dataset ---")
    for name, drugs in per_dataset.items():
        subset = frame.reindex(drugs)
        target = int((subset["target_class_l1"] != UNKNOWN).sum())
        role = int((subset["drug_role"] != UNKNOWN).sum())
        logger.info(
            "  %-12s n=%4d target_class %3d (%.0f%%)  drug_role %3d (%.0f%%)",
            name,
            len(drugs),
            target,
            100 * target / len(drugs),
            role,
            100 * role / len(drugs),
        )

    coverage = (frame["target_class_l1"] != UNKNOWN).mean()
    if coverage < _MIN_TARGET_COVERAGE:
        msg = (
            f"Target classification covers only {coverage:.0%} of drugs, below the "
            f"{_MIN_TARGET_COVERAGE:.0%} floor; an upstream source has likely changed"
        )
        raise CurationError(msg)


def main() -> None:
    """Build and write the central drug ontology parquet."""
    structures = read_structures()
    per_dataset = collect_drugs(structures)
    drug_ids = sorted({drug for drugs in per_dataset.values() for drug in drugs})

    failures: list[tuple[str, str]] = []
    inchikeys = {
        canonical: key for canonical, key in zip(structures["canonical_id"], structures["inchikey"], strict=True) if key
    }
    frame = _annotate(drug_ids, inchikeys, failures)
    # One cross-reference lookup per CID; InChIKeys come from the structure table.
    n_lookups = sum(1 for drug_id in drug_ids if drug_id.isdigit())
    enforce_failure_budget(failures, n_lookups, processing_config("pubchem")["max_failure_fraction"], FAILURES_PATH)
    frame["drug_name"] = fill_missing_names(
        frame["drug_name"], load_source_drug_names(DRUG_MAPPING_DIR, structures), UNKNOWN
    )
    frame = _add_combination_columns(frame, structures)
    _log_coverage(frame, per_dataset)

    for column in GROUPING_COLUMNS:
        frame[column] = frame[column].astype("category")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(str(OUTPUT_PATH))
    logger.info("Wrote %d drug annotations to %s", len(frame), OUTPUT_PATH)
