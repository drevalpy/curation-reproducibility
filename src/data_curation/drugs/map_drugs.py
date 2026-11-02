"""Map drug identifiers to PubChem CIDs via the PubChem REST API.

Supports GDSC (name-based lookup), CTRP (bundled CIDs for v1, SMILES-based
lookup for v2), and PDX_Bruna / BeatAML2 (name-based lookup with curated
overrides). Drugs that cannot be resolved get a dataset-prefixed pseudo-ID so
no measurements are silently dropped. PRISM is mapped by ``map_prism``, which
builds on this module's lookups.

Combination therapies are recognised from their name and deliberately kept as
pseudo-IDs, with their components recorded separately; see
``_drug_combinations``.
"""

import logging
import time
import urllib.error
from collections.abc import Callable

import pandas as pd
import pubchempy as pcp
from upath import UPath

from data_curation.cli._progress import track
from data_curation.core import layout
from data_curation.core.config import dataset_config, processing_config
from data_curation.core.errors import CurationError
from data_curation.core.paths import find_file
from data_curation.drugs.mapping.combinations import annotate_combinations, is_combination
from data_curation.drugs.mapping.overrides import (
    BEATAML2_CID_OVERRIDES,
    CCLE_CID_OVERRIDES,
    PDX_BRUNA_CID_OVERRIDES,
)
from data_curation.drugs.ontology.pubchem import enforce_failure_budget
from data_curation.response.gdsc_tags import library_drug_ids

logger = logging.getLogger(__name__)

_REQUEST_DELAY = 0.25  # seconds between PubChem requests (stay under 5 req/s)
_MAX_ATTEMPTS = 4
_RETRY_BACKOFF = 1.5  # seconds, multiplied by the attempt number

# No dataset screens the same compound more than a handful of times, so a larger
# group sharing one CID means a lookup degenerated (as "nan" once did, absorbing
# 47 GDSC1 drugs into N-acetyl-beta-neuraminic acid) rather than a real replicate.
_MAX_DRUGS_PER_CID = 4

# Columns of the GDSC raw file the mapper needs. Passed to read_csv as a predicate: ty cannot match a list of
# names to pandas' ``SequenceNotStr`` overload of ``usecols``, whereas a callable type-checks.
_GDSC_RAW_COLUMNS = frozenset({"TAG", "DRUG_ID"})

# What a failed lookup can look like. A genuine "no match" is not among these:
# pubchempy turns a 404 into an empty result. A 400 (BadRequestError) is PubChem
# rejecting the identifier itself, e.g. a PRISM SMILES field listing several
# structures; the answer is as final as a 404, so it is "no match", not a failure.
_LOOKUP_ERRORS = (pcp.PubChemPyError, urllib.error.URLError, OSError)
_NOT_RETRYABLE = (pcp.MethodNotAllowedError, pcp.UnimplementedError)


class _LookupLog:
    """Counts PubChem lookups and the ones that failed, for the failure budget."""

    def __init__(self) -> None:
        self.total = 0
        self.failures: list[tuple[str, str]] = []

    def reset(self) -> None:
        self.total = 0
        self.failures = []


_LOOKUPS = _LookupLog()


def _lookup_pubchem(identifier: str, namespace: str = "name") -> tuple[int | None, str | None]:
    """Query PubChem for a compound, returning (CID, SMILES) or (None, None).

    Transient failures (rate limiting, server busy, network errors) are retried
    with a linear backoff. Without this, a flaky request silently degrades a
    resolvable drug into a pseudo-ID that then gets baked into the output.
    A genuine "no match" comes back as an empty result, not an exception, so it
    is not retried. A lookup that still fails is recorded in ``_LOOKUPS`` and
    answered with (None, None) so the drug keeps its pseudo-ID; ``main`` then
    checks the failure budget.
    """
    _LOOKUPS.total += 1
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        try:
            compounds = pcp.get_compounds(identifier, namespace)
        except pcp.NotFoundError:
            return None, None
        except pcp.BadRequestError as exc:
            logger.info("PubChem rejected %s '%s': %s", namespace, identifier, exc)
            return None, None
        except _LOOKUP_ERRORS as exc:
            if attempt == _MAX_ATTEMPTS or isinstance(exc, _NOT_RETRYABLE):
                logger.warning(
                    "PubChem %s lookup for '%s' failed after %d attempt(s): %s",
                    namespace,
                    identifier,
                    attempt,
                    exc,
                )
                _LOOKUPS.failures.append((f"{namespace}:{identifier}", f"{type(exc).__name__}: {exc}"))
                return None, None
            logger.debug(
                "PubChem %s lookup for '%s' failed (attempt %d/%d), retrying: %s",
                namespace,
                identifier,
                attempt,
                _MAX_ATTEMPTS,
                exc,
            )
            time.sleep(_RETRY_BACKOFF * attempt)
            continue

        if compounds and compounds[0].cid:
            return compounds[0].cid, compounds[0].canonical_smiles
        return None, None

    return None, None


def _resolve_by_name(drug_name: str, synonyms: str) -> tuple[int | None, str | None]:
    """Try to resolve a drug via name then synonyms.

    Combination names are refused: PubChem matches some of them to an unrelated
    single compound, which would conflate a mixture with a real drug.
    """
    if is_combination(drug_name):
        logger.debug("'%s' is a combination, keeping a pseudo-ID", drug_name)
        return None, None

    time.sleep(_REQUEST_DELAY)
    cid, smiles = _lookup_pubchem(drug_name)
    if cid is not None:
        return cid, smiles

    for synonym in synonyms.replace(";", ",").split(","):
        synonym = synonym.strip()
        if not synonym:
            continue
        time.sleep(_REQUEST_DELAY)
        cid, smiles = _lookup_pubchem(synonym)
        if cid is not None:
            return cid, smiles

    return None, None


def _lookup_component(component_name: str) -> int | None:
    """Resolve a combination component name to a CID, for the component fallback."""
    time.sleep(_REQUEST_DELAY)
    cid, _ = _lookup_pubchem(component_name)
    return cid


def _record(drug_id: object, drug_name: str, cid: int | None, smiles: str | None, prefix: str) -> dict:
    """Build one output row, falling back to a pseudo-ID when unresolved.

    A successful mapping logs at debug: with a progress bar reporting the loop's
    position, one line per resolved drug would bury the warnings for the ones
    that failed, which are the only entries an operator needs to look at.
    """
    pubchem_id = str(cid) if cid else f"{prefix}_{drug_id}"
    if cid:
        logger.debug("Mapped '%s' -> CID %s", drug_name, cid)
    else:
        logger.warning("No PubChem match for '%s' (id=%s), using %s", drug_name, drug_id, pubchem_id)
    return {
        "DRUG_ID": drug_id,
        "DRUG_NAME": drug_name,
        "pubchem_id": pubchem_id,
        "canonical_smiles": smiles or "",
    }


def _map_gdsc(raw_dir: UPath) -> pd.DataFrame:
    """Map GDSC drugs by name, restricted to drugs screened in the library arm.

    Restricting to the library arm rather than to any well with a DRUG_ID keeps
    plate controls out; see ``_gdsc_tags``.
    """
    compounds = pd.read_csv(str(raw_dir / "compounds.csv"))

    raw_files = sorted(raw_dir.glob("*public_raw_data*"))
    if raw_files:
        raw_df = pd.read_csv(str(raw_files[0]), usecols=lambda column: column in _GDSC_RAW_COLUMNS)
        compounds = compounds[compounds["DRUG_ID"].isin(library_drug_ids(raw_df))]

    results = []
    for _, row in track(list(compounds.iterrows()), "Mapping GDSC drugs", total=len(compounds)):
        drug_name = str(row["DRUG_NAME"]).strip()
        # 203 of 621 GDSC compounds have no synonyms; str(NaN) would send the
        # literal "nan" to PubChem, which resolves it to an unrelated CID.
        raw_synonyms = row.get("SYNONYMS")
        synonyms = str(raw_synonyms).strip() if pd.notna(raw_synonyms) else ""
        cid, smiles = _resolve_by_name(drug_name, synonyms)
        results.append(_record(row["DRUG_ID"], drug_name, cid, smiles, "GDSC"))

    return pd.DataFrame(results)


def _map_ctrpv1(raw_dir: UPath) -> pd.DataFrame:
    """Map CTRPv1 drugs using the bundled pubchem_cid, falling back to name lookup."""
    informer = pd.read_csv(str(find_file(raw_dir, "*informer_set*")), sep="\t")

    results = []
    for _, row in track(list(informer.iterrows()), "Mapping CTRPv1 drugs", total=len(informer)):
        drug_name = str(row["cpd_name"]).strip()
        raw_cid = row.get("pubchem_cid")
        smiles = row.get("cpd_smiles")
        smiles = str(smiles) if pd.notna(smiles) else ""

        cid: int | None = None
        if pd.notna(raw_cid) and int(raw_cid) > 0:
            cid = int(raw_cid)
            logger.debug("Mapped '%s' -> CID %s (bundled)", drug_name, cid)
        else:
            synonyms = str(row.get("cpd_synonym", "")) if pd.notna(row.get("cpd_synonym")) else ""
            cid, looked_up_smiles = _resolve_by_name(drug_name, synonyms)
            smiles = smiles or (looked_up_smiles or "")

        results.append(_record(drug_name, drug_name, cid, smiles, "CTRP"))

    return pd.DataFrame(results)


def _map_ctrpv2(raw_dir: UPath) -> pd.DataFrame:
    """Map CTRPv2 drugs via SMILES lookup, falling back to name lookup."""
    compounds = pd.read_csv(str(find_file(raw_dir, "*meta.per_compound*")), sep="\t")

    results = []
    for _, row in track(list(compounds.iterrows()), "Mapping CTRPv2 drugs", total=len(compounds)):
        master_cpd_id = row["master_cpd_id"]
        drug_name = str(row["cpd_name"]).strip()
        smiles = str(row["cpd_smiles"]).strip() if pd.notna(row.get("cpd_smiles")) else ""

        cid: int | None = None
        # Combinations carry a "." mixture SMILES that PubChem sometimes matches to
        # a single compound, so skip the structure lookup for them entirely.
        if smiles and not is_combination(drug_name):
            time.sleep(_REQUEST_DELAY)
            cid, _ = _lookup_pubchem(smiles, namespace="smiles")

        if cid is None:
            cid, _ = _resolve_by_name(drug_name, "")

        results.append(_record(master_cpd_id, drug_name, cid, smiles, "CTRP"))

    return pd.DataFrame(results)


def _map_by_names(
    drug_names: list[str],
    prefix: str,
    overrides: dict[str, int],
) -> pd.DataFrame:
    """Map a list of drug names to PubChem CIDs, honouring curated overrides.

    Args:
        drug_names: Unique drug names as they appear in the raw response data.
        prefix: Prefix for pseudo-IDs of unresolvable drugs.
        overrides: Curated name -> CID map applied when PubChem finds nothing.

    Returns:
        DataFrame with DRUG_ID, DRUG_NAME, pubchem_id and canonical_smiles columns.
        DRUG_ID is the raw drug name, which is what the response tables join on.
    """
    results = []
    for drug_name in track(drug_names, f"Mapping {prefix} drugs"):
        cid, smiles = _resolve_by_name(drug_name, "")
        if cid is None and drug_name in overrides:
            cid = overrides[drug_name]
            _, smiles = _lookup_pubchem(str(cid), namespace="cid")
            logger.debug("Mapped '%s' -> CID %s (curated override)", drug_name, cid)
        results.append(_record(drug_name, drug_name, cid, smiles, prefix))

    return pd.DataFrame(results)


def _map_ccle(raw_dir: UPath) -> pd.DataFrame:
    """Map the 24 CCLE compounds by name, honouring the curated overrides.

    Barretina et al. name compounds by development code (``AZD6244``, ``TAE684``)
    as often as by generic name, so most resolve only through the override table.
    """
    from data_curation.response.prepare.ccle import DRUG_COLUMN, SUPPLEMENT_PATTERN, read_activity_table

    raw = read_activity_table(find_file(raw_dir, SUPPLEMENT_PATTERN))
    drug_names = sorted(raw[DRUG_COLUMN].dropna().astype(str).str.strip().unique())
    return _map_by_names(drug_names, "CCLE", CCLE_CID_OVERRIDES)


def _map_bruna(raw_dir: UPath) -> pd.DataFrame:
    """Map PDX_Bruna drugs by name from the raw single-agent screen."""
    raw = pd.read_csv(str(find_file(raw_dir, "RawDataDrugsSingleAgents*")), sep="\t")
    drug_names = sorted(raw["DRUG_ID"].dropna().astype(str).unique())
    return _map_by_names(drug_names, "PDX_Bruna", PDX_BRUNA_CID_OVERRIDES)


def _map_beataml(raw_dir: UPath) -> pd.DataFrame:
    """Map BeatAML2 inhibitors by name from the raw inhibitor screen."""
    raw = pd.read_csv(str(find_file(raw_dir, "*raw_inhibitor*")), sep="\t", low_memory=False)
    drug_names = sorted(raw["inhibitor"].dropna().astype(str).unique())
    return _map_by_names(drug_names, "BeatAML2", BEATAML2_CID_OVERRIDES)


MAPPERS = {
    "gdsc": _map_gdsc,
    "ctrpv1": _map_ctrpv1,
    "ctrpv2": _map_ctrpv2,
    "ccle": _map_ccle,
    "bruna": _map_bruna,
    "beataml": _map_beataml,
}


def _check_cid_collisions(frame: pd.DataFrame, dataset: str) -> None:
    """Fail if one CID absorbs an implausible number of distinct drugs.

    Re-screened compounds legitimately share a CID, so only distinct names are
    counted. A large group means a lookup is matching unrelated drugs onto one
    compound, which would silently merge their response columns downstream.

    Raises:
        CurationError: If any CID is shared by more than ``_MAX_DRUGS_PER_CID``
            distinct drug names.
    """
    resolved = frame[frame["pubchem_id"].astype(str).str.isdigit()]
    names = resolved["DRUG_NAME"].astype(str).str.strip().str.lower()
    grouped = names.groupby(resolved["pubchem_id"])
    offenders = grouped.nunique()[lambda counts: counts > _MAX_DRUGS_PER_CID]
    if offenders.empty:
        return

    detail = "; ".join(f"CID {cid} <- {sorted(set(grouped.get_group(cid)))}" for cid in offenders.index)
    msg = (
        f"{dataset}: {len(offenders)} PubChem CID(s) absorb more than "
        f"{_MAX_DRUGS_PER_CID} distinct drugs, so a lookup is matching unrelated "
        f"compounds onto one CID: {detail}"
    )
    raise CurationError(msg)


def main(dataset: str) -> None:
    """Map drugs to PubChem CIDs for the given dataset.

    Raises:
        CurationError: If the dataset's response format has no mapper, a CID absorbs
            unrelated drugs, or too many PubChem lookups fail.
    """
    fmt = dataset_config(dataset, "response")["format"]
    if fmt == "prism":
        msg = f"{dataset} is mapped by 'data-curation drugs map-prism', not 'drugs map'"
        raise CurationError(msg)
    if fmt not in MAPPERS:
        msg = f"Unknown response format '{fmt}' for dataset {dataset}"
        raise CurationError(msg)
    write_mapping(dataset, MAPPERS[fmt])


def write_mapping(dataset: str, mapper: Callable[[UPath], pd.DataFrame]) -> None:
    """Run a mapper over a dataset's raw directory and write the checked mapping table.

    Args:
        dataset: Dataset name.
        mapper: Builds the DRUG_ID / DRUG_NAME / pubchem_id / canonical_smiles frame from the raw directory.

    Raises:
        CurationError: If a CID absorbs unrelated drugs or too many PubChem lookups fail.
    """
    raw_dir = layout.raw_dir("response", dataset)
    output_path = layout.drug_mapping(dataset)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    _LOOKUPS.reset()
    mapping_df = mapper(raw_dir)
    mapping_df = annotate_combinations(mapping_df, _lookup_component)
    enforce_failure_budget(
        _LOOKUPS.failures,
        _LOOKUPS.total,
        processing_config("pubchem")["max_failure_fraction"],
        output_path.with_name(f"{dataset}_pubchem_failures.tsv"),
    )
    _check_cid_collisions(mapping_df, dataset)

    # Pin an explicit dtype contract before writing so identifiers are stored
    # as strings and never round-trip through Parquet as floats.
    for col in ("DRUG_ID", "pubchem_id", "DRUG_NAME", "component_pubchem_ids"):
        if col in mapping_df.columns:
            mapping_df[col] = mapping_df[col].astype(str)
    if "is_combination" in mapping_df.columns:
        mapping_df["is_combination"] = mapping_df["is_combination"].astype(bool)
    if "n_components" in mapping_df.columns:
        mapping_df["n_components"] = mapping_df["n_components"].astype(int)

    mapping_df.to_parquet(str(output_path), index=False)
    logger.info(
        "Wrote %d drugs (%d unique IDs) to %s",
        len(mapping_df),
        mapping_df["pubchem_id"].nunique(),
        output_path,
    )
