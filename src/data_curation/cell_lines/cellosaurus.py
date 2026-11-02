"""Resolve cell-line names to Cellosaurus accessions, and flag problem lines.

Cell Model Passports is tried first because it is the primary metadata source
and its accessions are the ones the omics modalities are keyed by. Cellosaurus
is consulted only for names CMP does not carry: it resolves 50 of CTRPv2's 52
unmatched lines, worth 5.7% of that dataset's measurements.

Rescued lines are response-only samples - they have no CMP entry, so they carry
no omics - which is correct rather than a defect: a screened cell line belongs in
the response matrix whether or not anyone profiled its transcriptome.
"""

import logging

import pandas as pd
from upath import UPath

from data_curation.cell_lines.sidm_mapping import normalize_cell_line_name
from data_curation.core import layout

logger = logging.getLogger(__name__)

CELLOSAURUS_PATH = layout.CELLOSAURUS

QUALITY_COLUMNS = ["problematic", "problem_description", "discontinued", "misspelling"]


def _index_by_normalized_name(frame: pd.DataFrame) -> dict[str, str]:
    """Map normalized primary names, then synonyms, to accessions.

    Primary names win: a synonym of one line is occasionally another line's
    official name, and the official name is the safer resolution.
    """
    mapping: dict[str, str] = {}
    for name, accession in zip(frame["name"], frame["cellosaurus_id"], strict=True):
        key = normalize_cell_line_name(name)
        if key:
            mapping.setdefault(key, accession)

    for synonyms, accession in zip(frame["synonyms"], frame["cellosaurus_id"], strict=True):
        if not isinstance(synonyms, str):
            continue
        for synonym in synonyms.split(";"):
            key = normalize_cell_line_name(synonym)
            if key:
                mapping.setdefault(key, accession)
    return mapping


def load_cellosaurus(path: UPath = CELLOSAURUS_PATH) -> pd.DataFrame:
    """Load the parsed Cellosaurus table.

    Args:
        path: Location of the parsed Parquet file.

    Returns:
        The table, indexed by accession.

    Raises:
        FileNotFoundError: If the download stage has not run.
    """
    if not path.exists():
        msg = f"{path} is missing; run the download_cellosaurus stage first"
        raise FileNotFoundError(msg)
    frame = pd.read_parquet(str(path))
    frame["problematic"] = frame["problematic"].astype("boolean").fillna(False).astype(bool)
    for column in ("name", "synonyms", "problem_description", "discontinued", "misspelling"):
        frame[column] = frame[column].fillna("").astype(str)
    return frame


def build_fallback_name_map(path: UPath = CELLOSAURUS_PATH) -> dict[str, str]:
    """Build normalized-name -> accession from Cellosaurus.

    Args:
        path: Location of the parsed Parquet file.

    Returns:
        Mapping of normalized name (and synonym) to CVCL accession.
    """
    return _index_by_normalized_name(load_cellosaurus(path))


def resolve_names(
    names: pd.Series,
    primary: dict[str, tuple[str, str]],
    fallback: dict[str, str] | None = None,
) -> tuple[pd.Series, dict[str, str]]:
    """Resolve cell-line names to accessions, falling back when *primary* misses.

    Args:
        names: Native cell-line names, one per measurement.
        primary: Cell Model Passports map of normalized name -> (accession, name).
        fallback: Optional Cellosaurus map of normalized name -> accession.

    Returns:
        The resolved accessions aligned to *names*, and the rescued
        ``{name: accession}`` pairs so the caller can report them.
    """
    keys = names.map(normalize_cell_line_name)
    resolved = keys.map(lambda key: primary[key][0] if key in primary else None)

    if not fallback:
        return resolved, {}

    unresolved = resolved.isna() & names.notna()
    if not unresolved.any():
        return resolved, {}

    rescued: dict[str, str] = {}
    for name, key in zip(names[unresolved], keys[unresolved], strict=True):
        accession = fallback.get(key)
        if accession is not None:
            rescued[str(name)] = accession

    if rescued:
        resolved = resolved.where(~unresolved, keys.map(fallback))
        logger.info(
            "Cellosaurus resolved %d cell lines Cell Model Passports does not carry: %s",
            len(rescued),
            sorted(rescued)[:10],
        )
    return resolved, rescued


def quality_flags(accessions: pd.Index, path: UPath = CELLOSAURUS_PATH) -> pd.DataFrame:
    """Return the Cellosaurus quality flags for the given accessions.

    Args:
        accessions: CVCL accessions to look up.
        path: Location of the parsed Parquet file.

    Returns:
        Frame indexed like *accessions* with the :data:`QUALITY_COLUMNS`.
        Accessions Cellosaurus does not know get ``problematic = False`` and
        empty descriptions.
    """
    frame = load_cellosaurus(path).drop_duplicates(subset="cellosaurus_id").set_index("cellosaurus_id")
    aligned = frame.reindex(pd.Index(accessions))[QUALITY_COLUMNS]
    aligned["problematic"] = aligned["problematic"].astype("boolean").fillna(False).astype(bool)
    for column in QUALITY_COLUMNS[1:]:
        aligned[column] = aligned[column].fillna("").astype(str)
    return aligned
