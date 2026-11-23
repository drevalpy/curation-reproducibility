"""Map the cell-line identifiers of the GDSC and CCLE omics files to Cellosaurus accessions.

The dataset-specific omics files key their samples by the producer's own
identifier: GDSC by COSMIC ID, CCLE by CCLE name (``A549_LUNG``). The Cell Model
Passports model list carries both as columns (``COSMIC_ID``, ``CCLE_ID``), so they
resolve through it like the SIDM and DepMap IDs do. A CCLE name the model list
lacks falls back to its stripped name (the part before the tissue suffix), which
resolves through the model names, synonyms and Cellosaurus like the response names.
"""

import logging
from collections.abc import Iterable

import pandas as pd
from upath import UPath

from data_curation.cell_lines.cellosaurus import CELLOSAURUS_PATH, build_fallback_name_map, resolve_names
from data_curation.cell_lines.sidm_mapping import _valid_cvcl, build_name_to_cellosaurus_map

logger = logging.getLogger(__name__)

#: Separator of the model list's multi-valued ID columns (``"1331031;1331030"``).
ID_SEPARATOR = ";"


def _clean_id(value: object) -> str:
    """Return one identifier as a string, dropping the ``.0`` pandas adds to integer columns with gaps."""
    text = str(value).strip()
    return text.removesuffix(".0") if text.endswith(".0") and text[:-2].isdigit() else text


def build_id_to_cellosaurus_map(model_list_path: UPath, column: str) -> dict[str, tuple[str, str]]:
    """Build native ID -> (Cellosaurus accession, model name) from one model list column.

    Every value of a multi-valued cell (``"1331031;1331030"``) maps to the model.
    When two models claim the same ID, the first one in the file wins.

    Args:
        model_list_path: Cell Model Passports ``model_list.csv``.
        column: ID column, e.g. ``"COSMIC_ID"`` or ``"CCLE_ID"``.

    Returns:
        The mapping, for models with a usable accession only.
    """
    models = pd.read_csv(str(model_list_path), low_memory=False)
    mapping: dict[str, tuple[str, str]] = {}
    for _, row in models.iterrows():
        value = row.get(column)
        cvcl = _valid_cvcl(row)
        if pd.isna(value) or cvcl is None:
            continue
        name = str(row["model_name"]) if pd.notna(row.get("model_name")) else ""
        for native in str(value).split(ID_SEPARATOR):
            key = _clean_id(native)
            if key and key not in mapping:
                mapping[key] = (cvcl, name)
    return mapping


def resolve_cosmic_ids(cosmic_ids: Iterable[object], model_list_path: UPath) -> dict[str, tuple[str, str]]:
    """Resolve COSMIC IDs through the model list's ``COSMIC_ID`` column.

    Args:
        cosmic_ids: COSMIC identifiers, as strings or integers.
        model_list_path: Cell Model Passports ``model_list.csv``.

    Returns:
        ``{cosmic_id: (accession, name)}`` for the resolved IDs, keyed by the cleaned string form.
    """
    mapping = build_id_to_cellosaurus_map(model_list_path, "COSMIC_ID")
    keys = [_clean_id(value) for value in cosmic_ids]
    resolved = {key: mapping[key] for key in keys if key in mapping}
    _log_unresolved("COSMIC IDs", keys, resolved)
    return resolved


def strip_ccle_name(ccle_name: str) -> str:
    """Return the cell-line part of a CCLE name, e.g. ``"NCIH854"`` for ``"NCIH854_LUNG"``.

    CCLE names join a stripped cell-line name, which holds no underscore, to an
    upper-case tissue that may (``HAEMATOPOIETIC_AND_LYMPHOID_TISSUE``).
    """
    return ccle_name.split("_", 1)[0]


def resolve_ccle_names(
    ccle_names: Iterable[str],
    model_list_path: UPath,
    cellosaurus_path: UPath = CELLOSAURUS_PATH,
) -> dict[str, tuple[str, str]]:
    """Resolve CCLE names via the model list's ``CCLE_ID``, then by their stripped name.

    Args:
        ccle_names: CCLE names such as ``"A549_LUNG"``.
        model_list_path: Cell Model Passports ``model_list.csv``.
        cellosaurus_path: Parsed Cellosaurus table for the name fallback.

    Returns:
        ``{ccle_name: (accession, name)}`` for the resolved names. Names rescued by
        Cellosaurus carry their stripped CCLE name as the cell-line name.
    """
    names = [str(name) for name in ccle_names]
    by_id = build_id_to_cellosaurus_map(model_list_path, "CCLE_ID")
    resolved = {name: by_id[name] for name in names if name in by_id}

    missing = [name for name in names if name not in resolved]
    if missing:
        name_map = build_name_to_cellosaurus_map(model_list_path)
        stripped = pd.Series([strip_ccle_name(name) for name in missing], index=missing)
        accessions, rescued = resolve_names(stripped, name_map, build_fallback_name_map(cellosaurus_path))
        cmp_names = dict(name_map.values())
        for ccle_name, accession in accessions.dropna().items():
            resolved[str(ccle_name)] = (str(accession), cmp_names.get(str(accession), stripped[ccle_name]))
        if rescued:
            logger.info("Resolved %d CCLE names by their stripped name", len(accessions.dropna()))
    _log_unresolved("CCLE names", names, resolved)
    return resolved


def _log_unresolved(kind: str, keys: list[str], resolved: dict[str, tuple[str, str]]) -> None:
    """Warn about the identifiers that resolved to no accession."""
    unresolved = sorted({key for key in keys if key not in resolved})
    if unresolved:
        logger.warning(
            "%d of %d %s have no Cellosaurus accession: %s", len(unresolved), len(keys), kind, unresolved[:10]
        )
