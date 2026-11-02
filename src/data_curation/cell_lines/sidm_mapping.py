"""Shared cell-line identifier mapping utilities for process stages."""

import logging
import re

import pandas as pd
from upath import UPath

logger = logging.getLogger(__name__)

# A single RRID field must hold one accession. Anything joined by these is
# ambiguous and cannot be used as a cell-line identifier.
_ACCESSION_SEPARATORS = (";", ",", "|", " ")


def build_sidm_to_cellosaurus_map(model_list_path: UPath) -> dict[str, tuple[str, str]]:
    """Build SIDM ID -> (cellosaurus_id, model_name) from the model list.

    Returns:
        Dict mapping SIDM model ID to (CVCL accession, human-readable name).
    """
    models = pd.read_csv(str(model_list_path))
    mapping: dict[str, tuple[str, str]] = {}
    for _, row in models.iterrows():
        sidm = row["model_id"]
        cvcl = _valid_cvcl(row)
        name = row["model_name"] if pd.notna(row.get("model_name")) else ""
        if pd.notna(sidm) and cvcl is not None:
            mapping[str(sidm)] = (cvcl, str(name))

    return mapping


def normalize_cell_line_name(name: object) -> str:
    """Normalize a cell line name for fuzzy matching.

    Strips all non-alphanumeric characters and uppercases, so that
    e.g. "MDA-MB-231", "MDAMB231" and "mda_mb_231" all collapse to
    the same key. Accepts any object because some sources carry purely numeric
    cell-line names that pandas reads as integers.
    """
    return re.sub(r"[^A-Za-z0-9]", "", str(name)).upper()


def _valid_cvcl(row: pd.Series) -> str | None:
    """Return the row's Cellosaurus accession if it is a usable CVCL ID.

    Rejects multi-accession values. ``model_list.csv`` occasionally records two
    accessions in one field (``SIDM00400``/SC-1 carries
    ``"CVCL_2717;CVCL_1888"``), and a bare ``startswith`` check let that literal
    string through as if it were an identifier.
    """
    cvcl = row.get("RRID")
    if pd.isna(cvcl):
        return None
    accession = str(cvcl).strip()
    if not accession.startswith("CVCL"):
        return None
    if any(sep in accession for sep in _ACCESSION_SEPARATORS):
        logger.warning("Ignoring ambiguous multi-accession RRID %r for model %r", accession, row.get("model_id"))
        return None
    return accession


def _add_primary_names(models: pd.DataFrame, mapping: dict[str, tuple[str, str]]) -> None:
    """Populate the mapping from primary model names."""
    for _, row in models.iterrows():
        cvcl = _valid_cvcl(row)
        name = row.get("model_name")
        if cvcl is None or pd.isna(name):
            continue
        key = normalize_cell_line_name(name)
        if key and key not in mapping:
            mapping[key] = (cvcl, str(name))


def _add_synonyms(models: pd.DataFrame, mapping: dict[str, tuple[str, str]]) -> None:
    """Fill remaining gaps from synonyms without overriding primary names."""
    for _, row in models.iterrows():
        cvcl = _valid_cvcl(row)
        synonyms = row.get("synonyms")
        if cvcl is None or pd.isna(synonyms):
            continue
        name = row.get("model_name")
        canonical = str(name) if pd.notna(name) else ""
        for synonym in str(synonyms).split(";"):
            key = normalize_cell_line_name(synonym)
            if key and key not in mapping:
                mapping[key] = (cvcl, canonical)


def build_name_to_cellosaurus_map(model_list_path: UPath) -> dict[str, tuple[str, str]]:
    """Build normalized cell_line_name -> (cellosaurus_id, model_name) from the model list.

    Primary model names take precedence over synonyms. Names are normalized
    via normalize_cell_line_name so callers must normalize their lookup keys too.

    Returns:
        Dict mapping normalized name to (CVCL accession, canonical model name).
    """
    models = pd.read_csv(str(model_list_path))
    mapping: dict[str, tuple[str, str]] = {}
    _add_primary_names(models, mapping)
    _add_synonyms(models, mapping)
    return mapping
