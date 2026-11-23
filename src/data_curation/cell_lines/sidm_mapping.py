"""Shared cell-line identifier mapping utilities for process stages."""

import logging
import re

import pandas as pd
from upath import UPath

logger = logging.getLogger(__name__)

# A single RRID field must hold one accession. Anything joined by these is
# ambiguous and cannot be used as a cell-line identifier.
_ACCESSION_SEPARATORS = (";", ",", "|", " ")


def _clean_id(value: object) -> str:
    """Return one identifier as a string, dropping the ``.0`` pandas adds to integer columns with gaps."""
    text = str(value).strip()
    return text.removesuffix(".0") if text.endswith(".0") and text[:-2].isdigit() else text


def build_id_to_cellosaurus_map(model_list_path: UPath, column: str) -> dict[str, tuple[str, str]]:
    """Build native ID -> (Cellosaurus accession, model name) from one model list column.

    Every value of a multi-valued cell (``"ACH-002392;ACH-002303"``) maps to the model.
    When two models claim the same ID, the first one in the file wins.

    Args:
        model_list_path: Cell Model Passports ``model_list.csv``.
        column: ID column: ``"model_id"`` (SIDM), ``"BROAD_ID"`` (DepMap), ``"COSMIC_ID"`` or ``"CCLE_ID"``.

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
        for native in str(value).split(";"):
            key = _clean_id(native)
            if key and key not in mapping:
                mapping[key] = (cvcl, name)
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
