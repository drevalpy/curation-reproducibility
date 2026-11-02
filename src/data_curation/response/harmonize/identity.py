"""Resolve native response identifiers to the canonical ones drevalpy expects.

drevalpy keys samples by a dataset-specific unique ID and drugs by PubChem CID.
For the cell-line datasets the sample ID is the Cellosaurus accession. The fit
runs on each dataset's own identifiers, so this module builds the two
native -> canonical maps and reports what fails to resolve.

Samples take one of three routes, by format:

* **GDSC** carries Cell Model Passports ``SANGER_MODEL_ID``s, resolved through
  ``model_list.csv``.
* **CTRPv1/v2 and CCLE** carry names, resolved against Cell Model Passports
  names and synonyms first and Cellosaurus second. The Cellosaurus fallback
  rescues 50 of CTRPv2's 52 otherwise-dropped lines, and 6 of CCLE's.
* **PDX_Bruna and BeatAML2** are PDTCs and primary patient specimens with no
  accession at all, so the native identifier *is* the canonical sample ID.

Drugs always resolve through ``drug_mapping/<ds>.parquet``, whose ``DRUG_ID`` column
holds whatever native key the corresponding preparer emitted.
"""

import logging
from dataclasses import dataclass

import pandas as pd
from upath import UPath

from data_curation.cell_lines.cellosaurus import build_fallback_name_map, resolve_names
from data_curation.cell_lines.sidm_mapping import (
    _valid_cvcl,
    build_name_to_cellosaurus_map,
    build_sidm_to_cellosaurus_map,
)
from data_curation.core.errors import CurationError

logger = logging.getLogger(__name__)

#: Formats whose native cell-line identifier is a name rather than an accession.
NAME_KEYED_FORMATS = frozenset({"ctrpv1", "ctrpv2", "ccle"})

#: Formats whose samples are not cell lines, so there is no reference: the native ID is canonical.
IDENTITY_KEYED_FORMATS = frozenset({"bruna", "beataml"})

#: Formats whose native cell-line identifier is a DepMap ACH-* ID resolved via
#: the ``BROAD_ID`` column in the Cell Model Passports model list.
DEPMAP_KEYED_FORMATS = frozenset({"prism"})


@dataclass(frozen=True)
class SampleIdentity:
    """Native sample identifiers resolved to canonical sample IDs.

    Attributes:
        canonical: The canonical sample ID per native identifier (missing where unresolved).
        cell_line_names: ``{sample ID: cell line name}`` for the datasets whose samples are
            cell lines; empty for the others, which have no such name.
    """

    canonical: pd.Series
    cell_line_names: dict[str, str]


def resolve_samples(natives: pd.Index, fmt: str, model_list_path: UPath | None = None) -> SampleIdentity:
    """Map native sample identifiers to canonical sample IDs.

    Cell-line formats resolve to Cellosaurus accessions; identity-keyed formats
    keep their native identifier.

    Args:
        natives: The native identifiers, i.e. the fitted ``obs_names``.
        fmt: The dataset's response format.
        model_list_path: Cell Model Passports model list for CMP-based resolution.
            Required for ``gdsc`` and name-keyed formats; unused for
            identity-keyed formats (``bruna``, ``beataml``).

    Returns:
        The sample ID per native identifier (missing where unresolved) and, for
        cell-line formats, an ``{accession: cell_line_name}`` map covering
        everything resolved.

    Raises:
        ValueError: If the format needs a model list and none is given.
        CurationError: If *fmt* has no sample resolution route.
    """
    values = pd.Series(natives.astype(str), index=natives.astype(str))
    if fmt in IDENTITY_KEYED_FORMATS:
        return SampleIdentity(canonical=values, cell_line_names={})
    if model_list_path is None:
        msg = f"model_list_path is required for format '{fmt}'"
        raise ValueError(msg)
    if fmt in NAME_KEYED_FORMATS:
        return _resolve_by_name(values, model_list_path)
    if fmt == "gdsc":
        return _resolve_by_sidm(values, model_list_path)
    if fmt in DEPMAP_KEYED_FORMATS:
        return _resolve_by_depmap_id(values, model_list_path)

    msg = f"No sample resolution route for response format '{fmt}'"
    raise CurationError(msg)


def _resolve_by_sidm(values: pd.Series, model_list_path: UPath) -> SampleIdentity:
    """Resolve Cell Model Passports model IDs through the model list."""
    sidm_to_cvcl = build_sidm_to_cellosaurus_map(model_list_path)
    canonical = values.map(lambda sidm: sidm_to_cvcl[sidm][0] if sidm in sidm_to_cvcl else None)
    return SampleIdentity(canonical=canonical, cell_line_names=dict(sidm_to_cvcl.values()))


def _resolve_by_name(values: pd.Series, model_list_path: UPath) -> SampleIdentity:
    """Resolve cell-line names via Cell Model Passports, then Cellosaurus.

    The rescued Cellosaurus lines have no Cell Model Passports entry, so without
    carrying their names back they would end up with an empty ``cell_line_name``.
    """
    name_to_cvcl = build_name_to_cellosaurus_map(model_list_path)
    canonical, rescued = resolve_names(values, name_to_cvcl, build_fallback_name_map())
    names = dict(name_to_cvcl.values()) | {accession: name for name, accession in rescued.items()}
    return SampleIdentity(canonical=canonical, cell_line_names=names)


def _resolve_by_depmap_id(values: pd.Series, model_list_path: UPath) -> SampleIdentity:
    """Resolve DepMap ACH-* IDs via the BROAD_ID column of model_list.csv.

    The Cell Model Passports model list stores the DepMap identifier in a column
    called ``BROAD_ID``. Building a ``BROAD_ID -> (cellosaurus_id, name)`` map
    from that column mirrors the SIDM route but uses a different key.

    Args:
        values: DepMap ACH-* identifiers, indexed by themselves.
        model_list_path: Cell Model Passports model list.

    Returns:
        Accession and names for every resolved identifier.
    """
    models = pd.read_csv(str(model_list_path), low_memory=False)
    mapping: dict[str, tuple[str, str]] = {}
    for _, row in models.iterrows():
        broad = row.get("BROAD_ID")
        if pd.isna(broad):
            continue
        cvcl = _valid_cvcl(row)
        if cvcl is None:
            continue
        name = str(row["model_name"]) if pd.notna(row.get("model_name")) else ""
        mapping[str(broad).strip()] = (cvcl, name)

    canonical = values.map(lambda dep_id: mapping[dep_id][0] if dep_id in mapping else None)
    names = {cvcl: name for cvcl, name in mapping.values()}
    return SampleIdentity(canonical=canonical, cell_line_names=names)


def resolve_drugs(natives: pd.Index, drug_mapping: pd.DataFrame) -> pd.Series:
    """Map native drug identifiers to PubChem CIDs (or their pseudo-IDs).

    Args:
        natives: The native identifiers, i.e. the fitted ``var_names``.
        drug_mapping: The ``drug_mapping/<ds>.parquet`` table.

    Returns:
        The canonical identifier per native identifier, missing where unresolved.
    """
    mapping = dict(
        zip(drug_mapping["DRUG_ID"].astype(str).str.strip(), drug_mapping["pubchem_id"].astype(str), strict=True)
    )
    values = pd.Series(natives.astype(str), index=natives.astype(str))
    return values.map(mapping)


def drop_unresolved(canonical: pd.Series, *, axis_name: str, failure_fraction: float) -> pd.Series:
    """Drop unresolved identifiers, reporting and bounding the loss.

    Reporting only the count of identifiers understates it: 52 unmatched CTRPv2
    cell-line names were 5.9% of that dataset's measurements, which is exactly
    how the loss went unnoticed. Report the share of the axis too, and fail above
    *failure_fraction* so a mapping regression cannot pass silently.

    Args:
        canonical: Canonical label per native identifier, missing where unresolved.
        axis_name: ``"sample"`` or ``"drug"``, for the message.
        failure_fraction: Share of the axis above which this is an error.

    Returns:
        *canonical* restricted to the resolved entries.

    Raises:
        CurationError: If the unresolved share exceeds *failure_fraction*.
    """
    unresolved = canonical.isna()
    if not unresolved.any():
        return canonical

    lost = sorted(canonical.index[unresolved].astype(str))
    lost_fraction = float(unresolved.mean())
    logger.warning(
        "Unmapped %s identifiers: %d of %d (%.2f%%). First few: %s",
        axis_name,
        len(lost),
        len(canonical),
        100.0 * lost_fraction,
        lost[:10],
    )

    if lost_fraction > failure_fraction:
        msg = (
            f"{100.0 * lost_fraction:.2f}% of {axis_name} identifiers are unmappable, above the "
            f"{100.0 * failure_fraction:.0f}% threshold. {len(lost)} unmapped, first few: {lost[:10]}"
        )
        raise CurationError(msg)

    return canonical[~unresolved]
