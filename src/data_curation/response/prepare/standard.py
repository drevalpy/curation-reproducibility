"""Prepare a dataset delivered in the ``standard`` input format for curve fitting.

The dataset's own code has already reshaped its raw data into ``wells.parquet``,
one row per well in curate's column names; :mod:`data_curation.core.standard`
defines and validates that contract. ``sample`` and ``drug`` are the dataset's
own identifiers, ``concentration`` is in µM and ``intensity`` is viability
relative to the untreated control.

The optional columns map one to one onto the series spec: ``replicate``
(technical replicate) is the series key, ``experiment`` (biological replicate)
the experiment, and ``position`` orders the wells of a series. Every other
column is per-series metadata and is carried through under its own name.
"""

import logging

from upath import UPath

from data_curation.core import standard
from data_curation.response.prepare.wells import WELL_COLUMNS, SeriesSpec, Wells

logger = logging.getLogger(__name__)


def prepare_standard(raw_dir: UPath, ds_params: dict) -> Wells:  # noqa: ARG001
    """Read the standard well table and describe its series.

    Args:
        raw_dir: Raw response directory holding ``wells.parquet``.
        ds_params: Unused; the signature is shared across preparers.

    Returns:
        Wells keyed by the native ``sample`` and ``drug``, intensity clipped at zero.

    Raises:
        CurationError: If ``wells.parquet`` is missing or breaks the contract.
    """
    frame = standard.read("wells", raw_dir)
    assert frame is not None  # noqa: S101 - only the optional "published" kind can be absent
    logger.info("Read %d standard wells from %s", len(frame), standard.path("wells", raw_dir))

    present = set(frame.columns)
    spec = SeriesSpec(
        sample="sample",
        drug="drug",
        key=("replicate",) if "replicate" in present else (),
        experiment=("experiment",) if "experiment" in present else (),
        metadata=tuple(column for column in frame.columns if column not in WELL_COLUMNS),
        position="position" if "position" in present else None,
    )
    frame["intensity"] = frame["intensity"].clip(0, None)
    return Wells(frame[spec.columns], spec)
