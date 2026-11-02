"""Prepare CTRPv1 and CTRPv2 drug response data for curve fitting.

Both are Broad CTD^2 screens of named cell lines, but they publish different
viability representations: CTRPv1 ships an already DMSO-normalized percent
viability, while CTRPv2 ships a log2 fold change against DMSO that has to be
linearized.

A CTRPv2 series is one compound's dose series on one assay plate of one
experiment (``experiment_id``, ``assay_plate_barcode``). Plates of the same
experiment are technical replicates, other experiments biological ones. The well
table has no well column, so its row order is the position that splits a series
duplicated on one plate.

CTRPv1 has no experiment-level metadata and its published data is already
aggregated, so every (cell line, compound) is a single series.

Identifiers stay native - ``ccl_name`` for the sample, ``cpd_name`` (v1) or
``master_cpd_id`` (v2) for the drug - and are resolved in
``harmonize_response`` after the fit. Keeping the name unresolved here is what
lets the Cellosaurus rescue of the 52 cell lines Cell Model Passports does not
carry cost no re-fit: their curves are fitted regardless.
"""

import logging

import numpy as np
import pandas as pd
from upath import UPath

from data_curation.core.paths import find_file
from data_curation.response.prepare.wells import SeriesSpec, Wells

logger = logging.getLogger(__name__)

#: Native identifier columns, per format.
SAMPLE_COLUMN = "ccl_name"
CTRPV1_DRUG_COLUMN = "cpd_name"
CTRPV2_DRUG_COLUMN = "master_cpd_id"

#: Row number of a well in the raw table, the only within-plate order CTRPv2 publishes.
ROW_COLUMN = "row_number"

#: ``meta.per_experiment`` columns carried into the series table.
EXPERIMENT_METADATA = (
    "run_id",
    "experiment_date",
    "culture_media",
    "cells_per_well",
    "growth_mode",
    "baseline_signal",
)

CTRPV1_SPEC = SeriesSpec(sample=SAMPLE_COLUMN, drug=CTRPV1_DRUG_COLUMN)

CTRPV2_SPEC = SeriesSpec(
    sample=SAMPLE_COLUMN,
    drug=CTRPV2_DRUG_COLUMN,
    key=("experiment_id", "assay_plate_barcode"),
    experiment=("experiment_id",),
    metadata=("experiment_id", "assay_plate_barcode", *EXPERIMENT_METADATA),
    position=ROW_COLUMN,
)

#: Columns read from the ~800 MB well table.
_CTRPV2_WELL_COLUMNS = frozenset(
    ["experiment_id", "assay_plate_barcode", "bsub_value_log2", "cpd_conc_umol", CTRPV2_DRUG_COLUMN]
)

_VALUE_SEPARATOR = ";"


def prepare_ctrpv1(raw_dir: UPath, ds_params: dict) -> Wells:  # noqa: ARG001
    """Read CTRPv1 viability, which ``cpd_avg_pv`` already reports fractionally.

    Args:
        raw_dir: Raw response directory for the dataset.
        ds_params: Unused; the signature is shared across preparers.

    Returns:
        Long-form viability keyed by ``ccl_name`` and ``cpd_name``, one series per pair.
    """
    raw_file = find_file(raw_dir, "*avg_pct_viability_data*")
    logger.info("Reading CTRPv1 viability data from %s", raw_file)
    df = pd.read_csv(str(raw_file), sep="\t", low_memory=False)

    df["intensity"] = df["cpd_avg_pv"].clip(0, None)
    df["concentration"] = df["cpd_conc_umol"]
    return Wells(df[CTRPV1_SPEC.columns], CTRPV1_SPEC)


def prepare_ctrpv2(raw_dir: UPath, ds_params: dict) -> Wells:  # noqa: ARG001
    """Linearize CTRPv2's log2 background-subtracted values and join metadata.

    Args:
        raw_dir: Raw response directory for the dataset.
        ds_params: Unused; the signature is shared across preparers.

    Returns:
        Wells keyed by ``ccl_name`` and ``master_cpd_id``, grouped into series per experiment and plate.
    """
    raw_file = find_file(raw_dir, "*data.per_cpd_well*")
    logger.info("Reading CTRPv2 viability data from %s", raw_file)
    df = pd.read_csv(
        str(raw_file),
        sep="\t",
        usecols=lambda column: column in _CTRPV2_WELL_COLUMNS,
        dtype={"assay_plate_barcode": "category"},
    )
    df[ROW_COLUMN] = np.arange(len(df))

    experiments = _per_experiment(pd.read_csv(str(find_file(raw_dir, "*meta.per_experiment*")), sep="\t"))
    cell_lines = pd.read_csv(str(find_file(raw_dir, "*meta.per_cell_line*")), sep="\t")

    # experiment_id -> master_ccl_id -> ccl_name
    df = df.merge(experiments, on="experiment_id", how="left")
    df = df.merge(cell_lines[["master_ccl_id", SAMPLE_COLUMN]], on="master_ccl_id", how="left")

    # bsub_value_log2 is log2 fold-change vs DMSO; convert to linear viability.
    df["intensity"] = np.power(2.0, df["bsub_value_log2"]).clip(0, None)
    df["concentration"] = df["cpd_conc_umol"]
    return Wells(df[CTRPV2_SPEC.columns], CTRPV2_SPEC)


def _per_experiment(experiments: pd.DataFrame) -> pd.DataFrame:
    """One row per ``experiment_id``: its cell line and its metadata.

    Some experiments are listed once per plating date; joining those rows as they
    are would duplicate every well, so their differing values are ``;``-joined.
    """
    present = [column for column in EXPERIMENT_METADATA if column in experiments.columns]
    aggregated = experiments.groupby("experiment_id", sort=False).agg(
        master_ccl_id=("master_ccl_id", "first"),
        **{column: (column, _join_distinct) for column in present},
    )
    return aggregated.reset_index().reindex(columns=["experiment_id", "master_ccl_id", *EXPERIMENT_METADATA])


def _join_distinct(values: pd.Series) -> str | None:
    """The distinct non-missing values of *values*, sorted and ``;``-joined."""
    distinct = sorted({str(value) for value in values.dropna()})
    return _VALUE_SEPARATOR.join(distinct) if distinct else None
