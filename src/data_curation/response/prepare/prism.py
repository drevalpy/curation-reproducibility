"""Prepare PRISM secondary screen drug response data for curve fitting.

Source: Corsello et al. 2019, Nature Cancer (doi:10.1038/s43018-019-0018-6).
PRISM Repurposing secondary screen: 1,448 compounds screened against ~489 cell
lines in an 8-step, 4-fold dilution starting from 10 uM.

The raw file ``secondary-screen-logfold-change.csv`` is a wide matrix of
log2 fold-change values (log2-transformed viability relative to DMSO,
batch-corrected with ComBat). Rows are cell lines identified by DepMap IDs
(``ACH-*``), columns are per-replicate condition identifiers decoded via
``secondary-screen-replicate-treatment-info.csv``.

Viability is recovered via ``2 ^ logfold_change``.

When MTS010 redo-screens are present for a drug (``screen_id == "MTS010"``),
they replace the original screen for that drug as recommended by the dataset
README.

One series is one compound's dose series on one detection plate (``detection_plate``)
for one cell-line pool (``PR500`` / ``PR300``; 20Q2 screens 96 cell lines in both).
The pool comes from the 20Q2 row names (``PR500_ACH-000007``); 19Q4 names only the
cell line, and only its MTS plate names carry its single pool, which then applies
to all its plates. The three replicate plates of a compound plate are technical
replicates.

Identifiers stay native: ``depmap_id`` for cell lines and ``broad_id`` for
drugs. Resolution to Cellosaurus accessions and PubChem CIDs happens in
``harmonize_response`` after the fit.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from upath import UPath

from data_curation.core.paths import find_file
from data_curation.response.prepare.wells import SeriesSpec, Wells

logger = logging.getLogger(__name__)

#: Native identifier columns.
SAMPLE_COLUMN = "depmap_id"
DRUG_COLUMN = "broad_id"

#: Cell-line pool of a measurement, e.g. ``PR500``.
POOL_COLUMN = "pool"

SPEC = SeriesSpec(
    sample=SAMPLE_COLUMN,
    drug=DRUG_COLUMN,
    key=("detection_plate", POOL_COLUMN),
    experiment=("compound_plate", POOL_COLUMN),
    metadata=("screen_id", "detection_plate", "compound_plate", "well", POOL_COLUMN),
    position="well",
)

#: The pool prefix of a 20Q2 LFC row name (``PR500_ACH-000007``) and the pool in a plate name.
_ROW_POOL_PATTERN = r"^(PR\d+)_"
_PLATE_POOL_PATTERN = r"_(PR\d+)[_.]"

#: screen_id value that supersedes the original screen for a drug.
_MTS010_SCREEN = "MTS010"

#: Only experimental treatments carry dose-response information.
_EXPERIMENTAL_TYPE = "experimental_treatment"


def _filter_mts010(treatment_info: pd.DataFrame) -> pd.DataFrame:
    """Keep MTS010 redo columns when available; otherwise keep originals.

    For drugs that have a ``screen_id == "MTS010"`` entry, all non-MTS010
    columns for that drug are dropped. Drugs without any MTS010 entry are
    kept as-is.

    Args:
        treatment_info: Replicate-treatment-info filtered to experimental
            treatments.

    Returns:
        Filtered treatment info with MTS010 preference applied.
    """
    has_mts010 = set(treatment_info.loc[treatment_info["screen_id"] == _MTS010_SCREEN, DRUG_COLUMN].unique())
    if not has_mts010:
        return treatment_info

    logger.info("Preferring MTS010 redo-screen for %d broad_id(s)", len(has_mts010))
    keep_mask = ~(treatment_info[DRUG_COLUMN].isin(has_mts010) & (treatment_info["screen_id"] != _MTS010_SCREEN))
    return treatment_info[keep_mask].copy()


def prepare_prism(raw_dir: UPath, ds_params: dict) -> Wells:  # noqa: ARG001
    """Convert PRISM secondary screen log-fold-change matrix to viability.

    Args:
        raw_dir: Raw response directory for the dataset.
        ds_params: Unused; the signature is shared across preparers.

    Returns:
        Wells keyed by ``depmap_id`` and ``broad_id``, grouped into series per detection plate and pool.
    """
    treatment_file = find_file(raw_dir, "*replicate-treatment-info*")
    logger.info("Reading PRISM treatment info from %s", treatment_file)
    treatment_info = pd.read_csv(str(treatment_file))

    if "perturbation_type" in treatment_info.columns:
        treatment_info = treatment_info[treatment_info["perturbation_type"] == _EXPERIMENTAL_TYPE].copy()
    else:
        logger.warning("No 'perturbation_type' column; using all treatment conditions")

    treatment_info = _filter_mts010(treatment_info)
    treatment_info = treatment_info.dropna(subset=["column_name"]).drop_duplicates("column_name")
    treatment_info["column_name"] = treatment_info["column_name"].astype(str)

    lfc_file = find_file(raw_dir, "*logfold-change*")
    header = pd.read_csv(str(lfc_file), nrows=0).columns
    valid = set(treatment_info["column_name"])
    keep_cols = [column for column in header[1:] if column in valid]
    logger.info("Reading %d of %d PRISM condition columns from %s", len(keep_cols), len(header) - 1, lfc_file)
    wanted = {header[0], *keep_cols}
    lfc = pd.read_csv(str(lfc_file), index_col=0, usecols=lambda column: column in wanted)

    cell_info_file = find_file(raw_dir, "*cell-line-info*")
    logger.info("Reading PRISM cell-line info from %s", cell_info_file)
    cell_info = pd.read_csv(str(cell_info_file))

    long = _stack(lfc, treatment_info.set_index("column_name").loc[lfc.columns], cell_info)
    long["intensity"] = np.power(2.0, long.pop("logfold_change")).clip(0, None)

    logger.info(
        "Prepared %d PRISM measurements: %d cell lines, %d drugs",
        len(long),
        long[SAMPLE_COLUMN].nunique(),
        long[DRUG_COLUMN].nunique(),
    )
    return Wells(long[SPEC.columns], SPEC)


def _stack(lfc: pd.DataFrame, conditions: pd.DataFrame, cell_info: pd.DataFrame) -> pd.DataFrame:
    """Long form of the measured cells of *lfc*, with each row's cell line and pool and each column's condition.

    Args:
        lfc: Log-fold-change matrix, rows named by ``row_name``, columns by condition.
        conditions: Treatment info indexed by condition, aligned to ``lfc.columns``.
        cell_info: Cell-line info mapping ``row_name`` to ``depmap_id``.

    Returns:
        One row per non-missing matrix cell.
    """
    row_names = lfc.index.astype(str)
    rows, cols = np.nonzero(lfc.notna().to_numpy())
    long = pd.DataFrame({"logfold_change": lfc.to_numpy()[rows, cols]})

    long[SAMPLE_COLUMN] = pd.Categorical(_depmap_ids(row_names, cell_info))[rows]
    for column in ("screen_id", "detection_plate", "compound_plate", "well", DRUG_COLUMN):
        long[column] = pd.Categorical(conditions[column].to_numpy())[cols]
    long["concentration"] = conditions["dose"].astype(float).to_numpy()[cols]

    # The 20Q2 row names carry the pool; 19Q4 names only the cell line, so its pool comes from the plate name.
    row_pool = pd.Series(row_names).str.extract(_ROW_POOL_PATTERN)[0].to_numpy()[rows]
    plate_pool = conditions["detection_plate"].astype(str).str.extract(_PLATE_POOL_PATTERN)[0].to_numpy()[cols]
    pool = np.where(pd.isna(row_pool), plate_pool, row_pool)
    # 19Q4 screened a single pool, which only its MTS plates name; its other plates are of the same pool.
    known = pd.unique(pool[pd.notna(pool)])
    if len(known) == 1:
        pool = np.full(len(pool), known[0], dtype=object)
    long[POOL_COLUMN] = pd.Categorical(pool)
    return long


def _depmap_ids(row_names: pd.Index, cell_info: pd.DataFrame) -> np.ndarray:
    """``depmap_id`` per LFC row; a row the cell-line info does not list keeps its own name."""
    if "row_name" not in cell_info.columns or SAMPLE_COLUMN not in cell_info.columns:
        return row_names.to_numpy()
    listed = cell_info.dropna(subset=[SAMPLE_COLUMN]).drop_duplicates("row_name", keep="last")
    mapping = pd.Series(listed[SAMPLE_COLUMN].astype(str).to_numpy(), index=listed["row_name"].astype(str))
    mapped = mapping.reindex(row_names).to_numpy()
    return np.where(pd.isna(mapped), row_names.to_numpy(), mapped)
