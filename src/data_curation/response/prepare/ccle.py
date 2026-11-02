"""Prepare CCLE drug response data for curve fitting.

Source: Barretina et al. 2012, Nature 483:603 (Cancer Cell Line Encyclopedia),
supplementary table S11. The table stores one row per (cell line, compound) with
the dose series and the median activity packed into two comma-separated strings,
so it is exploded into long form here.

**Activity is not viability.** The published ``Activity Data (median)`` is a
percent-change-in-growth measure on roughly ``[-100, 0]``: ``0`` means the
treated well matched the untreated control and ``-100`` means growth was
abolished. Every other preparer emits fractional viability where ``1.0`` is
untreated, so the values are converted with ``activity / 100 + 1``, which is the
transform the original notebook applied before its own CurveCurator run. A third
of the resulting values exceed 1 (treated wells that grew *faster* than control)
and 0.16% of the raw values fall below ``-100``; the same ``clip(0, None)`` the
other preparers use bounds the latter at zero rather than dropping them.

Identifiers stay native - ``Primary Cell Line Name`` and ``Compound`` - and are
resolved in ``harmonize_response``. The cell lines are named rather than
accessioned, so they map like CTRP: Cell Model Passports name/synonym first,
Cellosaurus as the fallback.
"""

import logging

import pandas as pd
from upath import UPath

from data_curation.core.errors import CurationError
from data_curation.core.paths import find_file
from data_curation.response.prepare.wells import SeriesSpec, Wells

logger = logging.getLogger(__name__)

#: Glob addressing the supplement workbook in the raw response directory.
SUPPLEMENT_PATTERN = "*supplementary_tables*"

#: Sheet index and header offset of table S11 in the supplement workbook.
SHEET_INDEX = 11
SKIPROWS = 2

#: Native identifier columns.
SAMPLE_COLUMN = "Primary Cell Line Name"
DRUG_COLUMN = "Compound"

#: The table holds one aggregated series per (cell line, compound).
SPEC = SeriesSpec(sample=SAMPLE_COLUMN, drug=DRUG_COLUMN)

DOSE_COLUMN = "Doses (uM)"
ACTIVITY_COLUMN = "Activity Data (median)"

#: ``Activity Data (median)`` is a percent change in growth relative to the
#: untreated control, so 0 is full viability and -100 is none.
_PERCENT_SCALE = 100.0


def prepare_ccle(raw_dir: UPath, ds_params: dict) -> Wells:  # noqa: ARG001
    """Explode the packed CCLE dose series and convert activity to viability.

    Args:
        raw_dir: Raw response directory for the dataset.
        ds_params: Unused; the signature is shared across preparers.

    Returns:
        Long-form viability keyed by ``Primary Cell Line Name`` and ``Compound``, one series per pair.
    """
    raw_file = find_file(raw_dir, SUPPLEMENT_PATTERN)
    logger.info("Reading CCLE viability data from %s (sheet %d)", raw_file, SHEET_INDEX)

    long = explode_dose_series(read_activity_table(raw_file))
    long["intensity"] = (long["activity"] / _PERCENT_SCALE + 1.0).clip(0, None)
    return Wells(long[SPEC.columns], SPEC)


def read_activity_table(raw_file: UPath) -> pd.DataFrame:
    """Read table S11 of the Barretina supplement.

    Args:
        raw_file: The supplement workbook.

    Returns:
        One row per (cell line, compound), with the packed dose and activity
        strings still packed.
    """
    return pd.read_excel(str(raw_file), sheet_name=SHEET_INDEX, skiprows=SKIPROWS)


def explode_dose_series(df: pd.DataFrame) -> pd.DataFrame:
    """Unpack the comma-separated dose and activity strings into one row per dose.

    Args:
        df: Frame carrying :data:`DOSE_COLUMN` and :data:`ACTIVITY_COLUMN` as
            comma-separated strings of equal length.

    Returns:
        Long-form frame with float ``concentration`` and ``activity`` columns.

    Raises:
        CurationError: If a row's dose and activity series have different lengths,
            which would silently pair a dose with the wrong measurement.
    """
    frame = df.copy()
    for column in (DOSE_COLUMN, ACTIVITY_COLUMN):
        frame[column] = frame[column].astype(str).str.split(",")

    mismatched = frame[DOSE_COLUMN].str.len() != frame[ACTIVITY_COLUMN].str.len()
    if mismatched.any():
        msg = f"{int(mismatched.sum())} row(s) pair a dose series with a differently sized activity series"
        raise CurationError(msg)

    long = frame.explode([DOSE_COLUMN, ACTIVITY_COLUMN], ignore_index=True)
    long["concentration"] = pd.to_numeric(long[DOSE_COLUMN], errors="coerce")
    long["activity"] = pd.to_numeric(long[ACTIVITY_COLUMN], errors="coerce")
    return long
