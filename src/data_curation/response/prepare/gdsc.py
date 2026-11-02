"""Prepare GDSC plate-level drug response data for curve fitting.

The raw files are well-level scan intensities, so viability has to be derived
per plate from the negative (untreated) and positive (blank) controls before
anything can be fitted.

One series is one drug's dose series in one library slot (``L4`` in the ``TAG``
``L4-D1-S``) of one plate (``BARCODE``). A slot screened twice on a plate (GDSC2's
repeated ``TAG`` blocks) is split by well ``POSITION`` in ``drevalpy.curation.curate()``. One experiment is one
cell line stock (``CELL_ID``) seeded on one day (the date of the plate's
``DATE_CREATED``, in UK time): its plates, libraries and slots are technical
replicates, and another stock or day is a biological replicate.

Identifiers stay native: ``SANGER_MODEL_ID`` and ``DRUG_ID`` go straight into the
``sample``/``drug`` columns. Resolving them to Cellosaurus accessions and
PubChem CIDs happens in ``harmonize_response``, after the fit.
"""

import logging

import pandas as pd
from upath import UPath

from data_curation.core.paths import find_file
from data_curation.response.gdsc_tags import library_wells
from data_curation.response.prepare.wells import SeriesSpec, Wells

logger = logging.getLogger(__name__)

#: Native identifier columns of the GDSC raw well table.
SAMPLE_COLUMN = "SANGER_MODEL_ID"
DRUG_COLUMN = "DRUG_ID"

#: Library slot of a well (e.g. ``L1``, ``L14``), extracted from its TAG.
SLOT_COLUMN = "lib_drug"

#: Day a plate was seeded: ``DATE_CREATED`` in UK time (GDSC1 stores midnight BST as ``23:00Z``).
SEEDING_DAY_COLUMN = "seeding_day"
_SITE_TIMEZONE = "Europe/London"

SPEC = SeriesSpec(
    sample=SAMPLE_COLUMN,
    drug=DRUG_COLUMN,
    key=("BARCODE", SLOT_COLUMN),
    experiment=("CELL_ID", SEEDING_DAY_COLUMN),
    metadata=(
        "BARCODE",
        "SCAN_ID",
        "DATE_CREATED",
        "SCAN_DATE",
        "RESEARCH_PROJECT",
        "DRUGSET_ID",
        "ASSAY",
        "DURATION",
        "SEEDING_DENSITY",
        "CELL_ID",
        "TAG",
    ),
    position="POSITION",
)

#: Raw columns read; the string-valued ones are read as categoricals to bound memory (GDSC2 is ~2 GB).
_STRING_COLUMNS = ["RESEARCH_PROJECT", "DATE_CREATED", "SCAN_DATE", "DRUGSET_ID", "ASSAY", "TAG", SAMPLE_COLUMN]
_NUMERIC_COLUMNS = ["BARCODE", "SCAN_ID", "CELL_ID", "SEEDING_DENSITY", "DURATION", "POSITION", "DRUG_ID", "CONC"]
_READ_COLUMNS = frozenset([*_STRING_COLUMNS, *_NUMERIC_COLUMNS, "INTENSITY"])

#: Regex extracting the library-position label (e.g. ``L1``, ``L14``) from TAG.
_LIB_DRUG_PATTERN = r"((L|R)\d+)(-D\d+)?-(S|C)"


def prepare_gdsc(raw_dir: UPath, ds_params: dict) -> Wells:
    """Control-normalize the GDSC well-level intensities.

    Args:
        raw_dir: Raw response directory for the dataset.
        ds_params: ``datasets.<ds>.response`` params, read for the control tags.

    Returns:
        Library wells keyed by ``SANGER_MODEL_ID`` and ``DRUG_ID``, grouped into series per plate and slot.
    """
    raw_file = find_file(raw_dir, "*public_raw_data*")
    logger.info("Reading raw viability data from %s", raw_file)
    df = pd.read_csv(
        str(raw_file),
        usecols=lambda column: column in _READ_COLUMNS,
        dtype=dict.fromkeys(_STRING_COLUMNS, "category"),
    )

    df = _drop_failed_wells(df)

    nc_tag = ds_params["negative_control_tag"]
    pc_tag = ds_params["positive_control_tag"]
    nc = df[df["TAG"] == nc_tag].groupby("SCAN_ID")["INTENSITY"].mean().rename("nc_mean")
    pc = df[df["TAG"] == pc_tag].groupby("SCAN_ID")["INTENSITY"].mean().rename("pc_mean")
    controls = pd.concat([nc, pc], axis=1)

    # Keep only library drug wells (excludes anchor/combination arms).
    treatments = library_wells(df).copy()
    del df
    treatments = treatments.merge(controls, on="SCAN_ID", how="left")

    denom = treatments["nc_mean"] - treatments["pc_mean"]
    treatments["intensity"] = ((treatments["INTENSITY"] - treatments["pc_mean"]) / denom).clip(0, None)
    treatments["concentration"] = treatments["CONC"]
    treatments[SLOT_COLUMN] = treatments["TAG"].astype(str).str.extract(_LIB_DRUG_PATTERN)[0]
    treatments[SEEDING_DAY_COLUMN] = _seeding_day(treatments["DATE_CREATED"])

    return Wells(treatments[SPEC.columns].reset_index(drop=True), SPEC)


def _seeding_day(created: pd.Series) -> pd.Series:
    """The UK calendar day (``YYYY-MM-DD``) of each ``DATE_CREATED`` timestamp, missing where unparsable."""
    values = pd.Series(created.astype("category").cat.categories)
    stamps = pd.to_datetime(values, utc=True, errors="coerce")
    days = stamps.dt.tz_convert(_SITE_TIMEZONE).dt.strftime("%Y-%m-%d")
    lookup = dict(zip(values, days.where(stamps.notna(), None), strict=True))
    return created.map(lookup).astype("category")


def _drop_failed_wells(df: pd.DataFrame) -> pd.DataFrame:
    """Remove wells GDSC marked as failed or that carry no drug.

    A drug that failed on a plate is recorded as ``UN-USED`` on one well, and the
    whole ``(SCAN_ID, POSITION)`` group it belongs to is unusable.
    """
    failed = df.loc[df["TAG"] == "UN-USED", ["SCAN_ID", "POSITION"]].drop_duplicates()
    if not failed.empty:
        wells = pd.MultiIndex.from_frame(df[["SCAN_ID", "POSITION"]])
        df = df[~wells.isin(pd.MultiIndex.from_frame(failed))]

    treatment_mask = df["TAG"].str.match(r"^(A|L|R)\d+", na=False).astype(bool)
    return df[~(df["DRUG_ID"].isna() & treatment_mask)]
