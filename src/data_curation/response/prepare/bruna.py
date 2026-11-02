"""Prepare PDX_Bruna drug response data for curve fitting.

Source: Bruna et al. 2016, Cell (breast cancer PDTX encyclopaedia). The raw
single-agent screen stores five doses per curve in wide form, along with a
per-plate ``Control`` (untreated) and ``Blank`` (no-cell) intensity.

The publication reports response as ``1 - (intensity - Blank) / (Control - Blank)``;
we keep viability, so the ``1 -`` is omitted.

Samples are PDTCs (short-term cultures). ``ID`` identifies the PDTC (i.e. the
individual mouse/passage) and is used as the sample identifier, while ``Model``
identifies the originating tumour and is retained only as metadata. These are
not cell lines, so no Cellosaurus mapping applies and ``harmonize_response``
carries the native ID straight through.

One series is one drug's five doses on one plate (``BARCODE``); plates of the
same PDTC are technical replicates.
"""

import logging

import pandas as pd
from upath import UPath

from data_curation.core.paths import find_file
from data_curation.response.prepare.wells import SeriesSpec, Wells

logger = logging.getLogger(__name__)

#: Native identifier columns of the wide single-agent table.
SAMPLE_COLUMN = "ID"
DRUG_COLUMN = "DRUG_ID"

SPEC = SeriesSpec(
    sample=SAMPLE_COLUMN,
    drug=DRUG_COLUMN,
    key=("BARCODE",),
    experiment=(SAMPLE_COLUMN,),
    metadata=("BARCODE",),
)

_N_DOSES = 5

#: Raw wide dose columns (``D1_CONC``, ``D1_INTENSITY``, ...) renamed to the stubs ``wide_to_long`` expects.
_DOSE_COLUMNS = {
    f"D{i}_{measure}": f"{measure}{i}" for i in range(1, _N_DOSES + 1) for measure in ("CONC", "INTENSITY")
}


def prepare_bruna(raw_dir: UPath, ds_params: dict) -> Wells:  # noqa: ARG001
    """Control-normalize the wide five-dose PDTC screen.

    Args:
        raw_dir: Raw response directory for the dataset.
        ds_params: Unused; the signature is shared across preparers.

    Returns:
        Wells keyed by ``ID`` and ``DRUG_ID``, one series per plate.
    """
    raw_file = find_file(raw_dir, "RawDataDrugsSingleAgents*")
    logger.info("Reading PDX_Bruna viability data from %s", raw_file)
    df = pd.read_csv(str(raw_file), sep="\t").rename(columns=_DOSE_COLUMNS)
    df["row"] = range(len(df))

    long = pd.wide_to_long(df, stubnames=["CONC", "INTENSITY"], i="row", j="MEASURE_NUM").reset_index()

    denom = long["Control"] - long["Blank"]
    long["intensity"] = ((long["INTENSITY"] - long["Blank"]) / denom).clip(0, None)
    long["concentration"] = long["CONC"]
    return Wells(long[SPEC.columns], SPEC)
