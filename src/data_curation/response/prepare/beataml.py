"""Prepare BeatAML2 drug response data for curve fitting.

Source: Bottomly et al. 2022, Cancer Cell (BeatAML waves 1-4). Samples are
primary AML patient specimens keyed by ``dbgap_subject_id``, not cell lines, so
no Cellosaurus mapping applies and ``harmonize_response`` carries the native ID
straight through.

``normalized_viability`` is a *percentage* (0-100+, tails far above 100), unlike
the fractional viabilities of GDSC/CTRP, so it is divided by 100 here to keep
the intensity scale consistent across datasets.

One series is one inhibitor's dose series in one run, panel, plate and native
replicate. Series of the same run and specimen (``dbgap_dnaseq_sample``,
``dbgap_rnaseq_sample``) are technical replicates; another run or specimen of the
same patient is a biological replicate.
"""

import logging

import pandas as pd
from upath import UPath

from data_curation.core.paths import find_file
from data_curation.response.prepare.wells import SeriesSpec, Wells

logger = logging.getLogger(__name__)

#: Native identifier columns of the raw inhibitor table.
SAMPLE_COLUMN = "dbgap_subject_id"
DRUG_COLUMN = "inhibitor"

#: The raw ``replicate`` column, renamed so it cannot be mistaken for the fitted replicate index.
NATIVE_REPLICATE_COLUMN = "replicate_native"

SPEC = SeriesSpec(
    sample=SAMPLE_COLUMN,
    drug=DRUG_COLUMN,
    key=("run_index", "inhibitor_panel", "plate_num", NATIVE_REPLICATE_COLUMN),
    experiment=("run_index", "dbgap_dnaseq_sample", "dbgap_rnaseq_sample"),
    metadata=("run_index", "inhibitor_panel", "plate_num", "time_of_read", NATIVE_REPLICATE_COLUMN),
)

_PERCENT_SCALE = 100.0


def prepare_beataml(raw_dir: UPath, ds_params: dict) -> Wells:  # noqa: ARG001
    """Rescale BeatAML2's percent viability to a fraction.

    Args:
        raw_dir: Raw response directory for the dataset.
        ds_params: Unused; the signature is shared across preparers.

    Returns:
        Wells keyed by ``dbgap_subject_id`` and ``inhibitor``, grouped into series per run, panel, plate and
        native replicate.
    """
    raw_file = find_file(raw_dir, "*raw_inhibitor*")
    logger.info("Reading BeatAML2 viability data from %s", raw_file)
    df = pd.read_csv(str(raw_file), sep="\t", low_memory=False).drop_duplicates()
    df = df.rename(columns={"replicate": NATIVE_REPLICATE_COLUMN})

    df["intensity"] = (df["normalized_viability"] / _PERCENT_SCALE).clip(0, None)
    df["concentration"] = df["well_concentration"]
    return Wells(df[SPEC.columns], SPEC)
