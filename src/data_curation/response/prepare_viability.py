"""Turn raw drug response files into the well table ``drevalpy.curation.curate()`` reads.

This is the cheap half of what used to be one ``process_response`` stage. Each
dataset's preparer normalizes its raw measurements into well-level rows keyed by
the dataset's **own** identifiers and names the columns that group them into dose
series; :func:`~data_curation.response.prepare.wells.to_wells` renames those into
curate's ``sample``, ``drug``, ``concentration``, ``intensity``, ``experiment``,
``replicate`` and ``position`` columns, followed by the dataset's metadata columns.
How wells become numbered series is curate's business, in ``fit_curves``.

Nothing here resolves identity. That deliberate omission is the point of the
split: the fit that consumes this table then depends only on the table and the
``curvecurator`` params, so a change to Cellosaurus resolution or the PubChem
drug mapping can no longer invalidate a ~9 h curve fit it cannot affect.
"""

import logging
from collections.abc import Callable

from upath import UPath

from data_curation.core import layout
from data_curation.core.config import dataset_config
from data_curation.core.errors import CurationError
from data_curation.response.prepare.beataml import prepare_beataml
from data_curation.response.prepare.bruna import prepare_bruna
from data_curation.response.prepare.ccle import prepare_ccle
from data_curation.response.prepare.ctrp import prepare_ctrpv1, prepare_ctrpv2
from data_curation.response.prepare.gdsc import prepare_gdsc
from data_curation.response.prepare.prism import prepare_prism
from data_curation.response.prepare.wells import Wells, metadata_columns, to_wells

logger = logging.getLogger(__name__)

PREPARERS: dict[str, Callable[[UPath, dict], Wells]] = {
    "gdsc": prepare_gdsc,
    "ctrpv1": prepare_ctrpv1,
    "ctrpv2": prepare_ctrpv2,
    "ccle": prepare_ccle,
    "bruna": prepare_bruna,
    "beataml": prepare_beataml,
    "prism": prepare_prism,
}


def prepare(dataset: str, ds_params: dict, raw_dir: UPath) -> Wells:
    """Run the preparer for a dataset's response format.

    Args:
        dataset: Dataset name, used only in the error message.
        ds_params: The ``datasets.<dataset>.response`` params block.
        raw_dir: Raw response directory for the dataset.

    Returns:
        The well-level measurements, keyed by native identifiers, and their series spec.

    Raises:
        CurationError: If the configured format has no preparer.
    """
    fmt = ds_params["format"]
    if fmt not in PREPARERS:
        msg = f"Unknown response format '{fmt}' for dataset {dataset}"
        raise CurationError(msg)

    logger.info("Preparing %s viability data (format=%s)...", dataset, fmt)
    return PREPARERS[fmt](raw_dir, ds_params)


def main(dataset: str) -> None:
    """Write the native-identifier well table of a dataset."""
    raw_dir = layout.raw_dir("response", dataset)
    output_path = layout.wells(dataset)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    wells = to_wells(prepare(dataset, dataset_config(dataset, "response"), raw_dir))
    logger.info(
        "Prepared %d wells over %d native samples and %d native drugs; metadata columns %s",
        len(wells),
        wells["sample"].nunique(),
        wells["drug"].nunique(),
        metadata_columns(wells),
    )

    wells.to_parquet(str(output_path), index=False)
    logger.info("Wrote %s", output_path)
