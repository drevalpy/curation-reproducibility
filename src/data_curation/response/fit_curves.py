"""Fit dose-response curves, the expensive stage.

Hands the native-identifier well table to the drevalpy/curate Nextflow pipeline
(:mod:`data_curation.response.pipeline`), which numbers the dose series, fits one pooled curve
per (sample, drug) and, with ``--replicate_fits``, every series on its own - the same AnnData
``drevalpy.curation.curate()`` writes. CurveCurator is identifier-agnostic, so the AnnData and
its ``uns["replicate_fits"]`` table are keyed by exactly the labels that were fitted - the
dataset's own sample and drug identifiers - and ``harmonize_response`` renames them afterwards.

This stage only resolves configuration: the fit options (``curvecurator`` and ``replicates``
params), the dataset's ``normalize`` flag, and the ``execution`` block - the pipeline and its
revision (DVC params, they pin the fitting libraries), the Nextflow executor
(``execution.backend``, overridable with ``DATA_CURATION_BACKEND``) and the chunk size.

Resuming
--------
Each dataset runs from its own Nextflow launch directory under ``data/cache/nextflow/`` with
``-resume``. A fitting task is cached by the content of its chunk - its curves and fit options -
so a rerun after an interruption, or after the wells of a few curves changed, refits only the
chunks that changed. Chunks are only found again in the same work directory: local and SLURM
runs share the one in the launch directory, AWS Batch keeps its own on S3. Changing
``max_curves_per_job`` cuts the curves into different chunks and refits all of them, without
changing any result.
"""

import logging

from data_curation.core import layout
from data_curation.core.config import dataset_config, processing_config
from data_curation.response.pipeline import FitOptions, fit_dataset

logger = logging.getLogger(__name__)


def main(dataset: str) -> None:
    """Fit every curve of a dataset and write the native-keyed AnnData."""
    execution = processing_config("execution")
    options = FitOptions(
        min_doses=int(processing_config("replicates")["min_doses"]),
        fit_speed=str(processing_config("curvecurator")["fit_speed"]),
        max_chunk_size=int(execution["max_curves_per_job"]),
    )
    normalize = bool(dataset_config(dataset, "response").get("normalize", False))

    logger.info("Fitting %s (%s, normalize=%s)...", layout.wells(dataset), options, normalize)
    output = fit_dataset(dataset, normalize=normalize, options=options, execution=execution)
    logger.info("Wrote %s", output)
